import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

import dgl
import higher

import time
import random
import argparse

import csv
import pandas as pd

from utils import *
from build_protein import *

#args
parser = argparse.ArgumentParser(description='Bilevel Protein Pretraining')
parser.add_argument('--mode', choices=['prt', 'ft'], type=str, default='prt')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--interval', default=5, type=int)
parser.add_argument("--base_model", choices=['bert', 'xlnet'], type=str, default='bert')
#pretrain
parser.add_argument("--prt_lr", default=(1e-3), type=float)#1e-3
parser.add_argument('--prt_epochs', default=5, type=int)#5
parser.add_argument("--prt_wd", default=0.0, type=float)
parser.add_argument("--mask_ratio", default=0.15, type=float)
parser.add_argument("--prt_coeff", default=0.1, type=float)
parser.add_argument("--prt_trade", choices=['both', 'global', 'local'], default='both', type=str)
parser.add_argument('--use_lm', default=1, type=int)
parser.add_argument('--batch_size', default=100 ,type=int) 
#finetune
parser.add_argument('--task', choices=['loc', 'water', 'enzyme','amps'], type=str, default='amps')
parser.add_argument('--ft_mode', choices=['base', 'bilevel-h', 'bilevel-b', 'deepfri'], type=str, default='bilevel-b')
parser.add_argument("--ft_lr", default=1e-4, type=float)
parser.add_argument('--ft_epochs', default=8, type=int)
parser.add_argument("--ft_wd", default=0.0, type=float)
parser.add_argument("--target_protein_name",default='DRAMP00275',type=str)#DRAMP00275
#distance_threshold
parser.add_argument("--distance_threshold",default=13,type=int)
 
#parse
args = parser.parse_args()

#global params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
center = torch.linspace(-np.pi, np.pi, steps=128).view(1, -1).to(device)


from Bio import SeqIO
from Bio.Align import PairwiseAligner

def read_sequence_from_pdb(file_path):
    for record in SeqIO.parse(file_path, "pdb-atom"):
        return str(record.seq)
    #return None

def compare_sequences(seq1, seq2):
    """Needleman-Wunsch"""
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match = 4  
    aligner.mismatch = -1  
    aligner.open_gap_score = -2 
    aligner.extend_gap_score = -0.5  
    alignments = aligner.align(seq1, seq2)
    best_alignment = max(alignments, key=lambda alignment: alignment.score)
    max_score = len(seq1) * aligner.match
    percent_score = (max(best_alignment.score,0) / max_score)
    #print(f"： {percent_score:.2f}")
    return percent_score


from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align

def calculate_tm_score_and_sequence_similarity(pdb_file1, pdb_file2):
    """
    Calculate the TM-score between two PDB files, normalized by the length of the first protein chain.
    
    Parameters:
    - pdb_file1: Path to the first PDB file.
    - pdb_file2: Path to the second PDB file.
    
    Returns:
    - TM-score normalized by the length of the first protein chain.
    """
    # Load the structures from the PDB files
    structure1 = get_structure(pdb_file1)
    structure2 = get_structure(pdb_file2)
    # Get the first chain from each structure
    chain1 = next(structure1.get_chains())
    chain2 = next(structure2.get_chains())
    # Extract coordinates and sequences
    coords1, seq1 = get_residue_data(chain1)
    coords2, seq2 = get_residue_data(chain2)
    # Calculate the TM-score using tmtools
    result = tm_align(coords1, coords2, seq1, seq2)

    seq_similarity=0

    sequence1 = read_sequence_from_pdb(pdb_file1)
    sequence2 = read_sequence_from_pdb(pdb_file2)
    if sequence1 and sequence2:
        seq_similarity = compare_sequences(sequence1, sequence2)
    else:
        seq_similarity = 0


    return result.tm_norm_chain1,seq_similarity



from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objs as go

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.metrics.pairwise import euclidean_distances
import warnings


import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import os
import warnings
import numpy as np
import os
import warnings
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import time
import concurrent.futures
def cluster_visualization_target(args):

    warnings.filterwarnings("ignore", message="'HEADER' line not found; can't determine PDB ID.")
    train_index = list(np.load("./amp_pdb.npy", allow_pickle=True))
    print("train index len", len(train_index))

    data = np.load("./protein_data.npz", allow_pickle=True)
    embeddings_matrix = data['embeddings_matrix']

    #embeddings_matrix = embeddings_matrix.astype(np.float32)

    protein_names = data['protein_names']
    protein_lengths = data['protein_lengths']

    if args.target_protein_name not in train_index:
        print("target is not in dataset.")
        return

    target_index = train_index.index(args.target_protein_name)
    target_embedding = embeddings_matrix[target_index]

    distances = euclidean_distances(embeddings_matrix, [target_embedding])
    distance_threshold = args.distance_threshold

    while True:
        
        filtered_indices = np.where(distances <= distance_threshold)[0]
        filtered_embeddings = embeddings_matrix[filtered_indices]
        
        try:
            tsne = TSNE(n_components=2, perplexity=3, random_state=42)
            embeddings_2d = tsne.fit_transform(filtered_embeddings)
            break
        except ValueError as e:
            print(f"ValueError: {e}")
            print(f"current distance_threshold: {distance_threshold}")
            distance_threshold += 1
            print(f"add distance_threshold to: {distance_threshold}")
    
    distance_threshold = distance_threshold +1
    distance_thresholds = [distance_threshold + 1, distance_threshold + 4, distance_threshold + 9]
    # 1 4 9
    traces = []

    for distance_threshold in distance_thresholds:
        filtered_indices = np.where(distances <= distance_threshold)[0]
        filtered_embeddings = embeddings_matrix[filtered_indices]
        filtered_protein_names = protein_names[filtered_indices]
        filtered_protein_lengths = protein_lengths[filtered_indices]
        filtered_distances = euclidean_distances(filtered_embeddings, [target_embedding]).flatten()

        tsne = TSNE(n_components=2, perplexity=3, random_state=42)
        embeddings_2d = tsne.fit_transform(filtered_embeddings)

        new_target_index = np.where(filtered_indices == target_index)[0][0]
        center_point = embeddings_2d[new_target_index]
        center_name = filtered_protein_names[new_target_index]
        center_length = filtered_protein_lengths[new_target_index]

        other_points = np.delete(embeddings_2d, new_target_index, axis=0)
        other_names = np.delete(filtered_protein_names, new_target_index, axis=0)
        other_lengths = np.delete(filtered_protein_lengths, new_target_index, axis=0)
        other_distances = np.delete(filtered_distances, new_target_index, axis=0)

        other_tmScores = []
        other_nwScores = []

        for other_name in other_names:
            tmScore, nwScore = calculate_tm_score_and_sequence_similarity(
                "./pdbs/" + center_name + ".pdb", "./pdbs/" + other_name + ".pdb"
            )
            other_tmScores.append(tmScore)
            other_nwScores.append(nwScore)


        trace_center = go.Scatter(
            x=[center_point[0]],
            y=[center_point[1]],
            mode='markers+text',
            marker=dict(color='red', size=12, opacity=0.8),
            text=[f"Peptide: {center_name}, Length: {center_length}"],
            textposition="bottom center",
            hoverinfo='none',
            name="Target Antibacterial Peptide",
            visible=True if distance_threshold == distance_thresholds[0] else False
        )
        traces.append(trace_center)
        # candidate trace
        trace_others = go.Scatter(
            x=other_points[:, 0],
            y=other_points[:, 1],
            mode='markers',
            marker=dict(color='blue', size=10, opacity=0.5),
            customdata=np.stack((
                other_names,
                other_lengths,
                other_tmScores,
                other_nwScores,
                [f"http://www.acdb.plus/ABPDB/abp-search.php?keywords={name}" for name in other_names]
            ), axis=-1),
            hovertemplate=(
                "<b>Peptide_ID:</b> %{customdata[0]}<br>"
                "<b>Length:</b> %{customdata[1]}<br>"
                "<b>TM-score:</b> %{customdata[2]:.2f}<br>"
                "<b>NW-score:</b> %{customdata[3]:.2f}<br>"
                "<extra></extra>"
            ),
            hoverinfo='none',
            name="candidate antibacterial peptides",
            visible=True if distance_threshold == distance_thresholds[0] else False
        )
        traces.append(trace_others)

    # TM-score > 0.48 
    top_candidates = [
        (name, tm_score, nw_score, f"http://www.acdb.plus/ABPDB/abp-search.php?keywords={name}")
        for name, tm_score, nw_score in zip(other_names, other_tmScores, other_nwScores)
        if tm_score >= 0.48 
    ]
    top_candidates = sorted(top_candidates, key=lambda x: x[1], reverse=True)[:5]
    remaining_candidates = [
        (name, tm_score, nw_score, f"http://www.acdb.plus/ABPDB/abp-search.php?keywords={name}")
        for name, tm_score, nw_score in zip(other_names, other_tmScores, other_nwScores)
        if tm_score >= 0.48 
    ]
    remaining_candidates = sorted(remaining_candidates, key=lambda x: x[1], reverse=True)[5:10]

    top_candidates_html = f'''
    <div id="top_candidates" 
         style="margin-top: 20px; padding: 10px; font-family: Arial, sans-serif; font-size: 12px; color: #333; line-height: 1.5; position: fixed; bottom: 20px; right: 25px; cursor: move; z-index: 1000;">
        <h3 style="font-size: 10px; font-weight: bold; color: #444; margin-bottom: 10px;">Candidates based on structure</h3>
        <ul style="padding-left: 20px; margin: 0;">
    '''
    #  TM-score and NW-score
    for name, tm_score, nw_score, link in top_candidates:
        top_candidates_html += f'''
        <li style="margin-bottom: 8px; font-size: 10px;">
            <a href="{link}" target="_blank" style="text-decoration: none; color: #007BFF;">{name}</a><br>
            <b>TM-score:</b> {tm_score:.2f}<br>
            <b>NW-score:</b> {nw_score:.2f}
        </li>
        '''

    # "Show More"
    top_candidates_html += '''
        </ul>
        <button id="showMoreBtn" style="font-size: 10px; margin-top: 10px; padding: 5px 15px; background-color: #007BFF; color: white; border: none; border-radius: 5px; cursor: pointer; transition: background-color 0.3s;">
            Show More
        </button>
        <ul id="hiddenCandidates" style="display:none; padding-left: 20px; margin: 0;">
    '''

    for name, tm_score, nw_score, link in remaining_candidates:
        top_candidates_html += f'''
        <li style="margin-bottom: 8px; font-size: 10px;">
            <a href="{link}" target="_blank" style="text-decoration: none; color: #007BFF;">{name}</a><br>
            <b>TM-score:</b> {tm_score:.2f}<br>
            <b>NW-score:</b> {nw_score:.2f}
        </li>
        '''

    top_candidates_html += '''
        </ul>
    </div>

    <script>
        var dragElement = document.getElementById("top_candidates");
        var offsetX, offsetY;

        dragElement.onmousedown = function(e) {
            e.preventDefault();
            offsetX = e.clientX - dragElement.getBoundingClientRect().left;
            offsetY = e.clientY - dragElement.getBoundingClientRect().top;

            document.onmousemove = function(e) {
                var newLeft = e.clientX - offsetX;
                var newTop = e.clientY - offsetY;

                var viewportWidth = window.innerWidth;
                var viewportHeight = window.innerHeight;

                newLeft = Math.max(0, Math.min(viewportWidth - dragElement.offsetWidth, newLeft));
                newTop = Math.max(0, Math.min(viewportHeight - dragElement.offsetHeight, newTop));

                dragElement.style.left = newLeft + "px";
                dragElement.style.top = newTop + "px";
            };

            document.onmouseup = function() {
                document.onmousemove = null;
                document.onmouseup = null;
            };
        };

        // Show more candidates on button click
        document.getElementById("showMoreBtn").onclick = function() {
            var hiddenCandidates = document.getElementById("hiddenCandidates");
            if (hiddenCandidates.style.display === "none") {
                hiddenCandidates.style.display = "block";
                this.innerHTML = "Show Less";
            } else {
                hiddenCandidates.style.display = "none";
                this.innerHTML = "Show More";
            }
        };
    </script>
    '''
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Search Range: "},
        pad={"t": 50},
        steps=[dict(
            label=str(threshold),
            method="update",
            args=[{
                "visible": [
                    i == distance_thresholds.index(threshold) * 2 or
                    i == distance_thresholds.index(threshold) * 2 + 1
                    for i in range(len(traces))
                ]
            }, {
                "title": f"Peptide Embeddings in 2D Space around {args.target_protein_name} (Search Range: {threshold})"
            }]
        ) for threshold in distance_thresholds]
    )]

    layout = go.Layout(
        title=f'Peptide Embeddings in 2D Space around {args.target_protein_name}',
        hovermode='closest',
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2'),
        sliders=sliders
    )

    fig = go.Figure(data=traces, layout=layout)

    file_path = os.path.join(".", "example", "2D", f"{args.target_protein_name}.html")
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.write_html(
        file_path,
        include_plotlyjs='cdn',
        full_html=True
    )

    with open(file_path, 'a') as f:
        f.write(top_candidates_html)  # Top Candidates Based on TM Score
        f.write('''<script>
            var myPlot = document.getElementsByClassName('plotly-graph-div')[0];
            myPlot.on('plotly_click', function(data) {
                if (data.points[0].data.name === "candidate antibacterial peptides") {
                    var link = data.points[0].customdata[4];
                    window.open(link, '_blank');
                }
            });
        </script>''')

    print("saved！")

if __name__ == '__main__':
    #print
    print(args)
    #set seed
    set_seed(args.seed)

cluster_visualization_target(args)

"""
    protein_names = np.load("./amp_pdb.npy", allow_pickle=True)
    for protein_name in protein_names:
        args.target_protein_name = protein_name
        file_path = os.path.join(".", "example", "2D", f"{args.target_protein_name}.html")
        if os.path.exists(file_path):
            continue  
        print(f"{protein_name}...")
        cluster_visualization_target(args)
#"""
