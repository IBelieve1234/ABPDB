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
 
#parse
args = parser.parse_args()

#global params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
center = torch.linspace(-np.pi, np.pi, steps=128).view(1, -1).to(device)



from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objs as go

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
    
def cluster_visualization(args):

    train_index_1 = list(np.load("./amp_pdb.npy", allow_pickle=True))
    abpdb_df = pd.read_csv("./ABPDB.csv", encoding="gbk") 
    abpdb_index = abpdb_df['ID'].tolist()

    train_index = list(set(train_index_1) & set(abpdb_index))
    print("train index len", len(train_index))

    #model def
    tokenizer, pretrained_lm = load_pretrained(model=args.base_model)
    pretrained_lm = pretrained_lm.to(device)
    mgin = MGIN(use_lm=args.use_lm)
    mgin = mgin.to(device)


    e=5
    print("begin loading weight from {}".format(e-1)+' epoch')
    load_prefix = './mypretrained/' + str(args.use_lm) + "_" + str(args.prt_coeff) + "_" + args.prt_trade + "_" + args.base_model
    mgin.gin.apply_func.weight.data = torch.load(load_prefix + "_" + 'gin.weight_{}'.format(e-1)+'.pt')
    mgin.gin.apply_func.bias.data = torch.load(load_prefix +  "_" + 'gin.bias_{}'.format(e-1)+'.pt')
    mgin.gin1.apply_func.weight.data = torch.load(load_prefix + "_" + 'gin1.weight_{}'.format(e-1)+'.pt')
    mgin.gin1.apply_func.bias.data = torch.load(load_prefix + "_" + 'gin1.bias_{}'.format(e-1)+'.pt')

    mgin.to(device)
    print("finish loading weight from {}".format(e-1)+' epoch')

    
    embeddings = []
    protein_names = []
    protein_lengths = []

    start = time.time()
    idx = 0 
    for index in train_index:

        seq, distance_matrix, graph, bond_length, angle = protein_preprocess(index)
        graph = dgl.graph(graph).to(device)
        angle[np.isnan(angle)] = 0.0
        scalar_angle = (torch.tensor(angle)/180).to(device)
        angle = scalar2vec(scalar_angle, center)
        bond_length = torch.tensor(bond_length).to(device)
        distance_matrix = torch.tensor(distance_matrix).to(device)
        with torch.backends.cudnn.flags(enabled=False):
            protein_emb = mgin.cluster_embeds(args,seq,tokenizer, pretrained_lm, angle, bond_length, graph, device)
            embeddings.append(protein_emb.detach().cpu().numpy()) 
            protein_names.append(index)  
            protein_lengths.append(len(seq.replace(" ", "")))


        if ((idx % args.interval) == 0) and (idx != 0):
            print("------------------------------------------------------")
            print(index)
            print("idx is：{}".format(idx))
            print("seq is:{}".format(seq))
            print("length is:{}".format(len(seq.replace(" ", ""))))
            print("time cost", time.time() - start)
            print("------------------------------------------------------")
            start = time.time()
        idx=idx+1
        #if(idx==500):#200
            #break
    print("this cluster idx is：{}".format(idx))

    max_embedding_size = max([emb.shape[1] for emb in embeddings])
    embeddings_matrix = np.zeros((len(embeddings), max_embedding_size))

    for i, emb in enumerate(embeddings):
        length = min(max_embedding_size, emb.shape[1])
        embeddings_matrix[i, :length] = emb[:length]


    dbscan = DBSCAN(eps=16.8, min_samples=4)
    cluster_labels = dbscan.fit_predict(embeddings_matrix)

    #t-SNE
    tsne = TSNE(n_components=2, perplexity=30,random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_matrix)

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    outlier_color = 'lightgrey'
    cluster_labels_set = set(cluster_labels) - {-1}
    color_sequence = px.colors.qualitative.Plotly
    cluster_color_dict = {label: color_sequence[i % len(color_sequence)] 
                      for i, label in enumerate(cluster_labels_set)}
    colors = [cluster_color_dict[label] if label != -1 else outlier_color for label in cluster_labels]

    trace = go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode='markers',
        marker=dict(color=colors, size=10, colorscale='Viridis', opacity=0.8),
        customdata=np.stack((protein_names, cluster_labels,protein_lengths,
                             [f"http://www.acdb.plus/ABPDB/abp-search.php?keywords={name}" for name in protein_names]
                             ), axis=-1),
        hovertemplate=
            "<b>Protein:</b> %{customdata[0]}<br>" +
            "<b>Cluster:</b> %{customdata[1]}<br>" +
            "<b>Length:</b> %{customdata[2]}<extra></extra>",  # <extra></extra> prevents the default hover info
        hoverinfo='none',  # Disable default hover info
        name = "Points"
    )

    layout = go.Layout(
        title='Peptide Embeddings',
        title_font=dict(size=24, color='black', family='Calibri'),  
        hovermode='closest'
    )

    fig = go.Figure(data=[trace], layout=layout)

    file_path = os.path.join(".", "All_protein_embeddings_2D.html")

    fig.write_html(
    file_path,
    include_plotlyjs='cdn',
    full_html=True
)

    with open(file_path, 'a') as f:
        f.write('''
    <script>
        var myPlot = document.getElementsByClassName('plotly-graph-div')[0];
        myPlot.on('plotly_click', function(data) {
            if (data.points[0].data.name === "Points") {
                var link = data.points[0].customdata[3];
                window.open(link, '_blank');
            }
        });

        var legendButton = document.createElement('button');
        legendButton.innerHTML = 'Legend';
        legendButton.style.position = 'fixed';
        legendButton.style.right = '20px';
        legendButton.style.bottom = '20px';
        legendButton.style.zIndex = 1000;
        legendButton.style.padding = '10px 20px';
        legendButton.style.backgroundColor = '#007bff';
        legendButton.style.color = 'white';
        legendButton.style.border = 'none';
        legendButton.style.borderRadius = '5px';
        legendButton.style.cursor = 'pointer';
        legendButton.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
        legendButton.style.transition = 'background-color 0.3s';

        legendButton.onmouseover = function() {
            legendButton.style.backgroundColor = '#0056b3';
        };
        legendButton.onmouseout = function() {
            legendButton.style.backgroundColor = '#007bff';
        };

        var legendDiv = document.createElement('div');
        legendDiv.style.position = 'fixed';
        legendDiv.style.right = '20px';
        legendDiv.style.bottom = '100px';
        legendDiv.style.zIndex = 999;
        legendDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
        legendDiv.style.padding = '15px';
        legendDiv.style.border = '1px solid #ddd';
        legendDiv.style.borderRadius = '10px';
        legendDiv.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.3)';
        legendDiv.style.display = 'none';
        legendDiv.style.transition = 'opacity 0.5s, transform 0.5s';
        legendDiv.style.opacity = '0';
        legendDiv.style.transform = 'translateY(20px)';
        legendDiv.style.maxWidth = '800px';  
        legendDiv.style.maxHeight = '600px';  
        legendDiv.style.width = 'auto'; 
        legendDiv.style.height = 'auto';  

        var legendImg = document.createElement('img');
        legendImg.src = './case_picture.png';
        legendImg.style.width = '100%';
        legendImg.style.height = 'auto';
        legendImg.style.borderRadius = '10px';
        legendImg.style.marginBottom = '10px';
        legendImg.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.2)';

        var legendText = document.createElement('p');
        legendText.innerHTML = 'Users can click on each point representing a peptide to view detailed information about the peptide, including its structure. Additionally, they can find the cluster containing the structures they are interested in and explore more similar peptide structures within that cluster. Hovering the mouse over a point will display the cluster number to which the peptide belongs, with different clusters shown in different colors. Users can also use the zoom and drag functionality in the top-right corner to freely explore and discover more clusters.';
        legendText.style.margin = '0';
        legendText.style.fontFamily = '"Roboto", sans-serif'; 
        legendText.style.fontSize = '14px';  
        legendText.style.lineHeight = '1.7';  
        legendText.style.color = '#555';  
        legendText.style.textAlign = 'justify';  
        legendText.style.textJustify = 'inter-word';  
        legendText.style.fontWeight = '400';  
        legendText.style.letterSpacing = '0.5px';  

        legendDiv.appendChild(legendImg);
        legendDiv.appendChild(legendText);
        document.body.appendChild(legendDiv);

        legendButton.onclick = function() {
            if (legendDiv.style.display === 'none' || legendDiv.style.opacity === '0') {
                legendDiv.style.display = 'block';
                setTimeout(function() {
                    legendDiv.style.opacity = '1';
                    legendDiv.style.transform = 'translateY(0)';
                }, 50); 
            } else {
                legendDiv.style.opacity = '0';
                legendDiv.style.transform = 'translateY(20px)';
                setTimeout(function() {
                    legendDiv.style.display = 'none';
                }, 500); 
            }
        };

        document.body.appendChild(legendButton);
    </script>
        ''')



if __name__ == '__main__':
    print(args)
    set_seed(args.seed)
    cluster_visualization(args)
