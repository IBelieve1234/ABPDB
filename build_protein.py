import os
import Bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import dgl
import dgl.data
from transformers import BertModel, BertTokenizer
import re
from dgl.nn import GINConv
import torch.backends.cudnn as cudnn
import higher 
import torch.optim as optim
import time
import Bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder
import random 

parser = PDBParser()
ppb=CaPPBuilder()

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def protein_preprocess(pdb_file='Q2G0W9'):
    #input: pdb_file
    #output(protein features):
    #   1. 1D residual seq
    #   2. distance matrix
    #   3. graph
    #   4. bond length
    #   5. angle

    pdb =  pdb_file + ".pdb"
    
    structure = parser.get_structure(pdb_file, './pdbs/'+pdb)

    #1. 1D residual seq
    model = structure[0]
    pp = ppb.build_peptides(structure)
    seq = pp[0].get_sequence()
    seq = " ".join("".join(str(seq).split()))
    seq = re.sub(r"[UZOB]", "X", seq)

    #234
    Res = list(model.get_residues())
    N_Res = len(Res)
    distance_matrix  = np.zeros((N_Res, N_Res))
    point1 = []
    point2 = []
    bond_length = []
    for i in range(N_Res):
        for j in range(i+1, N_Res):
            #2. distance matrix
            distance_matrix[i, j] = Res[i]['CA'] - Res[j]['CA']
            distance_matrix[j, i] = distance_matrix[i, j]
            if distance_matrix[i, j] < 7:
               #3. graph
               point1 = point1 + [i, j]
               point2 = point2 + [j, i]
               #4. bond length
               bond_length = bond_length + [distance_matrix[i, j], distance_matrix[j, i]]
    graph = (point1, point2)
    #validate the sequence len and the struc len is the same or not
    #5. angle
    angle = np.zeros((N_Res, 2), dtype='float32')
    model.atom_to_internal_coordinates()
    for r in range(N_Res):
        res = Res[r]
        angle[r][0] = res.internal_coord.get_angle("phi")
        angle[r][1] = res.internal_coord.get_angle("psi")
    angle[0][0] = 0
    angle[N_Res-1][1] = 0
    tmp = "".join(seq.split())
    #print("len s", len(tmp), N_Res)
    if len(tmp) != N_Res:
        #print("error")
        #exit(0)
        return None
    return seq, distance_matrix.astype(np.float32), graph, np.array(bond_length).astype(np.float32), angle.astype(np.float32)





