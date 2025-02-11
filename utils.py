import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import dgl
import dgl.data
from dgl.nn import GINConv
from dgl.nn import GraphConv
from transformers import BertModel, BertTokenizer, XLNetTokenizer, XLNetModel
import higher

import Bio
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder

import numpy as np
import re
import time
import random 

parser = PDBParser()
ppb=CaPPBuilder()

CE = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def adjust_learning_rate(optimizer, lr0, epoch, T):
    lr = lr0 * (1 + np.cos(np.pi * epoch * 1.0 / (T * 1.0))) / 2.0
    print("epoch {} use lr {}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_pretrained(layer_num=29, model='bert'):
    if model in ['bert']:
       tokenizer = BertTokenizer.from_pretrained("./Rostlab/prot_bert", do_lower_case=False)
       pretrained_lm = BertModel.from_pretrained("./Rostlab/prot_bert")
       modules = [pretrained_lm.embeddings, *pretrained_lm.encoder.layer[:29]]
    elif model in ['xlnet']:
       tokenizer = XLNetTokenizer.from_pretrained("./Rostlab/prot_xlnet", do_lower_case=False)
       xlnet_men_len = 512
       pretrained_lm = XLNetModel.from_pretrained("./Rostlab/prot_xlnet",mem_len=xlnet_men_len)
       modules = [pretrained_lm.word_embedding, *pretrained_lm.layer[:29]]
    pretrained_lm = pretrained_lm.eval()
    print("pretrained_lm", pretrained_lm)
    freeze = True
    if freeze:
       for module in modules:
           for param in module.parameters():
               param.requires_grad = False
    return tokenizer, pretrained_lm

class MGIN(nn.Module):
    def __init__(self, use_lm=1, node_emb_size=1280, model='bert'):
        super(MGIN, self).__init__()
        self.node_emb_size = node_emb_size
        lin = torch.nn.Linear(node_emb_size, node_emb_size)
        self.gin = GINConv(lin, 'sum')#.cuda()
        lin1 = torch.nn.Linear(node_emb_size, node_emb_size)
        self.gin1 = GINConv(lin1, 'sum')
        self.dis_nn = nn.Sequential(
                         nn.Linear(node_emb_size, int(node_emb_size/10)),
                         nn.ReLU(inplace=True),
                         nn.Linear(int(node_emb_size/10), 30)
                         )
        self.mask_nn = nn.Sequential(
                         nn.Linear(node_emb_size, int(node_emb_size/10)),
                         nn.ReLU(inplace=True),
                         nn.Linear(int(node_emb_size/10), 2),
                         nn.Tanh()
                         )
        self.model = model
        self.use_lm = use_lm


    def cluster_embeds(self, args, seq, tokenizer, pretrained_lm, node_feat, edge_feat, graph, device):
       # lm embeddings
        seq = tokenizer(seq, return_tensors='pt')
        seq['attention_mask'] = seq['attention_mask'].to(device)
        seq['input_ids'] = seq['input_ids'].to(device)
        seq['token_type_ids'] = seq['token_type_ids'].to(device)
        if args.use_lm:
            lm_embedding = pretrained_lm(**seq).last_hidden_state
            if self.model in ['bert']:
                node_feat0 = torch.cat([lm_embedding.squeeze(0)[1:-1,:], node_feat], dim=1)
            elif self.model in ['xlnet']:
                node_feat0 = torch.cat([lm_embedding.squeeze(0)[:-2,:], node_feat], dim=1)
        else:
            lm_embedding = F.one_hot(seq['input_ids'][0, 1:-1], num_classes=1024)
            node_feat0 = torch.cat([lm_embedding, node_feat], dim=1)
        node_feat = self.gin(graph=graph, feat = node_feat0.data, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        node_output = self.gin1(graph=graph, feat = node_feat, edge_weight = 1.0/(torch.pow(edge_feat, 2)+1e-6))
        if args.use_lm:
            node_output = node_output + node_feat0
        node_output = torch.mean(node_output, 0, True)
        node_output=node_output.reshape(1, -1)
        return node_output

class NodeEmb(nn.Module):
    def __init__(self, input_emb_size=1024, node_emb_size=1280):
        super(NodeEmb, self).__init__()
        self.L1 = nn.Linear(input_emb_size, node_emb_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.L2 = nn.Linear(node_emb_size, node_emb_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.node_emb_size = node_emb_size

    def forward(self, node_feat):
        if self.node_emb_size == node_feat.shape[1]:
            node_out = self.L1(node_feat) + node_feat
        else:
            node_out = self.L1(node_feat) + \
            torch.cat([node_feat, torch.zeros(node_feat.shape[0], self.node_emb_size-node_feat.shape[1]).to(device)], dim=1)
        node_out = self.relu1(node_out)
        node_out = self.L2(node_out) + node_out
        node_out = self.relu2(node_out)
        return node_out



class MutualInfo(nn.Module):
    def __init__(self, node_emb_size=1024):
        super(MutualInfo, self).__init__()
        self.seq_emb_layer = NodeEmb(input_emb_size=1024)
        self.struc_emb_layer = NodeEmb(input_emb_size=1280)
        self.softplus = nn.Softplus()
    def forward(self, seq_emb, stru_emb):
        seq_emb = self.seq_emb_layer(seq_emb)
        stru_emb = self.struc_emb_layer(stru_emb)
        distance = torch.mm(seq_emb, stru_emb.t())
        diag = torch.diag(distance)
        diag_loss = torch.mean(self.softplus(-diag))
        undiag = distance.flatten()[:-1].view(distance.shape[0] - 1, distance.shape[0] + 1)[:, 1:].flatten()
        undiag_loss = torch.mean(self.softplus(undiag))

        loss = diag_loss + undiag_loss
        return loss

def scalar2vec(angle, center):
    #angle seq_size*2
    #angle = torch.ones(4, 2)
    #center = torch.linspace(0, 2*np.pi, steps=3).view(1, -1)
    phi_angle = angle[:, 0].view(-1, 1)
    psi_angle = angle[:, 1].view(-1, 1)
    phi_vec = torch.exp(-10*torch.pow(phi_angle - center, 2))
    psi_vec = torch.exp(-10*torch.pow(psi_angle - center, 2))
    vec = torch.cat((phi_vec, psi_vec), dim=1)
    return vec

def remove_npy(array):
    new_array = []
    for a in array:
        tmp = a.split(".")[0]
        new_array.append(tmp)
    return new_array
