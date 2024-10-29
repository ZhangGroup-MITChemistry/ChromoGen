## Most codes are directly from EPCOT "https://github.com/liu-bioinfo-lab/EPCOT/tree/main/COP"

from hic.model import build_pretrain_model_hic
from util import prepare_train_data
import argparse
from torch.optim import lr_scheduler
import os, pickle, time
import random
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
import argparse
from hic_dataset import hic_dataset
from scipy.stats import pearsonr,spearmanr
import subprocess

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=200)
    parser.add_argument('--curr_chr', type=str, default='1')
    parser.add_argument('--crop', type=int, default=4)
    parser.add_argument('--nheads', default=4, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--enc_layers', default=1, type=int)
    parser.add_argument('--dec_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--shuffle_dataset', default=True, action='store_false', help='model testing')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--test', default=True, action='store_false', help='model testing')
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--ac_data', type=str, default='dnase_norm')
    parser.add_argument('--fine_tune', default=False, action='store_true')
    parser.add_argument('--trunk',  type=str, default='transformer')
    parser.add_argument('--pretrain_path',type=str,default='none',help='path to the pre-training model')
    parser.add_argument('--finetune_path',type=str,default='none',help='path to the fine_tuning model')
    args = parser.parse_args()
    return args
def get_args():
    args = parser_args()
    return args

def upper_tri(x):
    args = get_args()
    effective_lens = args.bins - 2 * args.crop
    triu_tup = np.triu_indices(effective_lens)
    array_indices = np.array(list(triu_tup[1] + effective_lens * triu_tup[0]))
    return x.reshape(-1,effective_lens**2,1)[:,array_indices, :]

#def load_data(data,ref_data,dnase_data,hic_data,chroms):
def load_data(data,ref_data,dnase_data,chroms):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    data=data.numpy().astype(int)
    input=[]
    for i in range(data.shape[0]):
        chr = args.curr_chr
        s,e=data[i][1],data[i][2]
        input.append(torch.cat((ref_data[chr][s:e],dnase_data[chr][s:e]),dim=1))

    input= torch.stack(input)

    return input.float().to(device)#,label.float().to(device)


def main():

    args = get_args()
    model= build_pretrain_model_hic(args)
    model.cuda()
    chroms = [args.curr_chr]#[str(i) for i in range(1, 23)]
    cl='GM12878'

    train_chrs = chroms

    dnase_data, ref_data=prepare_train_data(cl,chroms)
    train_dataset=hic_dataset(train_chrs)
    train_loader=DataLoader(train_dataset,batch_size=args.batchsize,shuffle=False)

    for epoch in range(args.epochs):
        model.eval()
        for step, input_indices in enumerate(train_loader):
            print("input_indeces", input_indices)
            input_data = load_data(input_indices,ref_data,dnase_data,chroms)
            output = model(input_data)
            torch.save(output.detach(), "chr_%s_%d.pt"%(args.curr_chr, step))
            subprocess.call(["mv chr_%s_%d.pt ./node_embedding/run_scripts_%s"%(args.curr_chr, step, args.curr_chr)],shell=True,stdout=subprocess.PIPE)

if __name__=="__main__":
    main()
