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

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=200)
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
    if len(torch.flatten(torch.tensor(x))) != 36864:
        return False
    return x.reshape(-1,effective_lens**2,1)[:,array_indices, :]

def load_data(data,ref_data,dnase_data,hic_data,chroms):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = get_args()
    data=data.numpy().astype(int)
    input=[]
    label=[]
    for i in range(data.shape[0]):
        chr=chroms[data[i][0]]
        s,e=data[i][1],data[i][2]
        input.append(torch.cat((ref_data[chr][s:e],dnase_data[chr][s:e]),dim=1))

        s,e=s//5,e//5
        temp=hic_data[chr][s+args.crop:e-args.crop,s+args.crop:e-args.crop].toarray()
        label.append(temp)
    input= torch.stack(input)
    
    intermediate = upper_tri(np.stack(label))
    if type(intermediate) == type(False):
        return False, False
    
    label= torch.tensor(intermediate)
    return input.float().to(device),label.float().to(device)


def main():

    args = get_args()
    model= build_pretrain_model_hic(args)
    model.cuda()
    criterion= torch.nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=1e-6)
    chroms = [str(i) for i in range(1, 23)]
    test_chrs=['3','11','17']
    valid_chrs=['9','16']
    cl='GM12878'

    train_chrs=list(set(chroms)-set(test_chrs)-set(valid_chrs))

    dnase_data, ref_data,hic_data=prepare_train_data(cl,chroms)

    train_dataset=hic_dataset(train_chrs)
    train_loader=DataLoader(train_dataset,batch_size=args.batchsize,shuffle=True)

    valid_dataset=hic_dataset(valid_chrs)
    valid_loader=DataLoader(valid_dataset,batch_size=args.batchsize,shuffle=False)

    testdataset=hic_dataset(test_chrs)
    test_loader=DataLoader(testdataset,batch_size=args.batchsize,shuffle=False)

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,steps_per_epoch=len(train_loader),
                        epochs=args.epochs, pct_start=0.025, div_factor=100,final_div_factor=0.01)
    best_criter=0
    for epoch in range(args.epochs):
        training_losses=[]
        model.train()
        for step, input_indices in enumerate(train_loader):
            t=time.time()
            input_data,input_label=load_data(input_indices,ref_data,dnase_data,hic_data,chroms)
            if type(input_data) == type(False):
                print("a wrong position")
                continue
            output = model(input_data)
            loss = criterion(output, input_label)
            loss = loss / args.accum_iter
            loss.backward()
            if ((step + 1) % args.accum_iter == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            cur_loss = loss.item()
            training_losses.append(cur_loss)
            print("Epoch:", '%04d' % (epoch + 1), "step:", '%04d' % (step + 1), "train_loss=",
                          "{:.7f}".format(cur_loss),
                          "time=", "{:.5f}".format(time.time() - t), "lr=", "{:.10f}".format(optimizer.param_groups[0]['lr'])
                          )

        validation_losses = []
        pccs=[]
        spearmans=[]
        model.eval()
        for step, input_indices in enumerate(valid_loader):
            t = time.time()
            with torch.no_grad():
                input_data, input_label = load_data(input_indices,ref_data,dnase_data,hic_data,chroms)
                if type(input_data) == type(False):
                    print("a wrong position")
                    continue
                if torch.sum(input_label) == 0:
                    continue
                output = model(input_data)
                loss = criterion(output, input_label)

                cur_loss = loss.item()
                validation_losses.append(cur_loss)
                out_array = output.cpu().data.detach().numpy().flatten()
                target_array = input_label.cpu().data.detach().numpy().flatten()

                pcc, _ = pearsonr(out_array.squeeze(), target_array.squeeze())
                spm,_=spearmanr(out_array.squeeze(), target_array.squeeze())
                pccs.append(pcc)
                spearmans.append(spm)
        print(np.mean(pccs))
        print(np.mean(spearmans))
        criter=np.mean(pccs)+np.mean(spearmans)

        train_loss = np.average(training_losses)
        print('Epoch: {} LR: {:.8f} train_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], train_loss))
        valid_loss = np.average(validation_losses)
        print('Epoch: {} LR: {:.8f} valid_loss: {:.7f}'.format(epoch, optimizer.param_groups[0]['lr'], valid_loss))

        with open('./models/log1_%s.txt' %cl, 'a') as f:
            f.write('Epoch: %s, LR: %s, train_loss: %s, valid_loss: %s\n' % (
            epoch, optimizer.param_groups[0]['lr'], train_loss, valid_loss))
        with open('./models/log1_%s.txt' %cl, 'a') as f:
            f.write('cl: %s, pcc: %s,spm:%s\n' % (cl, np.mean(pccs), np.mean(spearmans)))

        if criter > best_criter:
            best_criter = criter
            print('save model')
            torch.save(model.state_dict(),
                       './models/%s_%s.pt' % (cl, args.bins))
            if args.test:
                testing_losses = []
                pccs=[]
                spearmans=[]
                model.eval()
                with torch.no_grad():
                    for step, input_indices in enumerate(test_loader):
                        t = time.time()
                        input_data, input_label = load_data(input_indices,ref_data,dnase_data,hic_data,chroms)
                        if type(input_data) == type(False):
                            print("a wrong position")
                            continue
                        if torch.sum(input_label) == 0:
                            continue
                        output = model(input_data)
                        loss = criterion(output, input_label)
                        cur_loss = loss.item()
                        testing_losses.append(cur_loss)

                        out_array=output.cpu().data.detach().numpy().flatten()
                        target_array=input_label.cpu().data.detach().numpy().flatten()

                        pcc, _ = pearsonr(out_array.squeeze(), target_array.squeeze())
                        spm, _ = spearmanr(out_array.squeeze(), target_array.squeeze())
                        pccs.append(pcc)
                        spearmans.append(spm)
                print(np.mean(pccs))
                print(np.mean(spearmans))

                test_loss = np.average(testing_losses)
                with open('./models/log1_%s.txt' %cl, 'a') as f:
                    f.write('cl: %s, pcc: %s,spm:%s, loss: %s\n' % (cl, np.mean(pccs), np.mean(spearmans),test_loss))

if __name__=="__main__":
    main()
