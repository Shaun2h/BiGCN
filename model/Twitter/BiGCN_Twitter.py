import sys,os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import json
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import pprint

# python model\Twitter\BiGCN_Twitter.py PHEME 100 1> output_BIGCN_events.txt 2>&1


class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1=copy.copy(x.float())
        # print("TD")
        # print("x shape:",x.shape)
        # print("Edge_index shape:",edge_index.shape)
        # print("Edge_index:",edge_index)
        # try:
            # print("Maxes within edge:",th.max(edge_index,dim=1)[0])
            # for item in th.max(edge_index,dim=1)[0]:
                # if item>=x.shape[0]:
                    # print("VIOLATION:",item, "    VS     ",x.shape)
        # except IndexError:
            # print("no links.")
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        # print("BU")
        # print("x shape:",x.shape)
        # print("Edge_index shape:",edge_index.shape)
        # print("Edge_index:",edge_index)
        # try:
            # print("Maxes within edge:",th.max(edge_index,dim=1)[0])
            # for item in th.max(edge_index,dim=1)[0]:
                # if item>=x.shape[0]:
                    # print("VIOLATION:",item, "    VS     ",x.shape)
        # except IndexError:
            # print("no links.")
            
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,secretfinalfc=4):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc=th.nn.Linear((out_feats+hid_feats)*2,secretfinalfc)

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x,TD_x), 1)
        x=self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_GCN(treeDic, x_test, x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter,picklefear=True,commentary = ""):
    # Oh No! Naughty Injection via is_PHEME.
    is_PHEME= "PHEME" in dataname
    if not is_PHEME:
        model = Net(5000,64,64).to(device)
    else: # is pheme, i also change the class label possibility from 4 to 2.
        model = Net(256*768,64,64,secretfinalfc=2).to(device) # i'm assuming we're bert embedding this?? # yes shaun says lol.
    BU_params=list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
    optimizer = th.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':lr/5},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': lr/5}
    ], lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate,picklefear=picklefear)
    train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=0)
    test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=0)
     # jesus christ i'm not reinitiating this EVERY epoch. i'm moving this out to initiate only once. it shouldn't have any effects.
    for epoch in range(n_epochs):

        avg_loss = []
        avg_acc = []
        batch_idx = 0
        for _,Batch_data in enumerate(train_loader):
            
            # if Batch_data.rootindex.shape[0]!=3 and dataname=="PHEME":
                # continue
                Batch_data.to(device) # there appears to be some case where it's an empty... item? it's really confusing.
                # like there is no reason for that to happen. 

                out_labels= model(Batch_data)
                finalloss=F.nll_loss(out_labels,Batch_data.y)
                loss=finalloss
                optimizer.zero_grad()
                loss.backward()
                avg_loss.append(loss.item())
                optimizer.step()
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                     loss.item(),
                                                                                                     train_acc))
                batch_idx = batch_idx + 1
            # print("\nPERFORMANCE:\n",Batch_data,"\n",Batch_data.rootindex.shape[0],"\n")
            # print("Output shape:",out_labels.shape)
            
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        print("\n-------------------------------------------------------Initiating Test:--------------------------------------------------\n")
        rawcounts = {1:{"TP":0,"FP":0,"FN":0,"TN":0},2:{"TP":0,"FP":0,"FN":0,"TN":0},3:{"TP":0,"FP":0,"FN":0,"TN":0},4:{"TP":0,"FP":0,"FN":0,"TN":0}}

        for _, Batch_data in enumerate(test_loader):
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            # print(val_out.shape)
            # print(Batch_data.y.shape)
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4, rawdict= evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
            for classy in rawdict:
                for resultant_type in rawdict[classy]:
                    rawcounts[classy][resultant_type] = rawdict[classy][resultant_type] + rawcounts[classy][resultant_type]
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print(commentary,'results:', res)
        print(commentary,"rawcounts:")
        pprint.pprint(rawcounts)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN'+commentary, dataname,rawcounts,epoch)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4

if __name__=="__main__":
    lr=0.0005
    weight_decay=1e-4
    patience=10
    n_epochs=200
    batchsize=128 #theirs is 128.. for pheme with a bert embedding though.. it's not really possible without a massive space.
    TDdroprate=0.2
    BUdroprate=0.2
    datasetname=sys.argv[1] #"Twitter15"、"Twitter16"
    if datasetname=="PHEME":
        batchsize=12 # If pheme, adjust to this. explodes most lower end gpus otherwise. ( will already explode small ones)

    iterations=int(sys.argv[2])

    try:
        eventsplitter = sys.argv[3]=="True" or sys.argv[3]=="true" or sys.argv[3]=="yes" or sys.argv[3]=="Yes"
    except IndexError: # index out of range. picklefear not specified.
        print("3rd argument missing. If it was pheme, defaults to random folds of data. Else, nothing.")
        eventsplitter = False
    try:
        picklefear = sys.argv[3]!="pickle" # default to no pickle
    except IndexError:
        print("4th argument missing. If it was pheme, defaults to NOT pickle. Else, nothing.")
        picklefear = True
        
        
    model="GCN"
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    test_accs = []
    NR_F1 = []
    FR_F1 = []
    TR_F1 = []
    UR_F1 = []
    if not eventsplitter: # So just random folder:
        if datasetname!="PHEME":
            treeDic=loadTree(datasetname)
        else:
            treeDic = {} # Won't be using this...
        for iter in range(iterations):
            fold0_x_test, fold0_x_train, \
            fold1_x_test,  fold1_x_train,  \
            fold2_x_test, fold2_x_train, \
            fold3_x_test, fold3_x_train, \
            fold4_x_test,fold4_x_train = load5foldData(datasetname)
            train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                                       fold0_x_test,
                                                                                                       fold0_x_train,
                                                                                                       TDdroprate,BUdroprate,
                                                                                                       lr, weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       picklefear,
                                                                                                       "1")
            train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(treeDic,
                                                                                                       fold1_x_test,
                                                                                                       fold1_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       picklefear,
                                                                                                       "2")
            train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(treeDic,
                                                                                                       fold2_x_test,
                                                                                                       fold2_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       picklefear,
                                                                                                       "3")
            train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(treeDic,
                                                                                                       fold3_x_test,
                                                                                                       fold3_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       picklefear,
                                                                                                       "4")
            train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(treeDic,
                                                                                                       fold4_x_test,
                                                                                                       fold4_x_train,
                                                                                                       TDdroprate,BUdroprate, lr,
                                                                                                       weight_decay,
                                                                                                       patience,
                                                                                                       n_epochs,
                                                                                                       batchsize,
                                                                                                       datasetname,
                                                                                                       iter,
                                                                                                       picklefear,
                                                                                                       "5")
            test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
            NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
            FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
            TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
            UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
        print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
    else:
        with open("Eventsplit_details.txt","r") as eventsplitfile:
            eventsplits = json.load(eventsplitfile)
        treeDic = {} # Won't be using this...
        for event in eventsplits:
            print("-"*25,event,"-"*25)
            testfold = eventsplits[event]
            trainfold = []
            for notevent in eventsplits:
                if notevent==event:
                    continue
                trainfold.extend(eventsplits[notevent])

            train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(treeDic,
                                                                                                    testfold,
                                                                                                    trainfold,
                                                                                                    TDdroprate,BUdroprate,
                                                                                                    lr, weight_decay,
                                                                                                    patience,
                                                                                                    n_epochs,
                                                                                                    batchsize,
                                                                                                    datasetname+" "+event,
                                                                                                    0,
                                                                                                    picklefear,
                                                                                                    "event - "+event)
            test_accs.append(accs0)
            NR_F1.append(F1_0)
            FR_F1.append(F2_0)
            TR_F1.append(F3_0)
            UR_F1.append(F4_0)
            print("LOOP COMPLETED: EVENT- ",event)
        print("OVERALL RESULTS FOR :",event)
        print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
        sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))
