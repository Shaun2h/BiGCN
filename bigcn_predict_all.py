import sys, os
import argparse
import json
import pprint
import csv
from datetime import datetime
from time import time
sys.path.append(os.getcwd())
import torch as th
import torch.nn.functional as F
import numpy as np
import copy
from torch_scatter import scatter_mean
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel
model_load_name = "BiGCN4PHEME.m"

"""
This file is for creating an output csv file for a visualisation that is not available in this repository.
It also REQUIRES the pheme dataset's files within the directory to run.
"""

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



def process_thread(somethread,returnalt=False):
        """
        Process a Thread in Shaun's Format to their required format so we can use a similar loading func like they do.
        """
        threadtextlist,tree,rootlabel,source_id = somethread
        threaddict = {}
        pointerdict = {}
        counter = 0 
        for text in threadtextlist:
            threaddict[text[1]] = text[0]
            pointerdict[text[1]] = counter #tweet id
            counter+=1

        fromrow = []
        torow = []
        for sender in tree:
            for target in tree[sender]:
                fromrow.append(pointerdict[sender])
                torow.append(pointerdict[target])
        invertedpointer = dict(map(reversed, pointerdict.items()))
        allinputs_list = [] 
        for numbered in range(len(list(pointerdict.keys()))):
            allinputs_list.append(threaddict[invertedpointer[numbered]])

        data ={}
        with th.no_grad():
                allinputs_list = Bert_Tokeniser(list(allinputs_list), padding="max_length", max_length=256, truncation=True, return_tensors="pt")
                allinputs_list.to(device)
                allinputs_list = Bert_Embed(**allinputs_list)
                allinputs_list = allinputs_list[0].cpu()#.last_hidden_state
                allinputs_list = allinputs_list.cpu()

        with th.no_grad():
            data['root'] = Bert_Embed(**Bert_Tokeniser(threaddict[source_id], padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device))[0].cpu()#.last_hidden_state.cpu()
            

        data["rootindex"] = pointerdict[source_id]
        data["x"] = allinputs_list # you also need to convert this to a tensor.
        data["y"] = rootlabel[0]
        data["edgeindex"] = np.array([fromrow,torow]) # imitating their dataloading method.

        return data




def nodrop_get_td_bu_edges(edge_index_matrix):
    new_edgeindex = edge_index_matrix
    burow = list(edge_index_matrix[1])
    bucol = list(edge_index_matrix[0])
    bunew_edgeindex = [burow,bucol]
    return new_edgeindex, bunew_edgeindex
    
    
def get_td_bu_edges(tddroprate,budroprate,edge_index_matrix):
    if tddroprate > 0:
        row = list(edge_index_matrix[0])
        col = list(edge_index_matrix[1])
        length = len(row)
        poslist = random.sample(range(length), int(length * (1 - tddroprate)))
        # poslist = random.sample(range(length), int(length * (1)))
        poslist = sorted(poslist)
        row = list(np.array(row)[poslist])
        col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]
    else:
        new_edgeindex = edge_index_matrix

    burow = list(edge_index_matrix[1])
    bucol = list(edge_index_matrix[0])
    if budroprate > 0:
        length = len(burow)
        poslist = random.sample(range(length), int(length * (1 - budroprate)))
        # poslist = random.sample(range(length), int(length * (1)))
        poslist = sorted(poslist)
        row = list(np.array(burow)[poslist])
        col = list(np.array(bucol)[poslist])
        bunew_edgeindex = [row, col]
    else:
        bunew_edgeindex = [burow,bucol]
    return new_edgeindex, bunew_edgeindex

def run_model(tree, threadtextlist, source_id, rootlabel, model):
    """
    tree dictionary: {0:[1,2] 1: [3], 2:[], 3:[]}   0 -> 1 -> 3 and 0 -> 2
    list of string of tweets
    """
    data = process_thread((threadtextlist,tree,rootlabel,source_id))
    edgeindex = data['edgeindex']
    new_edgeindex, bunew_edgeindex = get_td_bu_edges(0,0,edgeindex)

    output_data =Data(x=data['x'].reshape(data["x"].shape[0],-1),
                edge_index=th.LongTensor(new_edgeindex),BU_edge_index=th.LongTensor(bunew_edgeindex),
                y=th.LongTensor([int(data['y'])]), root=data['root'].reshape(data["root"].shape[0],-1),
                rootindex=th.LongTensor([int(data['rootindex'])]))
    output_data.num_nodes = output_data.x.size(0) # prevents a torch scatter error.
    output_data.batch = th.tensor([0]*output_data.x.shape[0]) # imitate torch geometric batch   
    output_data.to(device)
    with th.no_grad():
        val_out = model(output_data)

    _, pred = val_out.max(dim=-1)
    # print("")
    # if pred==0:
        # print("rumour")
    # else:
        # print("non-rumour")
        

    return val_out,pred



if __name__ == '__main__':
    detailed_link_appending = False

    global device
    device = "cuda:"+str(0) if th.cuda.is_available() else "cpu"
    global Bert_Tokeniser 
    Bert_Tokeniser =  BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    global Bert_Embed 
    Bert_Embed = BertModel.from_pretrained("bert-base-multilingual-uncased").to(device)
    Bert_Embed.resize_token_embeddings(len(Bert_Tokeniser))
    model = Net(256*768,64,64,secretfinalfc=2).to(device)
    
    model.load_state_dict(th.load(model_load_name,map_location=device))
    model.eval()
    
    
    # run_model({0:[1,2],1:[3],2:[],3:[]},[("First Tweet was about how we did stuff",0),("Second Tweet was reminiscing about how we did stuff",1),("Third Tweet was on something he missed on the thing.",2),("Fourth Tweet was about how second tweet could also think about something else.",3)],0,[0],model)
    with open("phemethreaddump.json","rb") as dumpfile:
        loaded_threads = json.load(dumpfile)
    allthreads = []

    for thread in loaded_threads:
        threadtextlist,tree,rootlabel,source_id = thread
        # print(tree)
        # print(threadtextlist)
        # input()
        val_out,pred = run_model(tree,threadtextlist,source_id,rootlabel,model)
        prediction = "rumour" if pred==0 else "non-rumour"
        # print(rootlabel)
        actual_label = "rumour" if rootlabel[0]==0 else "non-rumour"
        allthreads.append([source_id,prediction,actual_label,val_out])
        

    mainpath = os.path.join("all-rnr-annotated-threads")
    path_reference_dict = {}
    for eventwrap in os.listdir(mainpath):
        if eventwrap[0] == ".":
            continue
        for item in os.listdir(os.path.join(mainpath,eventwrap,"rumours")):
            if item[0]==".":
                continue
            path_reference_dict[item] = os.path.join(mainpath,eventwrap,"rumours",item)
        for item in os.listdir(os.path.join(mainpath,eventwrap,"non-rumours")):
            if item[0]==".":
                continue
            path_reference_dict[item] = os.path.join(mainpath,eventwrap,"non-rumours",item)

    treelist = []
    for i in allthreads:
        treeid = i[0]
        predicted = i[1]
        actual = i[2]
        prediction_value = i[3]

        readable = ['false', 'true', 'unverified']
        tree_path = path_reference_dict[str(treeid)]
        list_of_reactions = os.listdir(os.path.join(tree_path,"reactions"))
        tree_dict = {}
        with open(os.path.join(tree_path,"source-tweets",str(treeid)+".json"),"r",encoding="utf-8") as opened_source:
            loaded_source = json.load(opened_source)
            text = loaded_source["text"]
            source_id = loaded_source["id"]
            links = []
            tree_dict[source_id] = [text,source_id,links,predicted,actual,loaded_source["created_at"],loaded_source["user"]["screen_name"],prediction_value]
            
        for item in list_of_reactions:
            if item[0] == ".":
                continue
            with open(os.path.join(tree_path,"reactions",item),"r",encoding="utf-8") as opened_reaction:
                
                reaction_dict = json.load(opened_reaction)
                reactiontext = reaction_dict["text"]
                reactionid = reaction_dict["id"]
                links = []
                reaction_target = reaction_dict["in_reply_to_status_id"]
                retweetedornot = reaction_dict["retweeted"]
                
                
                if not reactionid in tree_dict:
                    tree_dict[reactionid] = [reactiontext,reactionid,links,predicted,actual,reaction_dict["created_at"],reaction_dict["user"]["screen_name"],prediction_value]
                else:
                    tree_dict[reactionid] = [reactiontext,reactionid,tree_dict[reactionid][2],predicted,actual,reaction_dict["created_at"],reaction_dict["user"]["screen_name"],prediction_value]
                    
                    
                if not detailed_link_appending:
                    if not reaction_target in tree_dict:
                        tree_dict[reaction_target] = [None,reaction_target,[reactionid],None,None,None,None,None]
                    else:
                        tree_dict[reaction_target][2].append(reactionid)
                    tree_dict[reactionid][2].append(reaction_target)
                
                else:
                    if reaction_target!="null":
                        if not reaction_target in tree_dict:
                            tree_dict[reaction_target] = [None,reaction_target,[[reactionid,reaction_target,"Reply"]],None,None,None,None,None]
                        else:
                            tree_dict[reaction_target][2].append([reactionid,reaction_target,"Reply"])
                        tree_dict[reactionid][2].append([reactionid,reaction_target,"Reply"])
                            
                            
                    if retweetedornot:
                        if not reaction_target in tree_dict:
                            tree_dict[reaction_target] = [None,reaction_target,[[reactionid,reaction_target,"Retweet"]],None,None,None,None,None]
                        else:
                            tree_dict[reaction_target][2].append([reactionid,reaction_target,"Retweet"])
                        tree_dict[reactionid][2].append([reactionid,reaction_target,"Retweet"])
                        
        # print(tree_dict)
        treelist.append(tree_dict)
    
    with open(model_load_name+"_bigcn_all_predictions_dump.json","w",encoding="utf-8") as treedumpfile:
        # csvwriter = csv.writer(treedumpfile)
        fieldnames = ["tweet_id","handle","text","tweet_type","is_misinformation","tweet_time","edges","actual","scoring"]
        csvwriter = csv.DictWriter(treedumpfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        for treeid in treelist:
            for node in treeid:
                timestampval = str(treeid[node][5])
                if timestampval!="None":
                # "Wed Jan 07 11:11:33 +0000 2015" -> 2012-02-23 09:15:26 +00:00
                    date = datetime.strptime(timestampval,"%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d %H:%M:%S %z")
                else:
                    date = "None"
                
                csvwriter.writerow({"text":treeid[node][0], "tweet_id":treeid[node][1], "edges":treeid[node][2],"is_misinformation":treeid[node][3],"actual":treeid[node][4],"tweet_time":treeid[node][5],"handle":treeid[node][6],"scoring":treeid[node][7]})
    print("Dumped:",model_load_name+"_bigcn_all_predictions_dump.json")
