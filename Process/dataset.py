import os
import numpy as np
import torch
import random
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel


class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph'),ispheme=False):
        if ispheme:
            # Pheme loader HERE.
            self.tddroprate = tddroprate
            self.budroprate = budroprate
            self.is_pheme_pointer = True
            with open("phemethreaddump.json","r",encoding="utf-8") as dumpfile: # YES LEAVE IT IN ***** # edit this in later lol
                allthreads = json.load(dumpfile)
            print("Tree extraction - PHEME. Will take a while: Extracts ALL trees at once. No need to do per dataload unlike theirs.")
            print("Drawback: Costs more memory, and really really long initial wait time.")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            tempytokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tempymodel = BertModel.from_pretrained("bert-base-uncased").to(device) # UHHH i mean whatever right? 
            # these boys will be garbage collected after this anyway..
            self.data_PHEME = []
            donecounter = 0
            for thread in allthreads:
                threadtextlist,tree,rootlabel,source_id = thread
                if not source_id in fold_x:
                    continue # it's not in this fold...
                # ["tweettext","tweetid","authid"]
                donecounter+=1
                if donecounter%100==0 and donecounter!=0:
                    print(str(donecounter)+"/"+str(len(fold_x))+" done.")
                    # break # 100 is enough for the test... 
                    # noteworthy is that their splits are for whole dataset splits, not like dataset into 5 splits then train test each.
                threaddict = {}
                pointerdict = {}
                counter = 0 
                for text in threadtextlist: # for easier reference.
                    threaddict[text[1]] = text[0] # ,text[2] #only append text, ignoring authors and ids. Can be edited later... # note here.
                                                # keep in mind you must edit collation function OR/AND the dataset class after.
                    pointerdict[text[1]] = counter #tweet id
                    counter+=1
                    # aim is to create an edge matrix like their preprocessing.
                fromrow = []
                torow = []
                for sender in tree:
                    for target in tree[sender]:
                        fromrow.append(pointerdict[sender])
                        torow.append(pointerdict[target])
                # edge matrix is created... for those that lack edges.. i'll raise this later in discussion.
                invertedpointer = dict(map(reversed, pointerdict.items()))
                

                allinputs_list = []  # allinputs_list acts as data["x"]
                for numbered in sorted(invertedpointer):
                    allinputs_list.append(threaddict[invertedpointer[numbered]])
                with torch.no_grad():
                    allinputs_list = tempymodel(**tempytokenizer(list(allinputs_list), padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)).last_hidden_state.cpu()
                # Note max source length is 512 which is the max for this bert model
                # but.. because of my gpu being small, it's 256. Anything above is TRUNCATED.
                data ={}
                with torch.no_grad():
                    data['root'] = tempymodel(**tempytokenizer(threaddict[source_id], padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)).last_hidden_state.cpu()
                data["rootindex"] = pointerdict[source_id]
                

                data["x"] = allinputs_list # you also need to convert this to a tensor.
                #torch.LongTensor()
                
                data["y"] = rootlabel[0]
                # print(rootlabel[0])  # rootlabel[0] acts as data["y"]
                # 0 = nonrumour, 1 = rumour
                data["edgeindex"] = np.array([fromrow,torow]) # imitating their dataloading method.
                # False "data" has been created. same structure as whatever they had.
                self.data_PHEME.append(data)
            print("Completed Loading Tree")
    
        else:
            self.is_pheme_pointer = False
            self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
            self.treeDic = treeDic
            self.data_path = data_path
            self.tddroprate = tddroprate
            self.budroprate = budroprate

    def __len__(self):
        if not self.is_pheme_pointer:
            return len(self.fold_x)
        else:
            return len(self.data_PHEME)

    def __getitem__(self, index):
        if not self.is_pheme_pointer:
            id =self.fold_x[index]
            data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
            print(data.files)
        else:
            # PHEME LOADER HERE
            data = self.data_PHEME[index]
            
        # EXPERIMENTS TO SEE THE DATA FORMAT SO WE CAN MIMIC.
        
        
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # print("root index: ",data['rootindex'])
        # print("Label:",data['y'])

        # print(data['root']) # always holds the root node's vector. however or whatever you encode it via.
        # print("Root shape:",data['root'].shape)
        
        # print(data['x'])
        # print("x shape:",data['x'].shape)
        # print("edge index shape: ",data['edgeindex'].shape)
        # edge index = edge matrix
        # row 1 FROM
        # row 2 TO
        
        # print(data['edgeindex'])
        # print(data['rootindex'])  
        # 0 = ['news', 'non-rumor'], 
        # 1 = ['false'], 
        # 2 = ['true'], 
        # 3 = ['unverified']
        
        edgeindex = data['edgeindex']
        
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
            
        if not self.is_pheme_pointer:
            # print(torch.LongTensor(bunew_edgeindex).shape)
            # print(torch.LongTensor(new_edgeindex).shape)
            # print(data["rootindex"])
            # print(torch.LongTensor(bunew_edgeindex).shape)
            # print(torch.LongTensor(new_edgeindex).shape)
            # print(torch.LongTensor(data["root"]).shape)
            # print(torch.tensor(data['x'],dtype=torch.float32).shape)
            return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))
        else:
            # print(torch.LongTensor(bunew_edgeindex).shape)
            # print(torch.LongTensor(new_edgeindex).shape)
            # print(data["root"].shape)
            # print(data["x"].shape) #you need to flatten and CHANGE the model shape in order to fit...
            return Data(x=data['x'].reshape(data["x"].shape[0],-1),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=data['root'].reshape(data["root"].shape[0],-1),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))
    
    
    
        


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
