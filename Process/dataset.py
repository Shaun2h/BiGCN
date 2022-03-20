import os
import numpy as np
import torch
import random
import pickle
import pprint
import json
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel


# def reset_models():
global device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
global Bert_Tokeniser 
Bert_Tokeniser =  BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
global Bert_Embed 
Bert_Embed = BertModel.from_pretrained("bert-base-multilingual-uncased").to(device)
Bert_Embed.resize_token_embeddings(len(Bert_Tokeniser))
# reset_models()


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
                 data_path=os.path.join('..','..', 'data', 'Weibograph'),ispheme=False,picklefear=True):
        if ispheme:
            # Pheme loader HERE.
            self.tddroprate = tddroprate
            self.budroprate = budroprate
            self.is_pheme_pointer = True
            self.picklefear=picklefear
            if not picklefear: # we will use Pickle. (NOT SUITABLE FOR NON SELF USE. PUBLIC USERS BEWARE.)
                # this is just because i ran this like nonstop
                if not os.path.exists("overall_data_dump.pck"): # if your comp crashed, we will load what we've got. If not, we initialise.
                    with open("overall_data_dump.pck","wb") as datadump:
                        pickle.dump({},datadump,pickle.HIGHEST_PROTOCOL)
                    alt_dump_holder_data = {}
                else:
                    with open("overall_data_dump.pck","rb") as datadump: # CRASH PREVENTION.
                        alt_dump_holder_data = pickle.load(datadump)


                
                
                with open("phemethreaddump.json","rb") as dumpfile:
                    allthreads = json.load(dumpfile)
                    
                    
                print("Tree extraction - PHEME. Will take a Massive amount of time: Extracts ALL trees at once. No need to do per dataload unlike theirs.")
                print("Drawback: Costs more memory, and really really long initial wait time.")
                print("Picks up where it 'crashed off', if it hasn't finished compiling, or sees a new thing.")

                tempytokenizer = Bert_Tokeniser
                tempymodel = Bert_Embed
                # these boys will be garbage collected after this anyway..
                # print("number inside:",len(list(alt_dump_holder_data.keys())))
                donecounter = 0
                for thread in allthreads:
                    threadtextlist,tree,rootlabel,source_id = thread

                    if not source_id in fold_x: # if the source ID isn't actually required.. this time, ignore.
                        continue
        
                    if source_id in alt_dump_holder_data: # we're just checking to "update" our database.
                        continue
                    
                    alt_data = self.process_thread(thread,returnalt=True)
                    
                    alt_dump_holder_data[source_id] = alt_data
                    with open("overall_data_dump.pck","wb") as datadump: # shove the update in.
                        pickle.dump(alt_dump_holder_data,datadump,pickle.HIGHEST_PROTOCOL)
                    donecounter+=1

                    
                # data dump exists... so we load relevant points.
                self.data_PHEME = []
                with open("overall_data_dump.pck","rb") as datadump:
                    alt_dump_holder_data = pickle.load(datadump)
                for source_id in fold_x:
                    relevantdict = alt_dump_holder_data[source_id]
                    forceddict = {} # i'm paranoid about shallow copies... the memory cost is negligble.
                    forceddict["edgeindex"] = np.array(relevantdict["edgeindex"])
                    forceddict["y"] = relevantdict["y"]
                    forceddict["x"] = torch.tensor(np.array(relevantdict["x"]))
                    forceddict["rootindex"] = relevantdict["rootindex"]
                    forceddict["root"] = torch.tensor(np.array(relevantdict["root"]))
                    self.data_PHEME.append(forceddict)
                    # ["tweettext","tweetid","authid"]
                print("Completed Loading Tree - (ALL POSSIBLE)")
                
                
            else: # fear of pickle.
                self.tokeniser = Bert_Tokeniser
                self.model = Bert_Embed
                
                with open("phemethreaddump.json","rb") as dumpfile:
                    loaded_threads = json.load(dumpfile)
                    
                self.allthreads = {}

                for thread in loaded_threads:
                    threadtextlist,tree,rootlabel,source_id = thread
                    if source_id in fold_x:
                        self.allthreads[source_id] = thread
                self.fold_x = fold_x
                print("Skipping Pre-Tree Load Due to picklefears.")
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
            if not self.picklefear:
                return len(self.data_PHEME)
            else:
                return len(self.fold_x)

    def process_thread(self,somethread,tempytokenizer,tempymodel,returnalt=False):
        threadtextlist,tree,rootlabel,source_id = somethread

            # break
            # if donecounter>400:
                # break
            #break # 100 is enough for the test... 
            # noteworthy is that their splits are for whole dataset splits, not like dataset into 5 splits then train test each.
        threaddict = {}
        pointerdict = {}
        counter = 0 
        for text in threadtextlist: # for easier reference.
            threaddict[text[1]] = text[0] # ,text[2] #only append text, ignoring authors and ids. Can be edited later... # note here.
                                        # keep in mind you must edit collation function OR/AND the dataset class after.
            pointerdict[text[1]] = counter #tweet id
            counter+=1
            # ["tweettext","tweetid","authid"]

            # aim is to create an edge matrix like their preprocessing.
        # pprint.pprint(source_id)
        # pprint.pprint(threaddict)
        # print(len(list(threaddict.keys())),counter-1)
        # input()
        fromrow = []
        torow = []
        for sender in tree:
            for target in tree[sender]:
                fromrow.append(pointerdict[sender])
                torow.append(pointerdict[target])
        # edge matrix is created... for those that lack edges.. i'll raise this later in discussion.
        invertedpointer = dict(map(reversed, pointerdict.items()))
        allinputs_list = []  # allinputs_list acts as data["x"]
        for numbered in range(len(list(pointerdict.keys()))):
            allinputs_list.append(threaddict[invertedpointer[numbered]])
        # print(threaddict)
        data ={}
        alt_data = {}
        # print(allinputs_list)
        # print("Reattempting")
        with torch.no_grad():
            # try:
                allinputs_list = tempytokenizer(list(allinputs_list), padding="max_length", max_length=256, truncation=True, return_tensors="pt")
            # torch.set_printoptions(threshold = 1000000)
            # print(allinputs_list.input_ids)
            # print(len(allinputs_list.input_ids),len(list(threaddict.keys())))
                allinputs_list.to(device)
                allinputs_list = tempymodel(**allinputs_list)
                allinputs_list = allinputs_list[0].cpu()#.last_hidden_state
                allinputs_list = allinputs_list.cpu()
            # except RuntimeError as e:
                # print(e)
                # trashtoken =  BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
                # trashembed = BertModel.from_pretrained("bert-base-multilingual-uncased")
                # trashembed.resize_token_embeddings(len(trashtoken))
                # allinputs_list = trashtoken(list(allinputs_list), padding="max_length", max_length=256, truncation=True, return_tensors="pt")
                # allinputs_list = trashembed(**allinputs_list)
                # allinputs_list = allinputs_list.last_hidden_state
                # print("Failcase") 
                # input()
                # Unknown why some of the data just errors out.
                # return False,False
                
        
        # Note max source length is 512 which is the max for this bert model
        # but.. because of my gpu being small, it's 256. Anything above is TRUNCATED.
        
        with torch.no_grad():
            data['root'] = tempymodel(**tempytokenizer(threaddict[source_id], padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device))[0].cpu()#.last_hidden_state.cpu()
            
        
        
        
        if returnalt:
            alt_data["rootindex"] = pointerdict[source_id]
            alt_data["x"] = allinputs_list.detach().cpu().numpy().tolist()
            alt_data["root"] = data["root"].detach().cpu().numpy().tolist()
            alt_data["y"] = rootlabel[0]
            alt_data["edgeindex"] = np.array([fromrow,torow]).tolist() # imitating their dataloading method.
            return alt_data
        else:
            data["rootindex"] = pointerdict[source_id]
            data["x"] = allinputs_list # you also need to convert this to a tensor.
            data["y"] = rootlabel[0]
            data["edgeindex"] = np.array([fromrow,torow]) # imitating their dataloading method.
            # print(rootlabel[0])  # rootlabel[0] acts as data["y"]
            # 0 = nonrumour, 1 = rumour
            return data
        # False "data" has been created. same structure as whatever they had.
        
        

    def roundback(self,index):
        """
        backtrack and find out more information on the original data/tree etc that constituted this particular index.
        
        Return format:
        threadtextlist,tree,rootlabel,source_id
        threadtextlist -> [ ["tweettext","tweetid","authid"], ["tweettext","tweetid","authid"] ....  ]
        tree -> {id1:[<CHILDREN>], id2:[<CHILDREN>] ...  }   # note that it should be strings for the id iirc
        rootlabel -> 0 or 1,   -->    0 = nonrumour, 1 = rumour
        source_id -> the original nodes' id
        """
        if self.is_pheme_pointer:
            threadtextlist,tree,rootlabel,source_id = self.data_PHEME[index]
            
            return threadtextlist,tree,rootlabel,source_id
        else:
            print("\n\n")
            print("A request was made but.. this method only works for my self written pheme stuff...")
            print("No proper explainability can be performed on the Twitter15/Twitter16 processed datasets.")
            print("An accounting issue is also present, please see https://github.com/Shaun2h/Rumor_RvNN")
            print("Specifically, Resource Fabrication Folder, which attempts to replicate her processing")
            print("And More importantly, Run accountability.py in that folder to see issues in the tweet ids provided in the actual set")
            print("Versus total tree number and tweet number inflation (that cannot be explained) in the processed data.")
            print("\n\n")
        return
        

    def __getitem__(self, index):
        if not self.is_pheme_pointer:
            id =self.fold_x[index]
            data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
            # print(data.files)
        else:
            # PHEME LOADER HERE
            if not self.picklefear:
                data = self.data_PHEME[index] # data was already preloaded.
            else:
                # print("Original index:",index)
                # while True:
                    # print(self.fold_x[index])
                    data = self.process_thread(self.allthreads[self.fold_x[index]],self.tokeniser,self.model)
                    # if not data:
                        # print("OLD:",self.allthreads[self.fold_x[index]])
                        # print("errored out,trying:",index)
                        # index = random.randint(0,len(self.fold_x)-1) # try another index??
                        # print("NEW:",self.allthreads[self.fold_x[index]])
                    # else:

                        # break # obtained a workable sample...
                    
            
            
            
        # EXPERIMENTS TO SEE THE DATA FORMAT SO WE CAN MIMIC.
        
        
        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # print("root index: ",data['rootindex'])

        # print(data['root']) # always holds the root node's vector. however or whatever you encode it via.
        # print("Root shape:",data['root'].shape)
        
        # print(data['x'])
        # print("x shape:",data['x'].shape)
        # print(data['edgeindex'])
        # print("edge index shape: ",data['edgeindex'].shape)
        # edge index = edge matrix
        # row 1 FROM
        # row 2 TO
        
        
        # print("Label:",data['y'])
        # 0 = ['news', 'non-rumor'], 
        # 1 = ['false'], 
        # 2 = ['true'], 
        # 3 = ['unverified']
        
        edgeindex = data['edgeindex']
        try:
            if torch.max(torch.tensor(data["edgeindex"][0]),dim=0)[0]>=data["x"].shape[0]:
                print("(FROM) VIOLATION IN DATASET.",torch.max(torch.tensor(data["edgeindex"][0]),dim=0)[0],"      VS      ", data["x"].shape[0])
                print("There are more reported nodes in the edge matrix compared to the ACTUAL data matrix.")
                print("In my case, it was due to a node pointing to itself as the parent in the dataset, causing a double report of a node.")
                print("i.e the SAME NODE is reported TWO TIMES!")
                print("won't crash on cpu for some reason. BUT EVENTUALLY RESULTS IN CUDA DEVICE ASSERTION ERROR DUE TO INDEXING.")
        except IndexError:
            pass #can't violate.
        try:
            if torch.max(torch.tensor(data["edgeindex"][1]),dim=0)[0]>=data["x"].shape[0]:
                print("(TO) VIOLATION IN DATASET",torch.max(torch.tensor(data["edgeindex"][1]),dim=0)[0],"      VS      ", data["x"].shape[0])
                print("There are more reported nodes in the edge matrix compared to the ACTUAL data matrix.")
                print("In my case, it was due to a node pointing to itself as the parent in the dataset, causing a double report of a node.")
                print("i.e the SAME NODE is reported TWO TIMES!")
                print("won't crash on cpu for some reason. BUT EVENTUALLY RESULTS IN CUDA DEVICE ASSERTION ERROR DUE TO INDEXING.")
        except IndexError:
            pass #can't violate.
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            # poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = random.sample(range(length), int(length * (1)))
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
            # poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = random.sample(range(length), int(length * (1)))
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
            
            output_data = Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))

        else:
            # print(torch.LongTensor(bunew_edgeindex).shape)
            # print(torch.LongTensor(new_edgeindex).shape)
            # print(data["root"].shape)
            # print(data["x"].shape) #you need to flatten and CHANGE the model shape in order to fit...
            # print("overall number of tweets shape:",data["x"].shape)
            # print("original:",np.array(self.data_PHEME[index]["edgeindex"]))
            # print("BUEdgeIndex:",torch.LongTensor(new_edgeindex))
            # print("TDEdgeIndex:",torch.LongTensor(bunew_edgeindex))
            output_data = Data(x=data['x'].reshape(data["x"].shape[0],-1),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), root=data['root'].reshape(data["root"].shape[0],-1),
                    rootindex=torch.LongTensor([int(data['rootindex'])]))
        output_data.num_nodes = output_data.x.size(0) # prevents a torch scatter error.
        return output_data

    
    
        


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
