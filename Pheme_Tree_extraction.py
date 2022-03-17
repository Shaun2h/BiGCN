import torch
import json
import os
import numpy as np

# this serves to generate the REQUIRED file for pheme tree parsin..
# please understand there are a lot of sparse trees so this is a highly DIFFICULT dataset if you're gcning it
# python generation.py 1> generation_output.txt 2>&1



if not os.path.exists("phemethreaddump.json") or not os.path.exists("labelsplits.txt"):
    if not os.path.exists("phemethreaddump.json"):
        pheme_root = "all-rnr-annotated-threads" # POINT ME
        eventlist = os.listdir(pheme_root)
        allthreads = []
        print("beginning to load pheme dataset/setup files since they weren't done.")
        for event in eventlist:
            eventname = event.replace("-all-rnr-threads","")
            for classification in ["non-rumours","rumours"]:
                if classification == "non-rumours":
                    rootlabel = (0,"non-rumours",eventname)
                else:
                    rootlabel = (1,"rumour",eventname)
                    
                for somethread in os.listdir(os.path.join(pheme_root,event,classification)):
                    if "." ==somethread[0]:
                        continue
                    sourcetweetfile = os.path.join(pheme_root,event,classification,somethread,"source-tweets",somethread+".json")
                    
                    with open(sourcetweetfile,"r",encoding="utf-8") as opened_sourcetweetfile:
                        sourcedict = json.load(opened_sourcetweetfile)
                    sourcetweet = (sourcedict["text"], sourcedict["id_str"], sourcedict["id"])
                    # ["tweettext","tweetid","authid"]
                    threadtextlist = [sourcetweet]
                    tree = {sourcetweet[1]:[]}
                    
                    reactions = os.listdir(os.path.join(pheme_root,event,classification,somethread,"reactions"))
                    for reactionfilename in reactions:
                        if "." ==reactionfilename[0]:
                            continue
                        reactionsfilepath = os.path.join(pheme_root,event,classification,somethread,"reactions",reactionfilename)
                        with open(reactionsfilepath,"r",encoding="utf-8") as opened_reactiontweetfile:
                            reactiondict = json.load(opened_reactiontweetfile)
                            reactiontweet = (reactiondict["text"], reactiondict["id_str"], reactiondict["id"]) 
                            # ["tweettext","tweetid","authid"]

                            threadtextlist.append(reactiontweet)
                            replytarget = reactiondict["in_reply_to_status_id"]
                            if not reactiondict["id_str"] in tree: # if self isn't in tree.
                                tree[reactiondict["id_str"]] = [] # place self into treedict
                            
                            if str(replytarget)+".json" in reactions:
                                # print(replytarget)
                                # print(reactions)
                                if not str(replytarget) in tree:
                                    tree[str(replytarget)] = [] # if the response target hasn't been added but is a valid tweet in the dataset,
                                tree[str(replytarget)].append(reactionfilename.replace(".json",""))
                    
                    allthreads.append([threadtextlist,tree,rootlabel,sourcedict["id_str"]])
                    
        print("Parsed all files.")
        with open("phemethreaddump.json","w",encoding="utf-8") as dumpfile:
            json.dump(allthreads,dumpfile,indent=4)
        print("Thread dump completed (you can even delete the dataset now! wow!)")
    
    if not os.path.exists("PHEME_labelsplits.txt"):
        # falsify their silly little items.
        
        with open("phemethreaddump.json","r",encoding="utf-8") as dumpfile: # YES LEAVE IT IN ***** # edit this in later lol
            allthreads = json.load(dumpfile)
        with open("PHEME_labelsplits.txt","w") as labelfile:
        # no encoding here. Their code doesn't use encoding. it's safer to use the system default as a result.
        # unless.. it's utf-8 in that system default?
        # ANYWAY WRITE THIS DOWN AS A VULN.
            for thread in allthreads:
                threadtextlist,tree,rootlabel,source_id = thread
                
                if rootlabel[0]==0:
                    # note how i invert this, because his WEIBO reader is inverted labels from mine.
                    labelfile.write(str(source_id)+" 1\n")
                else:
                    labelfile.write(str(source_id)+" 0\n") 
                    
                    