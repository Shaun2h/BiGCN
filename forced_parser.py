import matplotlib.pyplot as plt
import ast
import json

"""
Overall utility.
Used on an output file. i.e you run model/Twitter/BiGCN_Twitter.py 1> output.txt 2>&1
and you run this on that file.
It gives you a comparison on the difference between reported scores and actual scores.
Hint: it's a minor difference that depends on the spread.
Sometimes theirs is lower, sometimes the actual is lower.
Really minor but please note.
"""

overalldict = {}
targetfiles = ["output.txt"]
for target in targetfiles:
    with open(target,"r", encoding="utf-8") as openedfile:
        triggerphrase = False
        newestcalling=""
        for line in openedfile:
            if triggerphrase:
                if "rawcounts:" in line: # ignore the rawcounts line.
                    continue
                # print(line)
                overalldict[newestcalling][-1].append(ast.literal_eval(line[4:-2]))
                # print(ast.literal_eval(line[4:-2]))
                triggerphrase = False
            else:
                if "results:" in line:
                    newestcalling = line.split("results:")[0]
                    if not newestcalling in overalldict:
                        overalldict[newestcalling] = []
                    their_report = line.split("results:")[1]
                    their_list = ast.literal_eval(their_report.strip())
                    reportedacc = float(their_list[0].split(":")[1])
                    reported_scores = ast.literal_eval("["+their_list[1].split(":")[1]+"]")
                    overalldict[newestcalling].append(reported_scores)
                    
                    triggerphrase = True

for event in overalldict:
    for epoch in range(len(overalldict[event])):
        their_acc = overalldict[event][epoch][0]
        their_prec = overalldict[event][epoch][1]
        their_rec = overalldict[event][epoch][2]
        their_f1 = overalldict[event][epoch][3]
        try:
            prec = overalldict[event][epoch][-1]["TP"]/(overalldict[event][epoch][-1]["TP"]+overalldict[event][epoch][-1]["FP"])
        except ZeroDivisionError:    
            prec = 0 
        try:
            rec = overalldict[event][epoch][-1]["TP"]/(overalldict[event][epoch][-1]["TP"]+overalldict[event][epoch][-1]["FN"])
        except ZeroDivisionError:
            rec = 0
        try:
            f1 = (prec*rec*2)/(prec+rec)
        except ZeroDivisionError:
            f1 = 0
        acc = (overalldict[event][epoch][-1]["TP"]+overalldict[event][epoch][-1]["TN"])/(overalldict[event][epoch][-1]["TP"]+overalldict[event][epoch][-1]["FP"]+overalldict[event][epoch][-1]["TN"]+overalldict[event][epoch][-1]["FN"])
        overalldict[event][epoch] = {}
        overalldict[event][epoch]["actual_acc"] = acc
        overalldict[event][epoch]["actual_precision"] = prec
        overalldict[event][epoch]["actual_recall"] = rec
        overalldict[event][epoch]["actual_f1"] = f1
        overalldict[event][epoch]["reported_acc"] = their_acc
        overalldict[event][epoch]["reported_precision"] = their_prec
        overalldict[event][epoch]["reported_recall"] = their_rec
        overalldict[event][epoch]["reported_f1"] = their_f1
        

with open("overall_total_outputfile.json","w",encoding="utf-8") as outputfile:
    json.dump(overalldict,outputfile,indent=4)
    
    