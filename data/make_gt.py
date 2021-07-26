import json
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='label')
parser.add_argument('--trn_name_file', default='VATEX/public_split/trn_names.npy')
parser.add_argument('--verb_freq_file', default='verb_words_freq.json')
parser.add_argument('--noun_freq_file', default='noun_words_freq.json')
parser.add_argument('--verb_concept_file', default='sent_concept_verb.json')
parser.add_argument('--noun_concept_file', default='sent_concept_noun.json')


parser.add_argument('--saved_file', default='verb_gt.json')
parser.add_argument('--saved_file2', default='noun_gt.json')
parser.add_argument('--saved_file3', default='verb_gt_all.json')
parser.add_argument('--saved_file4', default='noun_gt_all.json')
args = parser.parse_args()

verb_freq = json.load(open(args.verb_freq_file))
noun_freq = json.load(open(args.noun_freq_file))
verb_concept = json.load(open(args.verb_concept_file))
noun_concept = json.load(open(args.noun_concept_file))
trn_name = np.load(args.trn_name_file)
verb_topk = list(verb_freq)[:512]
noun_topk = list(noun_freq)[:1024]

verb_gt = {}
noun_gt = {}

for i, name in enumerate(trn_name):
    if i % 100 == 0:
        print(i)

    labels = []
    for j, sent in enumerate(verb_concept[name]):
        labels_tmp = []
        if len(sent):
            for k in sent:
                try:
                    idx = verb_topk.index(k)
                    labels_tmp.append(idx)
                except:
                    pass
        labels.append(labels_tmp)

    verb_gt[name] = labels


    labels = []
    for j, sent in enumerate(noun_concept[name]):
        labels_tmp = []
        if len(sent):
            for k in sent:
                try:
                    idx = noun_topk.index(k)
                    labels_tmp.append(idx)
                except:
                    pass
        labels.append(labels_tmp)
    noun_gt[name] = labels


json_str = json.dumps(verb_gt, indent=2)
with open(args.saved_file, 'w') as json_file:
    json_file.write(json_str)
json_str2 = json.dumps(noun_gt, indent=2)
with open(args.saved_file2, 'w') as json_file:
    json_file.write(json_str2)

'''
################################################################
'''
verb_gt_all = {}
noun_gt_all = {}

for i, name in enumerate(trn_name):
    if i % 100 == 0:
        print(i)

    labels = {}
    for j, sent in enumerate(verb_concept[name]):
        if len(sent):
            for k in sent:
                try:
                    idx = verb_topk.index(k)
                    try:
                        labels[idx] += 1
                    except:
                        labels[idx] = 1
                except:
                    pass
    max_num = 0
    for k in labels:
        if labels[k] > max_num:
            max_num = labels[k]
    for k in labels:
        labels[k] = round(labels[k] / max_num, 2)
    labels = dict(sorted(labels.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    verb_gt_all[name] = labels


    labels = {}
    for j, sent in enumerate(noun_concept[name]):
        if len(sent):
            for k in sent:
                try:
                    idx = noun_topk.index(k)
                    try:
                        labels[idx] += 1
                    except:
                        labels[idx] = 1
                except:
                    pass
    max_num = 0
    for k in labels:
        if labels[k]> max_num:
            max_num = labels[k]

    for k in labels:
        labels[k] = round(labels[k] / max_num, 2)
    labels = dict(sorted(labels.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    noun_gt_all[name] = labels


json_str = json.dumps(verb_gt_all, indent=2)
with open(args.saved_file3, 'w') as json_file:
    json_file.write(json_str)
json_str2 = json.dumps(noun_gt_all, indent=2)
with open(args.saved_file4, 'w') as json_file:
    json_file.write(json_str2)