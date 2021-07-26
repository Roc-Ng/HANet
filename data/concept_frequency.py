import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
import argparse

# parser
parser = argparse.ArgumentParser(description='concept')
parser.add_argument('--ref_caption_file', default='VATEX/annotation/RET/ref_captions.json')
parser.add_argument('--trn_name_file', default='VATEX/public_split/trn_names.npy')
parser.add_argument('--saved_file', default='sent_concept_verb.json')
parser.add_argument('--saved_file2', default='sent_concept_noun.json')
parser.add_argument('--saved_file3', default='verb_words_freq.json')
parser.add_argument('--saved_file4', default='noun_words_freq.json')
args = parser.parse_args()

concept_verb = {}
concept_noun = {}
verb_words_freq = {}
noun_words_freq = {}
# stopwords
stop_words = set(stopwords.words('english'))
# punctuations
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
# lemmatizer
lemmatizer = WordNetLemmatizer()
# reverve_tags
reserve_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ref_captions = json.load(open(args.ref_caption_file))
trn_name = np.load(args.trn_name_file)

for i, name in enumerate(trn_name):
    if i % 100 == 0:
        print(i)
    verb_list = []
    noun_list = []
    for j, sent in enumerate(ref_captions[name]):
        verb_tmp = []
        noun_tmp = []
        words = word_tokenize(sent)
        words = [word for word in words if word not in english_punctuations]
        words_tags = nltk.pos_tag(words)

        words = [word for word in words if word not in stop_words]
        words_tags = [word for word in words_tags if word[0] not in stop_words]
        words = [words[k] for k in range(len(words)) if words_tags[k][1] in reserve_tags]
        words_tags = [word for word in words_tags if word[1] in reserve_tags]

        for k in range(len(words)):
            if 'V' in words_tags[k][1]:
                if words_tags[k][1] == 'VB':
                    word = words[k]
                else:
                    word = lemmatizer.lemmatize(words[k], pos='v')
                verb_tmp.append(word)
                try:
                    verb_words_freq[word] += 1
                except KeyError:
                    verb_words_freq[word] = 1
            elif 'NN' in words_tags[k][1]:
                if words_tags[k][1] == 'NN':
                    word = words[k]
                else:
                    word = lemmatizer.lemmatize(words[k])
                try:
                    noun_words_freq[word] += 1
                except KeyError:
                    noun_words_freq[word] = 1
                noun_tmp.append(word)
            else:
                raise Exception('Error!')
        verb_list.append(verb_tmp)
        noun_list.append(noun_tmp)
    concept_verb[name] = verb_list
    concept_noun[name] = noun_list

# sorted
verb_words_freq = dict(sorted(verb_words_freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
noun_words_freq = dict(sorted(noun_words_freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))

# save file
json_str = json.dumps(concept_verb, indent=2)
json_str2 = json.dumps(concept_noun, indent=2)
with open(args.saved_file, 'w') as json_file:
    json_file.write(json_str)
with open(args.saved_file2, 'w') as json_file:
    json_file.write(json_str2)


# save file
json_str = json.dumps(verb_words_freq, indent=2)
with open(args.saved_file3, 'w') as json_file:
    json_file.write(json_str)
json_str2 = json.dumps(noun_words_freq, indent=2)
with open(args.saved_file4, 'w') as json_file:
    json_file.write(json_str2)



