import pandas as pd
import csv
import random
import sys
from tqdm import tqdm
import json
import logging
import pathlib
import itertools
from collections import defaultdict

#RANDOM_SEED = 1
FEW_SHOT = float(sys.argv[1]) # n-shot: 0.5% 1% 3%
RANDOM_SEED = int(sys.argv[2]) # set random set
# PATH = pathlib.Path(sys.argv[3])
PATH = pathlib.Path('./')

random.seed(RANDOM_SEED)

def json_prec_dump(data, prec=6):
    return json.dumps(json.loads(json.dumps(data), parse_float=lambda x: round(float(x), prec)))

all_types = []
for json_file in PATH.iterdir():
    if json_file.suffix != '.json':
        print(f'Ignoring file {json_file.name} by suffix.')
        continue
    if json_file.name == 'T5_filtered_prompt.json' or json_file.name == 'answer_vocab.json' or json_file.name == 'all.json':
        print(f'Ignoring file {json_file.name}.')
        continue
    # json_data = json.loads(json_file.read_text())
    json_data = json.loads(json_file.read_text())

    all_types.extend(json_data)

b = json.dumps(all_types)
f = open('all.json', 'w')
f.write(b)
f.close

df = pd.read_json("all.json")
# df = pd.read_json("/data1/xh/vqa/karpathy_test.json")
# df = pd.read_json("/data1/xh/vqa/val.json")
number = int(len(df)*FEW_SHOT)
print(len(df), number, FEW_SHOT, len(df)*FEW_SHOT)
df_train = df[:-1000].sample(n=number, random_state=RANDOM_SEED) # down sample to n-shot
print(len(df_train), df_train.keys(), df_train[0:2])
df_test = df[-1000:]
print('len is', len(df_test), df_test[0:2])




train_imnames = df_train['img'].tolist()
train_prompts = []
for i in range(len(df_train)):
    train_prompts.append(df_train['question'].tolist()[i].rstrip().lstrip() +' ' + df_train['prompt'].tolist()[i].rstrip().lstrip())

test_imnames = []
test_prompts = []

print("Creating data split with ", FEW_SHOT, " training shot...")

for i in tqdm(range(len(df_test))):
    test_imnames.append(df_test['img'].tolist()[i])
    test_prompts.append(df_test['question'].tolist()[i].rstrip().lstrip() +' ' + df_test['prompt'].tolist()[i].rstrip().lstrip())

textfile = open("train.txt", "w")
for i in range(len(train_imnames)):
    textfile.write(train_imnames[i] + "*" + train_prompts[i] + "\n")
for i in range(len(test_imnames)):
    textfile.write(test_imnames[i] + "*" + test_prompts[i] + "\n")
textfile.close()

textfile = open("test.txt", "w")
for i in range(len(train_imnames)):
    textfile.write(train_imnames[i] + "*" + train_prompts[i] + "\n")
for i in range(len(test_imnames)):
    textfile.write(test_imnames[i] + "*" + test_prompts[i] + "\n")
textfile.close()

textfile = open("val.txt", "w")
textfile.close()

textfile = open("classnames.txt", "w")
for i in range(len(train_prompts)):
    textfile.write(train_prompts[i] + "\n")
for i in range(len(test_prompts)):
    textfile.write(test_prompts[i] + "\n")
textfile.close()
