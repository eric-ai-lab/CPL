import pandas as pd
import csv
import random
import sys
from tqdm import tqdm

#RANDOM_SEED = 1
FEW_SHOT = int(sys.argv[1]) # n-shot: 414 828 2483 4139
RANDOM_SEED = int(sys.argv[2]) # set random set

random.seed(RANDOM_SEED)

df = pd.read_json("dataset_coco.json")
df_train = df[df.images.str['split'] == 'val']
df_train = df_train.sample(n=FEW_SHOT, random_state=RANDOM_SEED) # down sample to n-shot
df_test = df[df.images.str['split'] == 'test']

# n-shot train data
train_imnames = df_train.images.str['filename'].tolist()
train_captions = []
for i in range(len(df_train)):
    train_captions.append(df_train.images.str['sentences'].tolist()[i][random.randint(0, 4)]['raw'].rstrip().lstrip())
    
# all test data
test_imnames = []
test_captions = []

print("Creating data split with ", FEW_SHOT, " training shot...")

for i in tqdm(range(len(df_test))):
    for j in range(5):
        test_imnames.append(df_test.images.str['filename'].tolist()[i])
        test_captions.append(df_test.images.str['sentences'].tolist()[i][j]['raw'].rstrip().lstrip())

# n-shot train + 5000 test
textfile = open("train.txt", "w")
for i in range(len(train_imnames)):
    textfile.write(train_imnames[i] + "*" + train_captions[i] + "\n")
for i in range(len(test_imnames)):
    textfile.write(test_imnames[i] + "*" + test_captions[i] + "\n")
textfile.close()

# n_shot train + 1000 test 
textfile = open("test.txt", "w")
for i in range(len(train_imnames)):
    textfile.write(train_imnames[i] + "*" + train_captions[i] + "\n")
for i in range(5000):
    textfile.write(test_imnames[i*5] + "*" + test_captions[i*5] + "\n")
textfile.close()

textfile = open("val.txt", "w")
textfile.close()

# file with all captions
textfile = open("classnames.txt", "w")
for i in range(len(train_captions)):
    textfile.write(train_captions[i] + "\n")
for i in range(len(test_captions)):
    textfile.write(test_captions[i] + "\n")
textfile.close()
