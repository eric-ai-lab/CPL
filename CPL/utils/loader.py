import torch
import torch.nn as nn
import numpy as np
import os
import zipfile
from configs import * 
from clip import clip
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mscoco")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--opts', type=str, default="['DATASET.NUM_SHOTS', '1', 'DATASET.SUBSAMPLE_CLASSES', 'fewshot']", help='Options')
parser.add_argument('--weights_folder', type=str, default= "../weights")
parser.add_argument('--file_prefix', type=str, default = '')
parser.add_argument('--label', type=str, default = None)
args = parser.parse_args()



assert not (os.path.exists(f'weights/{args.dataset}_{args.seed}_{args.opts}.pth'))

rootpath = './'
save_path = f'{args.weights_folder}'
merge_path = os.path.join(rootpath, save_path)
file_filter = args.file_prefix + f'*.pth'
pth_path = os.path.join(merge_path, file_filter)

files = glob.glob(pth_path, recursive = True)

uv = torch.zeros(torch.load(files[0]).size())

for file in files:
    utemp = torch.clamp(torch.load(file), min=0.0)
    uv += utemp

torch.save(uv.clamp(min=0, max=1), f'{args.weights_folder}/{args.dataset}_{args.seed}_{args.opts}.pth')