import os
import sys
import argparse
import torch

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip

import numpy as np
import matplotlib.pyplot as plt


def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


parser = argparse.ArgumentParser()
parser.add_argument("fpath", type=str, help="Path to the learned prompt")
parser.add_argument("topk", type=int, help="Select top-k similar words")
args = parser.parse_args()

fpath = args.fpath
topk = args.topk

assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_cpu()
token_embedding = clip_model.token_embedding.weight
print(f"Size of token embedding: {token_embedding.shape}")

prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
ctx = prompt_learner["ctx"]
ctx = ctx.float()
print(f"Size of context: {ctx.shape}")

if ctx.dim() == 2:
    # Generic context
    print('ctx dimension is', ctx.size(), ctx[:,0],token_embedding.size())
    distance = torch.cdist(ctx, token_embedding)
    print(f"Size of distance matrix: {distance.shape}")
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()].split('<')[0] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        print(f"{m+1}: {words} {dist}")

        if m ==2:
            words = ['guest', 'challenges', 'number', 'danger', 'call', 'handle', 'crater', 'hood', 'drove', 'leeds'] 
        if m ==3:
            words = ['gun', 'reasons', 'challenges', 'recl', 'booklet', 'dish', 'drying', 'screen', 'str', 'nd'] 
        a1 = dist
        x = np.arange(10)

        plt.rcParams["font.family"] = "Times New Roman"

        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        # 画柱状图
        axes.bar(x, a1,  width=0.4, label='Distance', color="#D2ACA3")
        # 图例
        axes.legend(loc='best')
        # 设置坐标轴刻度、标签
        axes.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # axes.set_yticks([160, 165, 170, 175, 180, 185, 190])
        # axes.set_ylim((0.8, 1.0))
        # axes.set_xticklabels(['zhouyi', 'xuweijia', 'lurenchi', 'chenxiao', 'weiyu', 'guhaiyao'])
        fontdict = {'fontsize': 6}
        axes.set_xticklabels(words, fontdict=fontdict) 
        # 设置title
        # axes.set_title('NLP group members heights')
        # 网格线
        axes.grid(linewidth=0.5, which="major", axis='y')
        # 隐藏上、右边框
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

        # for i in range(6):
        #     axes.text(x[i], a1[i], a1[i], ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        fig.savefig(f"sun_dist_{m}.png", dpi=800)

elif ctx.dim() == 3:
    # Class-specific context
    raise NotImplementedError




