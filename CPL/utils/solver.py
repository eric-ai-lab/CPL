import torch
import torch.nn as nn
import numpy as np
import os
import zipfile
from .configs import * 
from clip import clip
from clip.utils import *


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output




def mupdate(ust, ut):
    m = 0.999 #0.9
    u = ust * m + ut * (1. - m)
    return torch.clamp(u, min=0, max=1)



def get_next_edit(feat_original, feat_target, label, model, lab2cname, ntext_features, logit_scale, index, S=[]):
    highest_conf = 0.0
    edit = -1
    sm = nn.Softmax()
    with torch.no_grad():
        for i in range(feat_original.shape[1]):
            if len(list(filter(lambda x: x == i, S))): continue

            tmp = feat_original[index, i].clone()
            feat_original[index, i] = feat_target[index, i]

            image_features = feat_original[index, :]/ feat_original[index, :].norm(dim=-1, keepdim=True)
            
            l_i = float(logit_scale * image_features @ ntext_features.t())
            if l_i > highest_conf:
                highest_conf = l_i
                edit = i
            feat_original[index, i] = tmp
    return edit, highest_conf


def solver(model, image, nimg, label):
    args, config = set_args()
    dataset = args.root.split('/')[-1]
    if os.path.exists(f'weights/{dataset}_{args.seed}_{args.opts}.pth'): 
        print('load weights')
        return torch.load(f'weights/{dataset}_{args.seed}_{args.opts}.pth')
    else:
        try:
            import gdown
            url = 'https://drive.google.com/file/d/1fTy8Wxg15Xh9_8GMgE7eWHbagUWnNtDp/view?usp=sharing'
            gdown.download(url, f'preprocessed.zip', quiet=False)
            with zipfile.ZipFile("preprocessed.zip","r") as zip_ref:
                print('***** loading and unzipping preporcessed weights *****' )
                zip_ref.extractall("./weights/")
                print('\tCompleted!')
                os.remove('preprocessed.zip')
            return torch.load(f'weights/{dataset}_{args.seed}_{args.opts}.pth')
        except:
            if config.COCOOPCF == "true":
                u, ep = trained_solver(model, image, nimg, label, args)
                torch.save(u.t().repeat(4, 1), f'weights/{dataset}_{args.seed}_{args.opts}_{ep}.pth')
                return u
            else:
                dtype = model.dtype
                logit_scale = model.logit_scale
                with torch.no_grad():
                    source_features = model.visual(image.type(dtype)).cuda()
                    distractor_features = model.visual(nimg.type(dtype).cuda()).cuda()

                    lab2cname =  np.load('lab2cname.npy', allow_pickle=True).item()       # replace and load the corresponding preprocessed label to classnames dictionary in the root path
                    ScoreDict = np.load('score.npy', allow_pickle=True).item()

                    u = -torch.ones(len(lab2cname), model.visual.output_dim)
                    sm = nn.Softmax()
                    max_loops = source_features.shape[1]

                    ss = {}
                    for i in range(source_features.shape[0]):
                        S = []


                        nclassname = ScoreDict[lab2cname[int(label[i])]]


                        ntext = clip.tokenize(nclassname)
                        model = model.cuda()
                        ntext_features = model.encode_text(ntext.long().cuda())


                        classname = lab2cname[int(label[i])]


                        text = clip.tokenize(classname)
                        text_features = model.encode_text(text.long().cuda())

                        for j in range(max_loops):
                            

                            edit, conf = get_next_edit(source_features, distractor_features, label, model, lab2cname, ntext_features, logit_scale, i, S)
                            S.append(edit)

                            source_features[i, edit] = distractor_features[i, edit]


                            image_features = source_features[i, :]/ source_features[i, :].norm(dim=-1, keepdim=True)  
                            l_i = float((logit_scale * image_features @ text_features.t()))   
                            l_i2 = float((logit_scale * image_features @ ntext_features.t()))

                            if l_i2 > l_i: break
                        if int(label[i]) not in ss:
                            u[int(label[i]), S] = 1
                        ss[int(label[i])] = 1
                torch.save(u.t().repeat(4, 1), f'weights/{dataset}_{args.seed}_{args.opts}_{label[0]}.pth')
                return u.t().repeat(4, 1)
            


