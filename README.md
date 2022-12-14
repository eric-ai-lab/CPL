# CPL: Counterfactual Prompt Learning for Vision and Language Models

This repo contains the codebase for our EMNLP 2022 paper <a href="https://arxiv.org/abs/2210.10362"> CPL: Counterfactual Prompt Learning for Vision and Language Models

<p align="center">
  <img width=50% src="assets/motivation.png">
</p>

Prompt tuning is a new few-shot transfer learning technique that only tunes the learnable prompt for pre-trained vision and language models such as CLIP. However, existing prompt tuning methods tend to learn spurious or entangled representations, which leads to poor generalization to unseen concepts.
Towards non-spurious and efficient prompt learning from limited examples, this paper presents a novel Counterfactual Prompt Learning (CPL) method for vision and language models, which simultaneously employs counterfactual generation and contrastive learning in a joint optimization framework.
Particularly, CPL constructs counterfactual by identifying minimal non-spurious feature change between semantically-similar positive and negative samples that causes concept change and learns more generalizable prompt representation from both factual and counterfactual examples via contrastive learning.
Extensive experiments demonstrate that CPL can obtain superior few-shot performance on different vision and language tasks than previous prompt tuning methods on CLIP. On image classification, we achieve a 3.55% average relative improvement on unseen classes across seven datasets; on image-text retrieval and visual question answering, we gain up to 4.09% and 25.08% relative improvements across three few-shot scenarios on unseen test sets.

## How to Install
This code is built heavily on top of the toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), so you need to install the `dassl` environment first. 
To install the toolbox, you need to follow the instructions described in the original codebase [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. 
The engine is updated to be adapted to this work. So you also need to clone Dassl.pytorch from this repo.


## Dataset preparation
- Follow [DATASETS.md](CPL/DATASETS.md) and download the datasets used.
- Follow instructions in the [dataset_processing](CPL/dataset_processing) repo to preprocess the datasets.
- The off-the-shelf prompts of VQAv2 that can be used for customized purpose are available at [prompt](CPL/prompt)


## Dependencies

You'll need a working Python environment to run the code. The code is based on `python 3.6` and `pytorch 1.6`.
The recommended way to set up your environment is through the [Anaconda Python distribution](https://www.anaconda.com/download/) which provides the `conda` package manager.

The required dependencies are specified in the file `requirements.txt`.

Run the following command in the repository folder (where `requirements.txt` is located) to create a separate environment and install all required dependencies in it:

```shell
conda create -n env_name python=3.6   # create new environment
source activate env_name
pip install -r requirements.txt
```


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate env_name

or, if you're on Windows:

    activate env_name

This will enable the environment for your current terminal session. 


#### Running arguments: 
You can lauch corresponding experiments by passing the given arguments
```sh
cd CPL
python train.py 
```
for baseline method; Or 
```sh
cd CPL
python train_cf.py 
```
for CPL method.

Avaliable arguments are:

| Parameter          | Description  |
| :----------------: | :------------|
| --root ${DATA} \   | Your dataset location|
| --seed ${SEED} \     | Random seeds         |
| --trainer ${TRAINER} \ |CoCoOpcf/CoCoOp/zsclip    |
| --dataset-config-file configs/datasets/${DATASET}.yaml \ |Dataset onfig location|
| --config-file configs/trainers/${TRAINER}/${CFG}.yaml \ |Trainers config location|
| --output-dir ${DIR} \ | output directory location|
| DATASET.NUM_SHOTS ${SHOTS} \  | # shots for classification/ 0.5 1 3 5 for ITR and VQA|
| DATASET.SUBSAMPLE_CLASSES fewshot | 'fewshot' for ITR and VQA/ number e.g. 4 8 16 for classifcation|

Examples are for baseline:
```sh
cd CPL
# TRAINER=ZeroshotCLIP
TRAINER=CoCoOp
python train.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/mscoco.yaml \
    --config-file configs/trainers/${TRAINER}/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --output-dir output \
    DATASET.NUM_SHOTS 1 \
    DATASET.SUBSAMPLE_CLASSES fewshot
```
for cpl:
```sh
cd CPL
python train_cf.py \
    --root /path/to/datasets \
    --seed 1 \
    --trainer CoCoOpcf \
    --dataset-config-file configs/datasets/mscoco.yaml \
    --config-file configs/trainers/CoCoOpcf/vit_b16_c4_ep10_batch1_ctxv1.yaml \
    --output-dir output \
    DATASET.NUM_SHOTS 1 \
    DATASET.SUBSAMPLE_CLASSES fewshot
```

Or you can use bash

```sh
cd scripts
sh cocoop/main_cpl.sh $dataset 
```
for methods.

The averaged results of ours vs. cocoop on image classification: 88.10 vs. 86.31 (seen), 85.54 vs. 82.68 (unseen). On ITR: 73.85 vs. 71.99 (3%). On VQA: 36.32 vs. 30.18 (3%).

## Code structure
```
Main/
????????? Dassl.pytorch
???   ????????? Dassl
???   ???   ????????? engine (modified trainer and dataset loader for CPL)
???   ???   ???   ????????? ...
???   ???   ????????? evaluation (modified evaluator for ITR and VQA)
???   ???   ???   ????????? ...
???   ???   ????????? metrics (mesaured metrics)
???   ???   ???   ????????? ...
???   ???   ????????? ...
???   ????????? ...
????????? datasets (datatset processing files, added flickr8k, 30k, mscoco, and vqav2)
???   ???   ????????? flickr8k.py 
???   ???   ????????? flickr30k.py
???   ???   ????????? mscoco.py
???   ???   ????????? vqav2.py
????????? trainers
???   ????????? cocoop.py (original CoCoOp)
???   ????????? cocoopcf.py (CPL)
???   ????????? zsclip.py (zeroshot CLIP)
????????? prompt (T5 generated prompts for VQAv2)
????????? train_cf.py (main training file for CPL)
????????? train.py (main training file for CoCoOp)
?????????  ...
```


## Acknowledgements
We thank the authors of [CoOp](https://github.com/KaiyangZhou/CoOp) and [CoCoOp](https://github.com/KaiyangZhou/CoOp) for releasing their code and engine. We thank the authors of [CLIP Models are Few-shot Learners](https://arxiv.org/pdf/2203.07190.pdf) for their method of VQA prompt generation. This project is supported by the Google Ads Faculty Research Award. 


## Citation

Please consider citing the paper if you find it useful. Thks! :)
```
@inproceedings{he-2022-CPL,
    title = "{CPL}: Counterfactual Prompt Learning for Vision and Language Models",
    author = " Xuehai He and 
        Diji Yang and 
        Weixi Feng and
        Tsu-Jui Fu and
        Arjun Akula and 
        Varun Jampani and
        Pradyumna Narayana and 
        Sugato Basu and
        William Yang Wang and 
        Xin Eric Wang",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
}
```
