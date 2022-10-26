# Data praperation
- dataset_coco.json: Original Karpathy split. This needs to be downloaded from [Here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits?select=dataset_coco.json)

- Run `python split.py $n $a` to generate the following 4 split files for *n-shot* training using random seed *a*:

# MSCOCO
- train.txt contains:
  - *n* image-text pairs for few-shot training
    - *n* random images sampled from training set
    - The caption for each image is randomly selected from five golden captions, which will be used as the (fewshot) training example in [subsample_classes](../../datasets/oxford_pets.py)
  - 5000\*5 image-text pairs for unseen 
    - 5000 unseen images and their corresponding five captions
    - This provides all (unseen) captions information, which will be used later as the label dictionary in [subsample_classes](../../datasets/oxford_pets.py)
    - One sentences with more than 77 tokens have been shortened by hand

- test.txt contains:
  - *n* training pairs (testing on seen data is meaningless in our Image-Text Retrieval setting, yet we keep the training pairs here to maintain consistency with the data format of image classification)
  - Test set follows the original split

- val.txt: empty file

- classnames.txt: all captions (*n* + 5000\*5 in total)