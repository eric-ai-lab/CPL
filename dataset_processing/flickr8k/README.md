# split.py
Run `pyrhon split.py $n` to generate split files for n-shot training.



# Flickr8k

- train.txt:
  - n image-text pairs for few-shot training (fewshot)
    - n random image sampled from training set in Karpathy split
    - Caption for each image is randomly selected from five golden caption
  - 1000\*5 pairs for testing (unseen)
    - Testing set from Karpathy split



- test.txt: 
  - Same as (fewshot) in train.txt
  - 1000 image-text pairs for testing

- val.txt: empty file

- classnames.txt: all captions (n + 1000\*5 in total)


- dataset_flickr8k.json: Original Karpathy split

