# Image Classification

Image classification scripts take three input arguments: `DATASET_NAME`, `SEED`, and `SPLIT`.

Below are examples on training and testing the model one ImageNet using random seed 1:

```bash
# train on base(first half) and test on new(second half) - corresponding table 1
bash scripts/cocoop/base2new_train.sh imagenet 1 base
bash scripts/cocoop/base2new_test.sh imagenet 1 new

# train on new(second half) and test on base(first half) - corresponding table 5, Split One   
bash scripts/cocoop/base2new_train.sh imagenet 1 new
bash scripts/cocoop/base2new_test.sh imagenet 1 base

# train on base(first half) and test on new(second half) with shuffled labels - corresponding table 6, Split Two   
bash scripts/cocoop/base2new_train.sh imagenet 1 split_two
bash scripts/cocoop/base2new_test.sh imagenet 1 new
```