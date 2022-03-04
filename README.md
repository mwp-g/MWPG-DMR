# Disentangled Memory Retrieval Towards Math Word Problem Generation

Code for our SIGIR'22 short paper.

## Environment

The code is written and tested with the following packages:

- transformers==2.11.0
- faiss-gpu==1.6.1
- torch==1.8.1+cu111 

you can install them by:

```
pip install -r requirements.txt
```

## Datasets 

We provided our processed Math23K dataset in the `data` folder.

## Instructions

The scripts to reproduce our results can be found in the `scripts` folder. Here we give an example to reproduce our experiments. 

The hyperparameter settings are all in the corresponding `.sh` folder.

NOTE: You should check detailed information in the corresponding shell scripts.

0. do `export MTPATH=where_you_hold_your_data_and_models`
1. training data preprocessing: `sh scripts/prepare.sh` 
2. pre-training for equation-based retrieval module: `sh scripts/pretrain_eq.sh`
3. pre-training for topic-words-based retrieval module: `sh scripts/pretrain_tw.sh`
4. build the initial index: `sh scripts/build_index.sh`
5. training: `sh scripts/train.sh`
6. testing:   `sh scripts/work.sh `

## Running results

If you run our code step by step with two 3090 GUPs, you'll get the same or similar results.

| BLEU-4 | METEOR | ROUGE-L | ACC-eq |
| :----: | :----: | :-----: | :----: |
| 0.392  | 0.375  |  0.630  | 0.548  |

