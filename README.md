# SpaDE (CIKM'22)

Welcome! This is a repository for our paper ["SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval"](https://arxiv.org/abs/2209.05917) in CIKM'22.<br>

Build your environment with the following cmd before reproduction.<br>
We have confirmed that the results are reproduced successfully in Python version 3.7.15 and PyTorch version 1.12.1.<br>

```
git clone https://github.com/eunseongc/SpaDE
cd SpaDE
pip install -r requirements.txt
```

Please visit https://microsoft.github.io/msmarco/Datasets and https://github.com/DI4IR/SIGIR2021 (for expanded collection.tsv) to download data. <br>
Download training triples(qid, pos pid, neg pid) from here: 
Note that this training triples is same with the one given by MS, but we rearranged it and splitted the valid dataset. <br>
Locate 1) `collection.tsv` (or `expanded_collection.tsv`) and 2) `marco_triples.pkl` to `data/marco-passage/`. <br>


Run this script to train the SpaDE from the scratch.
(It took us about 40 hours with 1x3090Ti GPU when the top 2 tokens were expanded)

```
source scripts/run_train.sh 2
```
