# SpaDE (CIKM'22)

Welcome🙌! This is a repository for our paper ["SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval"](https://arxiv.org/abs/2209.05917) in CIKM'22.<br>

Build your environment with the following CLI before reproduction.<br>
We have confirmed that the results are reproduced successfully in Python version 3.7.15 and PyTorch version 1.12.1.<br>

## Preparing

```
git clone https://github.com/eunseongc/SpaDE
cd SpaDE
pip install -r requirements.txt
```

Please visit https://microsoft.github.io/msmarco/Datasets and https://github.com/DI4IR/SIGIR2021 (for `expanded_collection.tsv`) to download data.

You can download training triples (qid, pos pid, neg pid) from [here](https://drive.google.com/file/d/1cJ72zPQik2mOHJNumCeR6aDMgiNwyfEC/view?usp=sharing).<br>
(Note that this training triples have same negatives with the one given by MS MARCO, but we rearranged it and splitted the valid dataset.)



Before run the script, please locate 1) `collection.tsv` (or `expanded_collection.tsv`) and 2) `marco_triples.pkl` to `data/marco-passage/`.

## Training
Run this script to train the SpaDE from the scratch.<br>
(It took us about 40 hours with 1x3090Ti GPU when the top 2 tokens were expanded)

```
source scripts/run_train.sh 2
```

## Indexing

To be updated



## Evaluation

To be updated



## Citation
Please cite our paper:
```
@inproceedings{ChoiLCKSL22,
  author    = {Eunseong Choi and
               Sunkyung Lee and
               Minjin Choi and
               Hyeseon Ko and
               Young{-}In Song and
               Jongwuk Lee},
  title     = {SpaDE: Improving Sparse Representations using a Dual Document Encoder
               for First-stage Retrieval},
  booktitle = {Proceedings of the 31st {ACM} International Conference on Information
               {\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
  pages     = {272--282},
  publisher = {{ACM}},
  year      = {2022},
}
```
