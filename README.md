# MeSHProbeNet
[MeSHProbeNet: a self-attentive probe net for MeSH indexing](https://academic.oup.com/bioinformatics/article/35/19/3794/5372674)

## Prerequisites

* python==3.6.3
* pytorch==1.2.0
* torchtext==0.2.1
* numpy==1.16.2
* scipy==1.2.1

## Input data format

Take `./toy_data/` as an example
* train.tsv: The training set, where each line is a document. Each document is represented as `content word ids separated by spaces + '\t' + journal id + '\t' + MeSH ids separated by spaces`
* validation.tsv: The validation set in the same format as `train.tsv`
* vocab_w.txt: The vocabulary file for context words, where each line is `content word id + '\t' + content word`
* vocab_j.txt: The vocabulary file for journal names, where each line is `journal id + '\t' + journal`
* vocab_m.txt: The vocabulary file for MeSH terms, where each line is `MeSH id + '\t' + MeSH term`

## Run

Run on the toy data
```
python main_train.py \
  --do_save \
  --do_eval \
  --train_path ./toy_data/train.tsv \
  --dev_path ./toy_data/validation.tsv \
  --src_vocab_pt ./toy_data/vocab_w.txt \
  --jrnl_vocab_pt ./toy_data/vocab_j.txt \
  --tgt_vocab_pt ./toy_data/vocab_m.txt \
  --expt_path ./toy_data/save \
  --learning_rate 0.0025 \
  --weight_decay 0.0
```
