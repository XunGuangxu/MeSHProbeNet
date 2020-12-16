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
* train.tsv: the training dataset. Each line is a document.
* validation.tsv
* vocab_w.txt
* vocab_j.txt
* vocab_m.txt

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
