# SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models

This repository contains code for SafeRoute, as described in the paper.


**SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models**<br />
Seanie Lee*, Dong Bok Lee*, Dominik Wagner, Minki Kang, Haebin Seong, Tobias Bocklet, Juho Lee, Sung Ju Hwang (*: equal contribution)<br/>
Paper: https://arxiv.org/abs/2502.12464
<details>
<summary>
BibTeX
</summary>
  
```bibtex
@article{
lee2025learning,
title={SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models},
author={Lee, Seanie and Lee, Dong Bok and Wagner, Dominik and Kang, Minki and Seong, Haebin and Bocklet, Tobias and Lee, Juho and Hwang, Sung Ju},
journal={Findings of the Association for Computational Linguistics (ACL)},
year={2025}
}
```
</details>


## Installation of Dependencies
```bash
conda env create -n saferoute python=3.12
conda activate saferoute
pip install -r requirements.txt
```

## Create dataset w/o paraphrasing
Create training dataset of SafeRoute. Youn can choose either 3 or guardian for the version argument. You may need to adjust batch size.
```bash
python main.py --version 3
```

## Create dataset with paraphrasing
Create training dataset of SafeRoute. Youn can choose either 3 or guardian for the version argument. You may need to adjust batch size.
```bash
python paraphrase.py --num_rounds 7
```

```bash
python create_dataset_aug.py --round 6
```

## Download data
If you don't want to create dataset from scratch, you can download it from [here](https://drive.google.com/drive/folders/1245ifJQx1Wt8actLHVPvrWvdhqK89J49?usp=sharing).
Put `data` directory in the current project.

## Training without paraphrases and evaluation
```bash
python train_router.py \
--version 3 \
--train_features ./data/{version}/train_features.pt \
--train_labels ./data/{version}/train_labels.pt \
--val_features ./data/{version}/val_features.pt \
--val_labels ./data/{version}/val_labels.pt
```

```bash
python eval.py --version 3
```



## Training without paraphrases and evaluation
```bash
python train_router.py \
--version 3 \
--train_features ./data/{version}/round6_train_features.pt \
--train_labels ./data/{version}/round6_train_labels.pt \
--val_features ./data/{version}/round6_val_features.pt \
--val_labels ./data/{version}/round6_val_labels.pt
```

```bash
python eval.py --version 3
```




