# SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models

This repository contains code for SafeRoute, as described in the paper.


**SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models**<br />
Seanie Lee*, Dong Bok Lee*, Dominik Wagner, Minki Kang, Haebin Seong, Tobias Bocklet, Juho Lee, Sung Ju Hwang<br/>
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
conda env create -n redteam python=3.12
conda activate saferoute
pip install -r requirements.txt
```

## Create Dataset
Create training dataset of SafeRoute. Youn can choose either 3 or guardian for the version argument. You may need to adjust batch size.
```bash
python main.py --version 3
```

## Training and evaluation
```
bash
python train_router.py --version 3
```

```
bash
python eval.py --version 3
```


