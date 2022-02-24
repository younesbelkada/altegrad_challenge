# Altegrad 2021-2022 - Citation Prediction Challenge

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada) & [Zucker Arthur](https://github.com/ArthurZucker)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

The kaggle challenge is the following : https://www.kaggle.com/c/altegrad-2021/leaderboard
## :mag_right: Introduction :

In this challenge, we are given a large scientific citation graph, with each node corresponding to a certain article. The dataset consists of 138 499 vertices i.e articles, with their associated abstract and list of authors. The goal is to be able to predict whether two nodes are citing each other, given all this information. In the next sections, we will try to elaborate on the various intuitions behind our approaches, and present the obtained results as well as some possible interpretations for each observations. The provided code corresponds to the code that we have used for the best model (\textit{i.e} the right commit).

## :hammer: Getting started

```
pip3 install requirements.txt
```

Then,

```
sh download_data.sh
```

```
python3 main.py
```

## :round_pushpin: Tips
The best model can be used using the `best-model` branch, as it does not use this implementation of the code. 
This branch is the final code as it allows customization of the various embeddings and corresponds to the latest version of the code.

## :mag_right: Results:

<p align="center">
    
| Model| loss validation |loss test (private leaderboard) | Run  |
|---|---|---|---|
| Best model | 0.07775 | 0.07939 | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/altegrad-gnn-link-prediction/test-altegrad/runs/1cwlegzz?workspace=user-clementapa) |
</p>

All experiments are available on wandb: \
 [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/altegrad-gnn-link-prediction/altegrad_challenge?workspace=user-clementapa)\
[![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/altegrad-gnn-link-prediction/test-altegrad?workspace=user-clementapa)

## Some tools used: 
 
[![](https://raw.githubusercontent.com/huggingface/awesome-huggingface/main/logo.svg)](https://huggingface.co/sentence-transformers/allenai-specter)

[![](https://github.com/MaartenGr/KeyBERT/blob/master/images/logo.png)](https://maartengr.github.io/KeyBERT/)

## Some citations

```bibtex
@misc{cohan2020specter,
      title={SPECTER: Document-level Representation Learning using Citation-informed Transformers}, 
      author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
      year={2020},
      eprint={2004.07180},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
