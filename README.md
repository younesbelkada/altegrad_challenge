# altegrad_challenge

Altegrad 2021-2022 - Citation Prediction Challenge - APAVOU Clément - BELKADA Younes - ZUCKER Arthur

## Getting started

```pip3 install requirements.txt```

## Get the data

```sh download_data.sh```

## Reproduce the best submission

```python3 main.py --train False```

This model obtained the best results. However, it did not have the final model architecture. Which is why it has to be repoduced on the best-model branch: the order of the features is different in the new version. Here it is handcrafted, abstract, neighbors, keywords. This will first load the embeddings that have been used as well as the best model and run the inference

## Reproduce the model training

At the file ```config/hparams.py```, put the variables ```only_create_abstract_embeddings``` and ```only_create_keywords``` to ```True``` (line 96 and 97). After that run,
```python3 main.py```
The script will re-create the embeddings and train the model.