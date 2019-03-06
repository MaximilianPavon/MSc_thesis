# Repository for my master's thesis on the topic of Bayesian generative deep models for image data in the context of plant breeding application

## (Short) Introduction
The objective is to predict the so-called *crop-loss* of a field based on all 13 colour-channels recorded by the Sentinel 2 satellite. In order to better understand and also be able to generate satellite images, we use generative models, more specifically the Variational Auto-Encoder ([Auto-Encoding Variational Bayes by Diederik P Kingma, Max Welling](https://arxiv.org/abs/1312.6114)). 

Generative models have a number of parameters significantly smaller than the amount of data they are trained on, so the models are forced to discover and efficiently internalise the essence of the data in order to generate it. Generative models hold the potential to automatically learn the natural features of a dataset, whether categories or dimensions or something else entirely. More mathematically a generative model learns the underlying probability distribution of the data, from which random samples can be drawn. Those random samples represent new and artificial data points.

## Explanation of this repository

0. The folder `0_setup_env` contains `.yml` files for setting up the virtual python environment either with CPU or GPU option.
1. The folder `1_data_analysis_download` contains initial `.ipynb` files for previewing the given data as well as a script for downloading the satellite images from SentinelHub.
2. The folder `2_data` contains the original and processed data. Due to space limitations those folders are generally empty.
3. The folder `3_code` contains the most important python scripts. The VAE is defined in `vae.py`, the remaining files are mostly helper files and some old attempts.
4. The folder `4_runs` is thought for keeping the logs as well as written plots organised. Most of them will not be maintained in this repository as they are not final results. But I will upload some of the plots for demonstration purposes. 
