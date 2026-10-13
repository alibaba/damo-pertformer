# PertFormer


## Introduction
This is the official codebase for PertFormer: Zero-shot perturbation prediction and target discovery from wild-type biology.

Perturbation Transformer (PertFormer) is an LLM-based foundation model for zero-shot predictions of functional perturbations and cell fate dynamics. PertFormer has 3 billion parameters and was pretrained on massive bulk and single-cell multiomic data. Typically, without the need of task-specific training, PertFormer can realize the predictions of functional regulations, perturbations of genomic elements, genes, and sequences in a zero-shot manner, including the simulation of cell state transitions and disease treatment.


## Online apps
https://pertformer.ibreed.cn

This is an online web application to utilize the PertFormer model. Here, you can upload your own data and execute most of PertFormer's zero-shot predictions through simple, coding-free web applications. This website is free for public registration and use, and we have provided the GPUs for cloud computing resources. Your data will be private and safe in your own account and cannot be seen by others.


## Installation

* Clone the repository:
  * `git clone https://github.com/alibaba/damo-pertformer.git`
  * `cd damo-pertformer`

* Create the conda environment:
  * `conda env create -f environment.yml`
  * `conda activate pertformer`

* Install the Python dependencies:
  * `pip install -r requirements.txt`

> Note: FlashAttention is optional. PertFormer can run without it. Installation requires `nvcc` and `CUDA_HOME`.

* Optional:
  * `pip install --no-build-isolation flash-attn==2.7.4.post1`


## Pretrained PertFormer parameter
* Download from Hugging Face:
  * https://huggingface.co/GenomicIntelligenceDamoAcademy/pertformer
  * The pretrained PertFormer network parameters can be downloaded from the above Hugging Face repo (around 13GB).


## Run PertFormer
We have provided multiple tutorials.

* Tutorials/TF_knockdown
  * In this tutorial, we provided a full pipeline to use PertFormer to predict TF knockdown effects.
  * The pipeline includes data processing, motif scanning for KD loci, PertFormer inferences, and evaluations.
  * Deatils can be found in the .ipynb file.
* Tutorials/TF_overexpression
  * In this tutorial, we provided a full pipeline to use PertFormer to predict TF over-expression effects.
  * The pipeline includes data processing, motif scanning for KI loci, PertFormer inferences, and evaluations.
  * Deatils can be found in the .ipynb file. 
* Tutorials/EGCS_and_perturbation
  * In this tutorial, we provided general inference codes to extract PertFormer EGCS, and to model perturbation effects.
  * This tutorial provided general information of the model input format and the inference process.
  * Deatils can be found in the .ipynb file.


## Citing PertFormer
The manuscript is currently under review. You can cite the preprint version of this paper on bioRxiv: https://doi.org/10.1101/2024.12.19.629561


## Contact

For code, implementation, and GitHub-related questions, please contact:

- **Zikun Yang** (yangzikun.yzk@alibaba-inc.com)

For scientific questions and manuscript-related correspondence, please contact the paper's corresponding authors:

- **Zikun Yang** (yangzikun.yzk@alibaba-inc.com)
- **Lan Jiang** (jiangl@cncb.ac.cn)
- **Xiaodong Liu** (liuxiaodong@westlake.edu.cn)
- **Fei Gu** (gufei.gf@alibaba-inc.com)
