<h1 align="center">
  Diagnosing Aerial-View Object Detectors with&nbsp;Foundational Image Generative Models
</h1>

<p align="center">
  <a href="https://spanev.github.io/">Stanislav&nbsp;Panev</a><sup>1</sup>&emsp;
  <a href="https://www.linkedin.com/in/minhyekjeon">Minhyek&nbsp;Jeon</a><sup>1</sup>&emsp;
  <a href="https://vaishnvi.github.io/">Vaishnavi&nbsp;Khindkar</a><sup>1</sup>&emsp;
  <a href="https://www.linkedin.com/in/ahishd">Ahish&nbsp;Deshpande</a><sup>1</sup>&emsp;<br>
  <a href="https://celsodemelo.net/">Celso&nbsp;de&nbsp;Melo</a><sup>2</sup>&emsp;
  <a href="https://www.linkedin.com/in/shuowen-hu-661170149">Shuowen&nbsp;Hu</a><sup>2</sup>&emsp;
  <a href="https://shayokch.com/">Shayok&nbsp;Chakraborty</a><sup>1,3</sup>&emsp;
  <a href="https://www.cs.cmu.edu/~ftorre/">Fernando&nbsp;De&nbsp;la&nbsp;Torre</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Carnegie Mellon University&emsp;
  <sup>2</sup>DEVCOM Army Research Lab&emsp;
  <sup>3</sup>Florida State University
</p>

<p align="center">
  <b>ECCV 2026</b>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2607.02718"><img src="https://img.shields.io/badge/arXiv-2607.02718-b31b1b?logo=arxiv" alt="arXiv"></a>
    <a href="https://humansensinglab.github.io/AVODDiag/"><img src="https://img.shields.io/badge/-Project%20Page-blue?logo=github" alt="Project Page"></a>
</p>


## Table of contents

- [Overview](#overview)
- [APIs and supported models](#apis-and-supported-models)
- [Setup](#setup)
- [Config files](#config-files)
- [Usage](#usage)
- [Data](#data)
- [Citations](#bibtex-citations)


## Overview

AVODDiag is a suite of tools for generating synthetic aerial top-down view image benchmarks for diagnosing vehicle object detectors using commercial or opensource foundational text-to-image generative models, large language models (LLMs), and visual language models (VLMs). 


## APIs and supported models

> [!IMPORTANT]
> Unfortunately, Google has deprecated all versions of *Imagen* in their API.

- Google GenAI API
    - Image generation
        - Imagen 3 *(Deprecated)*
        - Imagen 4 *(Deprecated)*
        - Gemini 3 Pro Image (Nano Banana Pro)
        - Gemini 3.1 Flash Image (Nano Banana 2)
    
    - Image editing
        - Gemini 2.5 Flash Image (Nano Banana)
        
    - Attribute extraction and image annotations
        - Gemini 2.5 Flash
        - Gemini 2.5 Flash Lite

- OpenAI API
    - Prompt Composition
        - GPT-5


## Setup

**Create Anaconda Environment**

This project is based on Python 3.11+.
```bash
(base) $ conda create -n avoddiag python=3.11
(base) $ conda activate avoddiag
```

**Clone Project's GitHub Repository**

```bash
(avoddiag) $ git clone https://github.com/humansensinglab/AVODDiag.git
(avoddiag) $ cd ./AVODDiag
```


**Install Requirements**
```bash
(avoddiag) $ python -m pip install \
    --build-constraint build-constraints.txt \
    -r requirements.txt
```


## Config files

The `config` folder contains an example TOML config file `example.toml`. The config files currently hold the following parameters:
- API keys for the Google and OpenAI providers.
- Folder and file paths related to the bounding box approval process.


## Usage

This package provides seven notebooks related to steps for generating synthetic diagnostic aerial-view image datasets from a pre-defined attribute taxonomy. This is a minimal working example and the workflow is as follows:

1. Use `001_ImageGeneration.ipynb` to generate the primary synthetic dataset based on pre-defined taxonomy attributes.
1. Use `002_ExtractImageAttributes.ipynb` to extract the taxonomy attributes of the generated images.
1. Use `003_AnalyzeAttributes.ipynb` to analyze and process the input and the generated attributes of the primary dataset.
1. Use `004_ImageEditing.ipynb` to edit the primary dataset in order to enrich.
1. Use `003_AnalyzeAttributes.ipynb` again to analyze and process the generated attributes of the secondary (image editing) dataset.
1. Use `005_MergeDatasets.ipynb` to merge the primary and secondary datasets.
1. Use `006_ExtractImageAnnotations.ipynb` to extract automatic annotations for all generated images.
1. Use `007_ApproveAnnotations.ipynb` to start a web-based application to approve or reject the automatic annotations.



## Data

Coming soon


## BibTeX Citations

**arXiv**
```bibtex
@misc{panev2026diagnosingaerialviewobjectdetectors,
      title={Diagnosing Aerial-View Object Detectors with Foundational Image Generative Models}, 
      author={Stanislav Panev and Minhyek Jeon and Vaishnavi Khindkar and Ahish Deshpande and Celso M de Melo and Shuowen Hu and Shayok Chakraborty and Fernando De la Torre},
      year={2026},
      eprint={2607.02718},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.02718}, 
}
```