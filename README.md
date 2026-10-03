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
    <a href="https://link.springer.com/chapter/10.1007/978-3-032-37152-2_15"><img src="https://img.shields.io/badge/SpringerNatureLink-Publication-0060df" alt="arXiv"></a>
    <a href="https://arxiv.org/abs/2607.02718"><img src="https://img.shields.io/badge/arXiv-2607.02718-b31b1b?logo=arxiv" alt="arXiv"></a>
    <a href="https://humansensinglab.github.io/AVODDiag/"><img src="https://img.shields.io/badge/github.io-Project%20Page-blue?logo=github" alt="Project Page"></a>
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
> Unfortunately, Google has deprecated all versions of *Imagen* in their API. Gemini 3x (Nano Banana) models still can be used for image generation and editing.

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

Below we provide image count information about the synthetic diagnostic and the three supplementary real datasets we used in our paper. 

| Dataset Name | Type | Train Split | Test Split | Total |
| :--- | :---: | :---: | :---: | :---: |
| Imagen 3 | Synthetic | – | 5,453 | 5,453 |
| Urban (Miami) | Real | 2,284 | – | 2,284 |
| Industrial (LA) | Real | 2,000 | – | 2,000 |
| Desert (Phoenix) | Real | 2,000 | – | 2,000 |

### Synthetic Diagnostic Data

#### Imagen 3
- [**Download link**](https://datastore.shannon.humansensing.cs.cmu.edu/share/tpEHfOfU)
- **SHA256**: `d4ba7002fe5714ebb129a9d4cc528646b93b50a8a94510dbaaca3608e1b4dce2`

Folder structure:

```
Synthetic-Diagnostic_Imagen3.zip/
 ├─ annotations_coco/
 │  └─ matched_gemini-2.5-flash-lite_vikhyatk+moondream2_car_square_bboxes.json
 ├─ images/
 └─ metadata/
```


### Real Supplementary Datasets

Each real supplementary dataset `.zip` file contains a [QGIS](https://qgis.org/) project file and two subfolders—`Layers` and `Python`. To recreate each dataset, complete the steps below in the following order:
1. [Download](https://qgis.org/download/) and install the QGIS application on your device, if unavailable. We used version 3.44 to produce the project files, but the newer ones should also work fine.
1. Unzip the archive.
1. Open the provided `.qgz` project file with QGIS.
1. Open the Python console within QGIS.
1. Open and run `01_ExportRasterTiles.py` located in `Python` folder to download and save the raster image tiles in `images` subfolder, which will be automatically created.
1. Open and run `02_ExportCOCOAnnotations.py` located in `Python` folder to generate COCO format bounding box annotations for the *"small vehicle"* class as a JSON file. 
 

#### Urban Environment (Miami, FL, USA)
- [**Download link**](https://datastore.shannon.humansensing.cs.cmu.edu/share/VS46NK-K)
- **SHA256**: `a267436974a398c2f0eadbf2ddf4331da45f63637bba043d49fa504a734bc3fb`

#### Industrial Environment (Los Angeles, CA, USA)

- [**Download link**](https://datastore.shannon.humansensing.cs.cmu.edu/share/htNPrjR0)
- **SHA256**: `3256c93688cb29a07c404e06233d042177b38bf3d1c4e0c2755ce2fbdfaff8f2`

#### Desert Environment (Phoenix, AZ, USA)

- [**Download link**](https://datastore.shannon.humansensing.cs.cmu.edu/share/9tgLMPH9)
- **SHA256**: `d882a311e221176d193269d9e61d6fd7566c9df162e7b65215dd496a2b7c17db`


## BibTeX Citations

**ECCV 2026 Proceedings**
```bibtex
@InProceedings{10.1007/978-3-032-37152-2_15,
    author="Panev, Stanislav
    and Jeon, Minhyek
    and Khindkar, Vaishnavi
    and Deshpande, Ahish
    and de Melo, Celso M.
    and Hu, Shuowen
    and Chakraborty, Shayok
    and De la Torre, Fernando",
    editor="Favaro, Paolo
    and Kukelova, Zuzana
    and Maki, Atsuto
    and Rohrbach, Anna
    and Schindler, Konrad
    and Tombari, Federico",
    title="Diagnosing Aerial-View Object Detectors with Foundational Image Generative Models",
    booktitle="Computer Vision -- ECCV 2026",
    year="2026",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="260--278",
    abstract="Recent advances in large-scale image generative models enable photorealistic scene synthesis with controllable attributes. Beyond data augmentation, their potential as diagnostic tools for trained vision systems remains unexplored in the aerial and remote sensing domains. We introduce a synthetic diagnostic framework fors aerial-view vehicle detection that combines text-guided generation, attribute-controlled editing, and automated attribute verification to construct a controllable synthetic testbed. This enables fine-grained evaluation of pretrained detectors under diverse scene types and environmental conditions that are difficult to isolate in real datasets. Across three detection architectures and three real aerial datasets, synthetic scene-wise performance trends closely match real-world weaknesses. Guided by these diagnostics, targeted supplementation with small real datasets from the identified weak categories yields improvements of up to 13{\%} AP50 while requiring substantially fewer additional samples than non-targeted augmentation. Our results show that controlled synthetic probing can predict real-domain performance gaps and guide efficient data collection. The proposed diagnostic framework is modular and can incorporate alternative generative or vision-language models as capabilities evolve. Our code and datasets are available here: humansensinglab.github.io/AVODDiag/.",
    isbn="978-3-032-37152-2"
}


```

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