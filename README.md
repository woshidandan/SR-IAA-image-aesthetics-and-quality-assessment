[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

<div align="center">
<h1>
<b>
“Special Relativity” of Image Aesthetics Assessment: a Preliminary Empirical Perspective
</b>
</h1>
<h4>
<b>
Rui Xie, Anlong Ming, Shuai He, Yi Xiao, and Huadong Ma
    
Beijing University of Posts and Telecommunications
</b>
</h4>
</div>

## Introduction

Image aesthetics assessment (IAA) primarily examines image quality from a user-centric perspective and can be applied to guide various applications, including image capture, recommendation, and enhancement. The fundamental issue in IAA revolves around the quantification of image aesthetics. Existing methodologies rely on assigning a scalar (or a distribution) to represent aesthetic value based on conventional practices, which confines this scalar within a specific range and artificially labels it. However, conventional methods rarely incorporate research on interpretability, particularly lacking systematic responses to the following three fundamental questions:

1. Can aesthetic qualities be quantified?

2. What is the nature of quantifying aesthetics?

3. How can aesthetics be accurately quantified?

In this paper, we present a law called "Special Relativity" of IAA (SR-IAA) that addresses the aforementioned core questions. We have developed a Multi-Attribute IAA Framework (MAINet), which serves as a preliminary validation for SR-IAA within the existing datasets and achieves state-of-the-art (SOTA) performance. Specifically, our metrics on multi-attribute assessment outperform the second-best performance by 8.06% (AADB), 1.67% (PARA), and 2.44% (SPAQ) in terms of SRCC. We anticipate that our research will offer innovative theoretical guidance to the IAA research community. Codes are available in the supplementary material.

##  Contributions

- To the best of our knowledge, this study represents the first comprehensive and systematic analysis addressing three fundamental issues in IAA. 

- A law called “Special Relativity” of IAA (SR-IAA), proposed by Anlong Ming,  investigates three fundamental issues around quantifying aesthetics. The term “special” implies that the law is based on a framework constructed by aesthetic qualities, viewers and photographers, but does not account for temporal (across eras or ages) or spacial (environmental) influences, nor dynamic alterations in human neuronal structures and synaptic connections.
The term “special” indicates that the law is specifically applicable to a rational viewer, rather than a collective one. 

- Guided by SR-IAA, we have developed a Multi-Attribute IAA Framework (MAINet), which serves as a preliminary validation for SR-IAA within the existing datasets and achieves SOTA performance.

## Law Called SR-IAA

Regarding a rational viewer in an incentive-free environment:

**_I. Image aesthetics can be quantified by consistently and definitively determining the relative preference between two comparable images within a given duration. However, aesthetic inequalities do not adhere to the transitivity observed in mathematical inequalities._**

Note: The duration varies among individuals depending on the extent to which their neurons are adequately stimulated to significantly influence their aesthetic preference; if the duration is brief without additional stimulus input, it indicates similar aesthetics in both images. Furthermore, the aesthetic comparability of any two images cannot be guaranteed, as evidenced by the contrasting themes of probability statistics and natural scenery.

**_II. Quantifying image aesthetics primarily aims to simulate two abilities: perceiving aesthetics without any reference sample and perceiving aesthetics with a reference sample._**

Note: The first ability involves "locating" the input sample within an "experience sample" space, while the second one involves calculating aesthetic differences between two comparable input samples.

**_III. Quantifying aesthetic perception without any reference can be represented by establishing pairwise relative relationships among N samples, while the quantification with a reference involves identifying variations in multiple aesthetic attributes between two samples._**

Note: The value of _N_ should be large enough, with an aesthetically uniformly distributed sample set; the aesthetic attributes, which are defined and selected based on individual aesthetic values, can all be assigned to three roles [ITU-T_F740.4]: camera, photographer, and viewer.

## Architecture of MAINet

The architecture is illustrated in Fig. Given an RGB image \( x \in \mathbb{R}^{H \times W \times C} \), the Reference Selection Module (RSM) selects two reference images within a common theme, forming a triplet denoted as \( S \in \mathbb{R}^{3 \times H \times W \times C} \). Subsequently, the images in \( S \) are split into patches of size \( \frac{H}{32} \times \frac{W}{32} \) using a patchify strategy.

These patches are then processed by three modules:

- The Position Identification Module (PIM), which assigns each patch a position through a learnable matrix \( P \in \mathbb{R}^{3 \times \frac{H}{32} \times \frac{W}{32}} \)

- The Patch Embedding Module (PEM), which converts these patches into embeddings

- The Attribute Perception Module (APM), which analyzes their attributes for comparison purposes.

Ultimately, these module outputs are integrated and fed into a decoder to predict differences in aesthetic quality between images before being converted into an aesthetic score.

![image1](https://github.com/user-attachments/assets/bfb2aa42-2b7b-4ebe-a0a5-8cf4c5d171df)

![image2](https://github.com/user-attachments/assets/24e8af4a-828d-4821-ad3a-ac98b50f3902)

## Environment Installation

- einops 0.3.0
- opencv-python 4.2.0.34
- scikit-learn 0.19.0
- scipy 0.19.1
- torch 1.9.0+cu111
- torchaudio 0.9.0
- torchvision 0.10.0+cu111
- thop 0.1.1.post2209072238

## If you find our work is useful, pleaes cite our paper:

```latex
@inproceedings{10.1145/3664647.3681172,
author = {Xie, Rui and Ming, Anlong and He, Shuai and Xiao, Yi and Ma, Huadong},
title = {"Special Relativity" of Image Aesthetics Assessment: a Preliminary Empirical Perspective},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681172},
doi = {10.1145/3664647.3681172},
abstract = {Image aesthetics assessment (IAA) primarily examines image quality from a user-centric perspective and can be applied to guide various applications, including image capture, recommendation, and enhancement. The fundamental issue in IAA revolves around the quantification of image aesthetics. Existing methodologies rely on assigning a scalar (or a distribution) to represent aesthetic value based on conventional practices, which confines this scalar within a specific range and artificially labels it. However, conventional methods rarely incorporate research on interpretability, particularly lacking systematic responses to the following three fundamental questions: 1) Can aesthetic qualities be quantified? 2) What is the nature of quantifying aesthetics? 3) How can aesthetics be accurately quantified? In this paper, we present a law called "Special Relativity" of IAA (SR-IAA) that addresses the aforementioned core questions. We have developed a Multi-Attribute IAA Framework (MAINet), which serves as a preliminary validation for SR-IAA within the existing datasets and achieves state-of-the-art (SOTA) performance. Specifically, our metrics on multi-attribute assessment outperform the second-best performance by 8.06\% (AADB), 1.67\% (PARA), and 2.44\% (SPAQ) in terms of SRCC. We anticipate that our research will offer innovative theoretical guidance to the IAA research community. All resources are available here.},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {2554–2563},
numpages = {10},
keywords = {image aesthetics assessment, neural networks},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```
