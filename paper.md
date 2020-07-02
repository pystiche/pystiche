---
title: 'pystiche: A Framework for Neural Style Transfer'
tags:
  - Python
  - Neural Style Transfer
  - framework
  - PyTorch
authors:
  - name: Philip Meier
    orcid: 0000-0002-5184-1622
    affiliation: 1
  - name: Volker Lohweg
    orcid: 0000-0002-3325-7887
    affiliation: 1
affiliations:
 - name: inIT –- Institute Industrial IT, Technische Hochschule Ostwestfalen-Lippe (TH-OWL)
   index: 1
   
date: 2 July 2020
bibliography: docs/source/references.bib
---

# Summary and Purpose

The seminal work of Gatys, Ecker, and Bethge gave birth to the field of 
_Neural Style Transfer_ (NST) in 2016 [@GEB2016]. The general idea behind an NST can be 
conveyed with only three images and two symbols:

![](docs/source/graphics/banner/banner.jpg)

In words: an NST describes the merger between the content and artistic style of two 
arbitrary images. This idea is nothing new in the field of computational style transfer 
[@GG2001]. What distinguishes NST from traditional approaches is its generality: an NST 
only needs a single arbitrary content and style image as input and thus "makes -- for 
the first time -- a generalized style transfer practicable" [@SID2017].

Due to its vivid nature, the field of NST gained a lot of traction after its emergence 
[@JYF+2019]. While many new techniques or augmentations have been developed, the field 
lacks standardization, which is especially evident in the reference implementations of 
the authors. `pystiche` aims to fill this gap.

`pystiche` is a framework for NST written in Python. It is built upon the Deep Learning 
(DL) framework PyTorch [@PGM+2019], since at the core each NST algorithm utilizes a 
Deep Neural Network. `pystiche` pursues similar goals as DL frameworks:

1. **Accessibility**
   Starting off with NST can be quite overwhelming due to the sheer amount of 
   techniques one has to know and be able to deploy. `pystiche` aims to provide an 
   easy-to-use interface that reduces the necessary prior knowledge about NST and DL 
   to a minimum.
2. **Reproducibility**
   Implementing NST from scratch is not only inconvenient but also error-prone. 
   `pystiche` aims to provide reusable tools that let developers focus on their ideas 
   rather than worrying about bugs in everything around it.

`pystiche` provides built-in implementations of the most used NST techniques 
[@MV2014; @GEB2016; @LW2016]. Furthermore, due its modular implementation, it provides
the ability to mix current state-of-the-art techniques with new ideas with ease. 
Finally, `pystiche`s core audience are researchers, but its easy-to-use user interface 
opens up the field of NST for recreational use by laypersons.

# Acknowledgements

This contribution is part of the project _Fused Security Features_, which is funded by 
the _Ministry for Culture and Science of North Rhine-Westphalia_ (MKW NRW) under the 
Grant ID `005-1703-0013`. The authors thank Julian Bültemeier for extensive internal 
testing.

# References
