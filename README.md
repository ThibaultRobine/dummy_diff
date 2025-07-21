# Diffusion-Based Uncertainty Quantification (UQ)

This repository contains code and experiments from a research project conducted at ISIR, focused on **uncertainty quantification (UQ)** using **diffusion models**. The goal of the project was to develop a **density-based method for OOD detection** leveraging the score-based formulation of diffusion models.

The method was benchmarked against state-of-the-art approaches and integrated into the [OpenOOD](https://github.com/zhangmarvin/openood) framework for large-scale evaluation.

## Project Status

This work is **unfinished** and was originally intended to evolve into a research publication. Although the experiments and method are functional, and some parts may be incomplete or unrefined.

A draft of the unfinished article can be found here:  
 [`article_UQ.pdf`](./article_UQ.pdf)

##  Overview of the Method

The proposed method approximates log-density using the **learned score function of a diffusion model**, integrated along a path from the query point to a high-density reference point.


## Disclaimer

This is a personal research project carried out in collaboration with ISIR. The code is shared for reference purposes only.

##  Contact

For questions, feel free to contact me at:    
 thibaultrobine68@gmail.com
