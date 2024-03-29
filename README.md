# Learn To Bound: A Generative Cram\'er-Rao Bound
This repository contains a python package that computes the Generative Cram'er-Rao Bound given a Conditional Normalizing Flow. In addition, this repository contains a set of four examples of various types of signal processing examples.

## Abstract 
The Cram\'er-Rao bound (CRB), a well-known lower bound on the performance of any unbiased parameter estimator, has been used to study a wide variety of problems. However, to obtain the CRB,  requires an analytical expression for the likelihood of the measurements given the parameters, or equivalently a precise and explicit statistical model for the data. In many applications, such a model is not available.  Instead, this work introduces a novel approach to approximate the CRB using data-driven methods, which removes the requirement for an analytical statistical model. This approach is based on the recent success of deep generative models in modeling complex, high-dimensional distributions. Using a learned normalizing flow model, we model the distribution of the measurements and obtain an approximation of the CRB, which we call Generative Cram\'er-Rao Bound (GCRB). Numerical experiments on simple problems validate this approach,  and experiments on two image processing tasks of image denoising and edge detection with a learned camera noise model demonstrate its power and benefits.

# Usage

## Install

To run this reposotiy, you will neeed the NormFlowPy package (https://github.com/haihabi/NormFlowPy) and the installion of the requirments file. 
Note that to run GCRB on NoiseFlow[2] model you will also need to get PyTorch NoiseFlow-SiLU[1] from (https://github.com/haihabi/noise_flow) as for the training it is performaned in the NoiseFlow repo.

## How To Run & Train
'''
python experiments/main.py --model_type MODEL_TYPE 
'''
There three model types here and there configuration files: 
* Linear [1] with Gaussion noise (experiments/config/linear.config.json).
* Scale [1] with non-Gaussion noise (experiments/config/scale.config.json). 
* Sinusoidal with Gaussion noise, quantization and winner phase noise (experiments/config/sine.config.json). 

# Contribution & Problems

We welcomes contributions from anyone and if you find a bug or have a question, please create a GitHub issue.


# References

[1] Habi, Hai Victor, Hagit Messer, and Yoram Bresler. "Learning to Bound: A Generative Cramér-Rao Bound." IEEE Transactions on Signal Processing (2023).

[2] Abdelhamed, Abdelrahman, Marcus A. Brubaker, and Michael S. Brown. "Noise flow: Noise modeling with conditional normalizing flows." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
