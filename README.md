
# Density Estimation using Normalizing Flows

## Abstract

The fusion of generative models and transformer-inspired architectures has propelled significant advancements in the field of Computer Vision. Together, this powerful combination has given rise to architectures that exhibit exceptional performance in a wide range of imaging tasks, adeptly capturing intricate patterns and dependencies within the data. This report focuses on the task of density estimation using flow-based generative models, specifically, the effectiveness of an iterative patch-based architecture. By leveraging the potential of Normalizing Flows, we aim to push the boundaries of density estimation and showcase the capabilities of our proposed method. Through a series of experiments, we not only highlight the promising aspects of our approach but also shed light on its inherent limitations. We provide a comprehensive analysis of the architecture, discussing its strengths and weaknesses, while also presenting potential avenues for future research. Ultimately, trying to set the stage for further advancements and offer a foundation for future explorations in this domain.

## Setup

Create an environment and install all the required packages from the requirements.txt file. 

```bash
pip install -r requirements.txt
```

Disclaimer: The above command only works correctly on Linux. The installation guide for jax, jaxlib, and flax differs for Windows OS.
## Sample from the model

Random samples can be generated as follows; Here for instance for generating 16 samples with sampling temperature 0.7 and setting the random seed to 0:

```bash
python sampling.py 16 -t 0.7 -s 0 --model_path [path]
```
## Acknowledgements

This project was a part of my research module in my Master programme at University of Potsdam. The project was supervised by Eshant English, PhD Candidate at Hasso Plattner Institute. 
I would like to thank him for his guidance and mentorship.

