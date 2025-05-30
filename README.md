# Implementation of SPTTE

This repository contains the code used in our paper: [SPTTE: A Spatiotemporal Probabilistic Framework for Travel Time Estimation](https://arxiv.org/abs/2411.18484).

## Abstract

Accurate travel time estimation is essential for navigation and itinerary planning. Although existing studies have employed probabilistic models to quantify the uncertainty of travel times, modeling the evolving joint distribution of multi-trip travel times remains a significant challenge. Unlike conventional multivariate time series, trip data exhibits inherent fragmentation that severely impedes models from capturing systematic spatiotemporal dependencies. This challenge is further compounded by spatial heterogeneity in data coverage, which significantly degrades model learning efficacy across different network regions. To address this challenge, we propose SPTTE, a deep spatiotemporal probabilistic framework that models the evolving joint distribution of multi-trip travel times by formulating it as a spatiotemporal stochastic process regression problem with fragmented observations. SPTTE incorporates an RNN-based temporal Gaussian process parameterization to regularize sparse observations and capture temporal dependencies. Additionally, it employs a prior-based heterogeneity smoothing strategy to correct unreliable learning caused by unevenly distributed trips. Evaluations on real-world datasets demonstrate that SPTTE outperforms state-of-the-art deterministic and probabilistic methods by an average of 6.77%. Ablation studies and visual analytics further validate both the learned distribution evolution and the effectiveness of individual model components.

## Model Overview

![SPTTE Framework](SPTTE/Model/Result/Arc.pdf)


## Requirements

To run the code, ensure your system meets the following requirements:

- **Operating System**: Ubuntu (tested on versions 16.04 and 18.04)
- **Programming Languages**:
  - [Julia](https://julialang.org/downloads/) >= 1.0
  - Python >= 3.6
- **Deep Learning Framework**:
  - PyTorch >= 0.4 (tested on versions 0.4 and 1.0)

To install the required Julia packages, run the following command in your terminal:

```bash
julia -e 'using Pkg; Pkg.add(["HDF5", "CSV", "DataFrames", "Distances", "StatsBase", "JSON", "Lazy", "JLD2", "ArgParse"])'
```

To install the required Python packages, you can use the provided `requirements.txt` file. This file lists all necessary dependencies.

```bash
pip install -r requirements.txt
```

## Dataset

Harbin: The dataset consists of over **1 million trips** collected by **13,000+ taxis**.

Chengdu: The dataset consists of over **1.4 million taxis** collecting more than **1.4 billion GPS records**.

### Download Dataset

Download the dataset from the following link:

Harbin: [Download Dataset](https://drive.google.com/open?id=1tdgarnn28CM01o9hbeKLUiJ1o1lskrqA)

Chengdu: [Download Dataset](https://challenge.datacastle.cn/v3/cmptDetail.html?id=175)

Extracting data from OpenStreetMap can be used to obtain the road network. You can download the road network data for Chengdu from [OpenStreetMap](https://www.openstreetmap.org/) and use it for map matching.


### Data Format

Each `.h5` file contains multiple trips recorded on a given day. Each trip consists of three fields:

- `lon` (longitude)
- `lat` (latitude)
- `tms` (timestamp)

To read `.h5` files, use the [`readtripsh5`](https://github.com/ChenXu02/ProbETA/tree/main/julia/Trip.jl#L28) function in Julia. If using your own dataset, refer to `readtripsh5` to format your trajectories correctly into `.h5` files.

## Preprocessing

### Map Matching

Before training, trips must be map-matched using the [Barefoot](https://github.com/boathit/barefoot) matching server. Follow the instructions in the Barefoot repository to set up the required servers.

Once the servers are running, execute the following command to match trips:

```bash
cd SPTTE/julia
julia -p 6 mapmatch.jl --inputpath ../data/h5path --outputpath ../data/jldpath
```

Here, `6` represents the number of available CPU cores.

## Training the Model

Before training, ensure the road network PostgreSQL server is set up by following the instructions in [Barefoot](https://github.com/boathit/barefoot).

To train the model, navigate to the `Model/ProbETA` directory and run:

```bash
cd SPTTE/Model/SPTTE
python main.py -data_path ../data/datapath
```
The dataset does not need to be manually split beforehand; it will be randomly divided into training and test sets automatically.
Once training is complete, the model will automatically run evaluation on the test set.


## Citation

If you use this repository in your research, please cite our paper:

```bibtex
@article{xu2024sptte,
  title={SPTTE: A Spatiotemporal Probabilistic Framework for Travel Time Estimation},
  author={Xu, Chen and Wang, Qiang and Sun, Lijun},
  journal={arXiv preprint arXiv:2411.18484},
  year={2024}
}
```

---

The data preprocessing part is the same as in our previous work, [ProbETA](https://arxiv.org/abs/2407.05895), and can be referred to from [here](https://github.com/ChenXu02/ProbETA).

```bibtex
@article{xu2024link,
  title={Link representation learning for probabilistic travel time estimation},
  author={Xu, Chen and Wang, Qiang and Sun, Lijun},
  journal={arXiv preprint arXiv:2407.05895},
  year={2024}
}
```



