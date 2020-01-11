# Adversarial-Social-Recommendation
This repository contains code for the CIKM 2019 publication titled "A Modular Adversarial Approach to Social Recommendation". 

If this code is helpful in your research, please cite the following publication

> Krishnan, Adit, et al. "A modular adversarial approach to social recommendation." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.

## Getting Started

These instructions will help you reuse and modify our framework. The basic framework uses VAE-CF () as the Generator (Base Recommender) and a GCN-based Discriminator.

### Platforms Supported

- Unix, MacOS, Windows (with appropriate compilers/environment)

### Prerequisites

We recommend using this code with a Python-2 Anaconda Tensorflow setup. Our anaconda environment can be reproduced with the following command:

```
$ conda env create -f author.yml
```

## Input File Format

For a paricular dataset, the model requires the following input files:

- **item2id.txt**: Each row of this file corresponds to a specific item. The original item IDs (from the actual dataset) are mapped to a conitnuous set of integers to be used in all other input files.
- **profile2id.txt**: This is the same as item2id, except it maps each user ID to an integer value. 
- 

Both the files should be placed inside the **Data** folder.

## Running the Model

The model can be executed using the following command.

```
$ ./CMAP <options>
```

where possible options include:

```
--dataset <name>:         Name of the dataset to use (required)
--model <type>:           Model-Type. 0 for unified, 1 for factored (default 0)
--hr:                     Use the hierarachical version of the model
--thread <num_thread>:    Use the threaded version of the model. Specify number of threads to use
--G <num_groups>:         Specify the value of number of groups (default: 20)
--K <num_topics>:         Specify number of topics (Use only for unified model) (default: 20)
--K_w <num_text_topics>:  Specify number of text topics (Use only for factored model) (default: 20)
--K_b <num_behav_topics>: Specify number of behavior topics (Use only for factored model) (default: 5)
--scale <s>:              Specify the value of scale parameter (default: 1.5)
--discount <d>:           Specify the value of dicount parameter (default: 0.5)
--iter <num_iter>:        Specify the number of iterations to run (default: 500)
--help:                   Print help
```

Providing the dataset name is mandatory. Please refer to the paper for optimal values of these parameters.

## Sample Run
A sample dataset named biology is present in the Data folder. For running the hierarchical variation of the unified model with G = 20 and K = 10 for 100 iterations, execute the following command:
```
$ ./CMAP --dataset biology --model 0 --hr --G 20 --K 10 --iter 100
```
