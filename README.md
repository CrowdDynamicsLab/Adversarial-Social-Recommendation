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

- **<dataset>_pre_processed.txt**: Each row of this file corresponds to one data point in your dataset and has 4 columns - Text, UserId, Behaviour and Timestamp (all tab separated).  The columns are described below:
    - **Text**: The text in your data point. Pre-process the text for efficient use.
    - **UserId**: A user index between 0 to num_users-1 corresponding to the user of the data point.
    - **Behaviour**: The action observed in the data point. Eg, questioning, answering, commenting, etc. for Stack-Exchanges
    - **Timestamp**: The normalized value of time of data point. The value must be between 0.01 to 0.99 and should be truncated to 2 decimal places. 
- **<dataset>_links.txt**: This file is optional and can be provided if you have social interaction information as part of your dataset. Each row corresponds to a link from a data point i to data point j and has 2 colums - i and j (tab-separated). The data points are zero-indexed and indexing is defined from the <dataset>_pre_processed file.

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
