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

For a paricular dataset, the model requires the following input files (samples of every input file can be found in our sample_dataset folder):

- **item2id.txt**: Each row of this file corresponds to a specific item. The original item IDs (from the actual dataset) are mapped to a conitguous set of integers to be used in all other input files.
- **profile2id.txt**: This is the same as item2id, except it maps user IDs to contiguous integer values. For both users and items, the mapped continuous integer values (and not the original IDs) are used in the input files.
- **all_ratings.csv**: Contains the set of all user likes or positive ratings in the entire dataset. Each line is a user index followed by an item they liked or rated positively.
- **train.csv**: The train subset of all_ratings, containing about 80% of the users and all their ratings.
- **val_tr, val_te.csv**: The validation and test sets contain about 10% of the users each. For each user in the validation and test sets, their ratings are split into 2 groups - input (tr) and output (te). Since VAE is an inductive model, at validation time the input (tr) part of a specific user's ratings is provided as input to the VAE and evaluation attempts to predict the output for the user, namely the val_te part.
- **user_link.csv**: The set of all social links, one per line. Each user must have atleast one social link. We created dummy links (with a unique fake user) for each user without social connections.

All the above files are placed in the dataset directory.

## Running the Model

The model first needs a set of predetermined social embeddings, for which we used a Graph Auto-Encoder (GAE). To run GAE, execute the following command inside the gae-master/gae/ path (the author tensorflow environment works with GAE as well):

```
$ source activate tensorflow_environment_with_prerequisites
$ cd gae-master/gae/
$ python train.py <path-to-dataset-directory>
```

After GAE is executed, two additional files are created in the dataset directory:

- **user_emb.npy**: Numpy file containing the set of social embeddings for users, as computed by GAE.
- **user_p.npy**: Pre-computed social proximities of users, obtained by the dot product of their social embeddings. These proximities are used to create true pairs for the GAN.


To run the actual model, execute train.py in the home path. 

```
$ python train.py
```

By default the model is set to run on the sample dataset. To run the model on a different dataset or modify model options, go to flags.py. The list of flags are also described in the train.py file.
