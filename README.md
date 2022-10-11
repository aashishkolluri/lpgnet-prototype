# LPGNet

We provide a prototype of Link Private Graph Networks (LPGNet) that can be used for semi-supervised node classification with edge differential privacy. Please read our [paper](https://arxiv.org/abs/2205.03105) for more information.

We suggest using Anaconda for managing the environment
## Setting up conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n [env_name] --file environment.yml
```

The data for linkteller is given in their official [repository](https://github.com/AI-secure/LinkTeller). Please download the zip file from their drive and extract it into the data folder.

## Arguments and Usage
## Usage
```
usage: main.py [-h]
               [--dataset {cora,citeseer,pubmed,facebook_page,twitch/ES,twitch/RU,twitch/DE,twitch/FR,twitch/ENGB,twitch/PTBR,flickr,bipartite,chameleon}]
               --arch {mlp,mmlp,simple_mmlp,2layergcn,gcn} [--nl NL]
               [--num_seeds NUM_SEEDS] [--sample_seed SAMPLE_SEED]
               [--cuda_id CUDA_ID] [--no_cuda] [--eps EPS] [--w_dp]
               [--hidden_size HIDDEN_SIZE] [--num_hidden NUM_HIDDEN]
               [--outdir OUTDIR]
               [--test_dataset {twitch/ES,twitch/RU,twitch/DE,twitch/FR,twitch/ENGB,twitch/PTBR}]
               [--md]
               {train,evaluate,attack} ...
```
## Arguments
### Quick reference table
|Short|Long            |Default                      |
|-----|----------------|-----------------------------|
|`-h` |`--help`        |                             |
|     |`--dataset`     |`<Dataset.Cora: 'cora'>`     |
|     |`--arch`        |`<Architecture.MMLP: 'mmlp'>`|
|     |`--nl`          |`-1`                         |
|     |`--num_seeds`   |`1`                          |
|     |`--sample_seed` |`42`                         |
|     |`--cuda_id`     |`0`                          |
|     |`--no_cuda`     |                             |
|     |`--eps`         |`0.0`                        |
|     |`--w_dp`        |                             |
|     |`--hidden_size` |`16`                         |
|     |`--num_hidden`  |`2`                          |
|     |`--outdir`      |`../results`                 |
|     |`--test_dataset`|`None`                       |

### Subparser train reference table
|Short|Long            |Default                      |
|-----|----------------|-----------------------------|
|     |`--dropout`     |`0.1`                        |
|     |`--lr`          |`0.05`                       |
|     |`--num_epochs`  |`500`                        |

### Subparser attack reference table
|Short|Long            |Default                      |
|-----|----------------|-----------------------------|
|     |`--dropout`     |`0.1`                        |
|     |`--lr`          |`0.05`                       |
|     |`--influence`   |`0.001`                      |
|     |`--sample_type` |`balanced`                   |
|     |`--attack_mode` |`efficient`                  |
|     |`--n-test`      |`500`                        |
|     |`--model_path`  |                             |

### `--dataset` (Default: <Dataset.Cora: 'cora'>)
cora|citeseer|pubmed...

### `--arch` (Default: <Architecture.MMLP: 'mmlp'>)
Type of architecture to train: mmlp|gcn|mlp

### `--nl` (Default: -1)
Only use for MMLP, Number of stacked models, default=-1

### `--eps` (Default: 0.0)
The privacy budget. If 0, then do not DP train the arch

### `--w_dp`
Run with DP guarantees - if eps=0.0 it throws a warning

### `--hidden_size` (Default: 16)
Size of the hidden layers

### `--num_hidden` (Default: 2)
Number of hidden layers

### `--outdir` (Default: ../results)
Directory to save the models and results

### `--test_dataset` (Default: None)
Test on this dataset, used for Twitch

### `--sample_type` (Default: balanced)
Determines how we sample edges for attack.

### `--attack_mode` (Default: efficient)
Choose baseline running LPA and efficient for LinkTeller.

## Quick Start: Training and Attacking single models

### Run training for a single model and dataset with DP

`python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] train --lr [Lr] --dropout [Dropout]`

Here is an example to train a GCN

`python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.2`

Here is an example to train an LPGNet (mmlp) and store results in ../results

`python main.py --dataset cora --arch mmlp --nl 2 --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2  --outdir ../results train --lr 0.01 --dropout 0.2`

You can also run for multiple seeds using the --num_seeds option. The results are stored in the folder defined in globals.py or the directory specified using the --outdir option. The trained models are stored in the args.outdir/models directory.

### Run the attacks on a single trained model

To run attack on a trained model, we need all the options used for training that model and a few options in addition such as the attack_mode and sample_type (samples for evaluation).

`python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] --outdir [Outdir] **attack** --lr [Lr] --dropout [Dropout] --attack_mode [bbaseline (lpa) | efficient (linkteller)] --sample_type [balanced | unbalanced]`

Here is an example to attack an LPGNet model stored in ../results/models/

`python main.py --dataset cora --arch mmlp --nl 2 --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2 --outdir ../results attack --lr 0.01 --dropout 0.2  --attack_mode baseline --sample_type balanced`

You can also give a custom model path for attack. For instance, here is an example to attack an LPGNet (mmlp) with 2 additional mlp models. Say the models are stored in 
 1. ../results/models/mmlp_0/mmlp_0.pth (base MLP)
 2. ../results/models/mmlp_1/mmlp_1.pth (additional stack layer 1), and
 3. ../results/models/mmlp_2/mmlp_2.pth (additional stack layer 2)

`python main.py --dataset cora --arch mmlp --nl 2 --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2 --outdir ../results attack --lr 0.01 --dropout 0.2 --model_path mmlp_0/mmlp_0.pth,mmlp_1/mmlp_1.pth,mmlp_2/mmlp_2.pth --attack_mode efficient --sample_type unbalanced`

The attack results are stored inthe directory with name _eval\_[dataset]_ which is placed in the current directory.

## Reproducing the results

To reproduce the results we provide a script in run_exp.py. You can write your own scipt as well that follows the general procedure explained below.

### Hyperparameter search for transductive datasets

`python run_exp.py --num_seeds 30 --command train --outdir ../data-hyperparams --hyperparameters --todos_dir [todos]`

or if you wish to run on more GPUs

`for cuda in $(seq 0 7); do python run_exp.py --num_seeds 30 --command train --outdir ../data-hyperparams --hyperparameters --todos_dir [todos] --cuda_id $cuda  & done`


### Hyperparameter search for inductive datasets

`python run_exp.py --num_seeds 30 --command train --outdir ../data-hyperparams-inductive --hyperparameters --todos_dir [todos] --inductive`

or if you wish to spawn one process per GPU (0-7)

`for cuda in $(seq 0 7); do python run_exp.py --num_seeds 30 --command train --outdir ../data-hyperparams-inductive --hyperparameters --todos_dir [todos] --cuda_id $cuda --inductive & done`

Note: For Twitch datasets please add this option --num_epochs 200


### Parse the best configuration for (dataset, architecture)

`python run_exp.py --parse_config_dir [data-hyperparameters-dir]`

This creates a file `best_config.pkl` in the local directory which contains the best hyperparameters.

## Training Models

### Run the training based on the best hyperparameters for transductive

`python run_exp.py --num_seeds 30 --command train --outdir [results-dir] --todos_dir [todos] --best_config_file [best_config.pl]`

The [results-dir] and [todos] can be any directory paths where you want to save the results and cache the todo/finished tasks respectively.

### Run the training based on the best hyperparameters for inductive
For Twitch datasets

``python run_exp.py --num_seeds 30 --num_epochs 200 --command train --inductive --datasets TwitchES --best_config_file [best_config.pl] --outdir [results-dir] --todos_dir [todos]``

For Flickr dataset

``python run_exp.py --inductive --datasets Flickr --command train --num_seeds 5 --todos_dir [todos] --outdir [results-dir] --num_epochs 500``

This command will take large resources (close to 100GB memory per such command). This will also cache additional matrices in your data directory where the flickr training matrices are loaded from.
 
## Attacking the Trained Models

### Run the attacks on the trained models for transductive

For attacks the --outdir option is used to provide the path to the trained models which is the same as the corresponding path used to run the training command. The attack commands save the results in the current directory with "eval_" as a prefix.

`python run_exp.py --num_seeds 5 --command attack --best_config_file [best_config.pl] --outdir [results-dir] --todos_dir [todos]`

### Run the attacks on the trained models for inductive
For Twitch datasets

`python run_exp.py --num_seeds 5 --num_epochs 200 --command attack --datasets TwitchES --best_config_file [best_config.pl] --outdir [results-dir] --todos_dir [todos]`

For Flickr dataset

`python run_exp.py --outdir [results-dir] --todos_dir [todos] --num_seeds 5 --command attack --datasets Flickr`

## Parsing the Results

### Parse results to get utility scores
Provide path to the results directory used during the training.

For transductive

`python parser_ash.py --results_dir [results-dir]`

For inductive

`python parser_ash_ind_utility.py --results_dir [results-dir]`

### Parse results to get attack AUC scores
For both transductive and inductive provide the path to the directory with saved models ( "eval_" as prefix)

`python parser_ash_trans_attack.py --results_dir [results-dir]`

All the parsed results will be output in the results folder.
