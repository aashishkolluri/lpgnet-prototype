# LPGNet

We suggest using Anaconda for managing the environment
## Setting up conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n [env_name] --file environment.yml
```

The data for linkteller is given in their official [repository](https://github.com/AI-secure/LinkTeller). Please download the zip file from their drive and extract it into the data folder.

## Quick Start: Training and Attacking single models

### Run training for a single model and dataset with DP

`python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] train --lr [Lr] --dropout [Dropout]`

Here is an example to train a GCN

`python main.py --dataset cora --arch gcn --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2 train --lr 0.01 --dropout 0.2`

Here is an example to train an LPGNet (mmlp) and store results in ../results

`python main.py --dataset cora --arch mmlp --nl 2 --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2  --outdir ../results train --lr 0.01 --dropout 0.2`

You can also run for multiple seeds using the --num_seeds option. The results are stored in the folder defined in globals.py or the directory specified using the --outdir option. The trained models are stored in the args.outdir/models directory.

### Run the attacks on a single trained model

To run attack on a trained model, we need all the options used for training that model and a few more options in addition such as the attack_mode and sample_type (samples for evaluation).

`python main.py --dataset [Dataset] --arch [mmlp|gcn|mlp] --nl [# stack layers for mmlp] --w_dp --eps [Eps] --sample_seed [Seed] --hidden_size [HID_s] --num_hidden [HID_n] --outdir [Outdir] **attack** --lr [Lr] --dropout [Dropout] --attack_mode [bbaseline (lpa) | efficient (linkteller)] --sample_type [balanced | unbalanced]`

Here is an example to attack a GCN model stored in ../results/models/

`python main.py --dataset cora --arch mmlp --nl 2 --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2 --outdir ../results attack --lr 0.01 --dropout 0.2  --attack_mode baseline --sample_type balanced`

You can also give a custom model path for attack. For instance, here is an example to attack an LPGNet (mmlp) with 2 additional mlp models. Say the models are stored in ../results/models/mmlp_0/mmlp_0.pth (base MLP), ../results/models/mmlp_1/mmlp_1.pth (additional stack layer 1), and ../results/models/mmlp_2/mmlp_2.pth (additional stack layer 2)

`python main.py --dataset cora --arch mmlp --nl 2 --w_dp --eps 4.0 --sample_seed 42 --hidden_size 256 --num_hidden 2 --outdir ../results attack --lr 0.01 --dropout 0.2 --model_path mmlp_0/mmlp_0.pth,mmlp_1/mmlp_1.pth,mmlp_2/mmlp_2.pth --attack_mode efficient --sample_type unbalanced`


## Reproducing the results

To reproduce the results we provide a script in run_exp.py. You can write your own scipt as well by following the general procedure explained below.

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


### Evaluation Data
We provide all the saved models, best configuration files, training and attack results in this [drive](https://drive.google.com/file/d/1c_J6uEe5LcmKB_ZyAMzld0rLkcJZi5IY/view?usp=sharing). The folder is large, so if you are not able to view it then we can provide the data in a different way on request.
