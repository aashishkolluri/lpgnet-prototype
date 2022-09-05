# LPGNet

We suggest using Anaconda for managing the environment
## Setting up conda
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda env create -n dpgcn --file environment.yml
```

The data for linkteller is given in their official [repository](https://github.com/AI-secure/LinkTeller). Please download the zip file from their drive and extract it into the data folder.

## Hyperparameter Search

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
