import os
from tkinter import N
from models import TwoLayerGCN, MLP, MultiMLP
import trainer
import utils
from data import LoadData
import argparse
import numpy as np
import torch
import time
import attacker
import torch.nn as nn
import models
from globals import MyGlobals


def run_training(
    run_config, arch, dataset, device, dp=False, seeds=[1], test_dataset=None
):
    print("Train and evaluate arch {}; seeds {}".format(arch.name, seeds))
    start_time = time.time()
    if test_dataset:
        print("Transfer learning with separate test graph")

    if arch == utils.Architecture.GCN:
        train_stats = trainer.train_gcn_on_dataset(
            run_config, dataset, device, dp=dp, seeds=seeds, test_dataset=test_dataset
        )
    elif arch == utils.Architecture.MLP:
        if run_config.eps > 0 or dp == True:
            print("MLP does not require DP")
            exit()
        train_stats = trainer.train_mlp_on_dataset(
            run_config,
            dataset,
            device,
            iter=len(seeds),
            seeds=seeds,
            test_dataset=test_dataset,
        )
    elif arch == utils.Architecture.MMLP:
        train_stats = trainer.train_mmlp_on_dataset(
            run_config,
            dataset,
            device,
            dp=dp,
            iter=len(seeds),
            seeds=seeds,
            test_dataset=test_dataset,
        )
    else:
        print("Arch {} not supported".format(arch))
        return None
    print(f"Training time: {time.time()-start_time}")
    return train_stats


def train(args):
    # Some sanity checks on the arguments
    if args.w_dp == True and args.eps == 0.0:
        print("You selected with DP but eps=0.0")
        exit()

    if args.w_dp == False and args.eps > 0:
        print("No DP selected")
        exit()

    if (args.w_dp == True or args.eps > 0) and args.arch == utils.Architecture.MLP:
        print("MLP is by default private, no need to select DP")
        exit()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Run the train/attack/eval on the selected GPU id
        if torch.cuda.is_available:
            torch.cuda.set_device(args.cuda_id)
            print("Current CUDA device: {}".format(torch.cuda.current_device()))

    run_config = trainer.RunConfig(
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_each_epoch=False,
        save_epoch=args.save_epoch,
        weight_decay=MyGlobals.weight_decay,
        output_dir=os.path.join(args.outdir, "models"),
        eps=args.eps,
        nl=args.nl,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden,
        dropout=args.dropout,
    )

    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)

    print("Running training")
    train_stats = run_training(
        run_config,
        args.arch,
        args.dataset,
        device,
        dp=args.w_dp,
        seeds=seeds,
        test_dataset=args.test_dataset,
    )

    if (
        args.arch == utils.Architecture.MMLP
        or args.arch == utils.Architecture.SimpleMMLP
    ):
        args.arch = args.arch.name + "_nl" + str(args.nl)

    if args.dataset == utils.Dataset.TwitchES:
        es_identifier = args.dataset.value[args.dataset.value.find("/") + 1 :]
        identifier = args.test_dataset.value[args.test_dataset.value.find("/") + 1 :]
        args.dataset = (args.dataset, args.test_dataset)
        dataset_name = es_identifier + "_" + identifier
    else:
        dataset_name = str(args.dataset)

    results_dict = {args.dataset: {args.eps: {}}}
    results_dict[args.dataset][args.eps][args.arch] = train_stats

    utils.save_results_pkl(
        results_dict, args.outdir, args.arch, dataset_name, run_config
    )


def attack(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available:
        print(args.cuda_id)
        torch.cuda.set_device(args.cuda_id)
        print("Current CUDA device: {}".format(torch.cuda.current_device()))

    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)

    run_config = trainer.RunConfig(
        learning_rate=args.lr,
        num_epochs=500,
        save_each_epoch=False,
        save_epoch=None,
        weight_decay=MyGlobals.weight_decay,
        output_dir=os.path.join(args.outdir, "models"),
        eps=args.eps,
        nl=args.nl,
        hidden_size=args.hidden_size,
        num_hidden=args.num_hidden,
        dropout=args.dropout
    )
    model_paths = args.model_path.split(",")
    print(run_config, model_paths)
    for i in range(len(seeds)):
        seed = seeds[i]
        trainer.set_torch_seed(seed)
        rng = np.random.default_rng(seed=seed)
        run_attack(
            args.dataset,
            args.arch,
            run_config,
            model_paths,
            device,
            args.influence,
            args.attack_mode,
            args.sample_type,
            args.mode,
            seed,
            rng,
            args.outdir,
            args.n_test,
            args.test_dataset,
            spnum = i%len(seeds)
        )
    return


def run_attack(
    dataset,
    arch,
    run_config,
    model_paths,
    device,
    influence,
    attack_mode,
    sample_type,
    mode,
    seed,
    rng,
    outdir,
    n_test,
    test_dataset,
    spnum = 0
):
    if run_config.eps > 0 and arch==utils.Architecture.GCN:
        dp_for_loading_data = True
    else:
        dp_for_loading_data = False
    data_loader = LoadData(
        dataset, eps=run_config.eps, dp=dp_for_loading_data, rng=rng, test_dataset=test_dataset, split_num_for_geomGCN_dataset=spnum
    )
    print("Loaded data")
    comms_file = None
    if arch == utils.Architecture.MMLP:
        comms_file = utils.get_comms_pkl_file_name(f'mmlp_nl{run_config.nl}', dataset, seed, test_dataset, run_config)
    model = load_model(
        arch, run_config, model_paths, device, data_loader, outdir, rng, attack=True, comms_file=comms_file
    )

    features = data_loader.test_features.to(device)
    test_labels = data_loader.test_labels.to(device)
    if arch == utils.Architecture.GCN:
        adjacency_matrix = data_loader.test_adj_csr.to(
            device
        )  # This is for the attack after loading the model
    else:
        adjacency_matrix = None  # No adjacency matrix required for MLP and LPGNet
    # if arch == utils.Architecture.MMLP:
    #    model.prepare_for_fwd(features, test_labels)
    attack = attacker.Attacker(
        dataset.value,
        model,
        features,
        test_labels,
        adjacency_matrix,
        data_loader.test_adj_orig_csr,
        influence,
        attack_mode,
        mode,
        sample_type,
        seed,
        n_test,
        rng,
        test_dataset,
    )

    if attack_mode == "efficient":
        if sample_type == "balanced_full" or sample_type == "balanced":
            (
                norm_exist,
                norm_nonexist,
            ) = attack.link_prediction_attack_efficient_balanced()
        else:
            norm_exist, norm_nonexist = attack.link_prediction_attack_efficient()
    elif attack_mode == "baseline" and sample_type in ["balanced_full", "balanced"]:
        norm_exist, norm_nonexist = attack.baseline_attack_balanced()
    elif attack_mode == "baseline" and sample_type in [
        "unbalanced",
        "unbalanced_hi",
        "unbalanced_lo",
    ]:
        norm_exist, norm_nonexist = attack.baseline_attack()
    else:
        print("{} not supported".format(attack_mode))

    if arch == utils.Architecture.MMLP or arch == utils.Architecture.SimpleMMLP:
        if test_dataset:
            prefix = (
                f"{arch.value}_nl{run_config.nl}-{dataset.name},{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
        else:
            prefix = (
                f"{arch.value}_nl{run_config.nl}-{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
    else:
        if test_dataset:
            prefix = (
                f"{arch.value}_nl{run_config.nl}-{dataset.name},{test_dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
        else:
            prefix = (
                f"{arch.value}_nl{run_config.nl}-{dataset.name}-"
                f"seed_{seed}-lr_{run_config.learning_rate}-hidden_size_{run_config.hidden_size}-"
                f"num_hidedn_{run_config.num_hidden}-dropout_{run_config.dropout}-"
                f"{run_config.eps}"
            )
    attack.compute_and_save(norm_exist, norm_nonexist, prefix)


def load_model(
    arch, run_config, model_paths, device, data_loader, outdir, rng, attack=False, comms_file=None
):
    numpy_adjacency_matrix = None
    if arch == utils.Architecture.MMLP:
        if attack or data_loader.is_inductive():
            numpy_adjacency_matrix = data_loader.test_adj_orig_csr.toarray()
        else:
            numpy_adjacency_matrix = data_loader.train_adj_orig_csr.toarray()

    if arch == utils.Architecture.SimpleMMLP:
        print("{} Not implemented. Skip.".format(arch))
        exit()

    eps = run_config.eps
    if arch == utils.Architecture.MMLP:
        eps = eps*1.0/run_config.nl
        if data_loader.dataset == utils.Dataset.Flickr:
            eps = eps/3.0
            print(f"For flickr changing eps from {eps} to {eps/3.0}")

    model = models.create_model(
        run_config,
        arch,
        data_loader.train_features.size(1),
        data_loader.num_classes,
        device=device,
        adjacency_matrix=numpy_adjacency_matrix,
        eps=eps,
        rng=rng,
    )

    path = [os.path.join(outdir, "models", m_path) for m_path in model_paths]
    if arch == utils.Architecture.MMLP:
        model.load_model_from(path, device, comms_file)
    else:
        model.load_model_from(path, device)
    return model


def load_and_test(args):
    """
    Currently works only if given all paths to models.
    Have to write something that can automatically parse them
    """
    print(args)
    print("Load and evaluate {} model".format(args.model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run the train/attack/eval on the selected GPU id
    if torch.cuda.is_available:
        torch.cuda.set_device(args.cuda_id)
        print("Current CUDA device: {}".format(torch.cuda.current_device()))

    run_config = trainer.RunConfig(
        eps=args.eps,
        nl=args.nl,
        num_hidden=args.num_hidden,
        hidden_size=args.hidden_size,
    )
    model_paths = args.model_path.split(",")

    accuracies = []
    seeds = utils.get_seeds(args.num_seeds, args.sample_seed)
    for seed in seeds:
        rng = np.random.default_rng(seed=seed)
        if args.arch == utils.Architecture.GCN:
            data_loader = LoadData(
                args.dataset,
                load_dir=str(args.dataset),
                dp=args.w_dp,
                eps=args.eps,
                rng=rng,
                test_dataset=args.test_dataset,
            )
        else:
            data_loader = LoadData(
                args.dataset, dp=False, rng=rng, test_dataset=args.test_dataset
            )

        model = load_model(
            args.arch, run_config, model_paths, device, data_loader, args.outdir, rng
        )
        is_rare = 'twitch' in args.dataset.value
        if args.arch == utils.Architecture.MMLP:
            test_loss, test_acc, f1_score, rare_f1_score = trainer.evaluate_mmlp(
                model,
                data_loader.test_features,
                data_loader.test_labels,
                data_loader.test_adj_orig_csr.toarray(),
                device,
                is_rare,
            )
        elif args.arch == utils.Architecture.GCN:
            tr = trainer.Trainer(model, rng, seed=seed)
            test_loss, test_acc, f1_score, rare_f1_score = tr.evaluate(
                data_loader.test_features,
                data_loader.test_labels,
                device,
                data_loader.test_adj_csr,
                test_dataset=args.test_dataset,
                is_rare=is_rare,
            )
        elif args.arch == utils.Architecture.MLP:
            tr = trainer.Trainer(model, rng, seed=seed)
            test_loss, test_acc, f1_score, rare_f1_score = tr.evaluate(
                data_loader.test_features,
                data_loader.test_labels,
                device,
                test_dataset=args.test_dataset,
                is_rare=is_rare,
            )
        print(
            f"\nPerformance on {args.dataset.name}:\n- "
            f"test accuracy = {test_acc:.3f} +- 0.000\n"
        )
        accuracies.append(test_acc)

    return accuracies


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=utils.Dataset,
        choices=utils.Dataset,
        default=utils.Dataset.Cora,
    )
    parser.add_argument(
        "--arch",
        type=utils.Architecture,
        choices=utils.Architecture,
        default=utils.Architecture.MMLP,
        required=True,
        help="Type of architecture to train",
    )
    parser.add_argument(
        "--nl", type=int, default=MyGlobals.nl, help="Number of stacked models"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=MyGlobals.num_seeds,
        help="Run over num_seeds seeds",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=MyGlobals.sample_seed,
        help="Run for this seed",
    )
    parser.add_argument("--cuda_id", type=int, default=MyGlobals.cuda_id)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument(
        "--eps",
        type=float,
        default=MyGlobals.eps,
        help="The privacy budget. If 0, then do not DP train the arch",
    )
    parser.add_argument(
        "--w_dp",
        default=MyGlobals.with_dp,
        action="store_true",
        help="Run with DP guarantees - if eps=0.0 it throws a warning",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=MyGlobals.hidden_size,
        help="Size of the first hidden layer",
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=MyGlobals.num_hidden,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=MyGlobals.RESULTDIR,
        help="Directory to save the models and results",
    )
    parser.add_argument(
        "--test_dataset", type=utils.Dataset, choices=utils.Dataset, default=None
    )

    # Train model commands
    # Should add more -- perhaps set a config file to make it easier to set all of these parameters
    subparsers = parser.add_subparsers(help="sub-command help")

    train_parser = subparsers.add_parser("train", help="train sub-menu help")
    train_parser.add_argument(
        "--lr", type=float, default=MyGlobals.lr, help="Learning rate"
    )
    train_parser.add_argument("--num_epochs", type=int, default=MyGlobals.num_epochs)
    train_parser.add_argument(
        "--save_epoch",
        type=int,
        default=MyGlobals.save_epoch,
        help="Save at every save_epoch",
    )
    train_parser.add_argument("--dropout", type=float, default=MyGlobals.dropout)
    train_parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Store the results in a pkl in args.outdir",
    )
    train_parser.set_defaults(func=train)

    # Load and evaluate trained model commands
    eval_parser = subparsers.add_parser("evaluate", help="test sub-menu help")
    eval_parser.add_argument("--model_path", type=str, default="")
    eval_parser.set_defaults(func=load_and_test)

    # model commands
    attack_parser = subparsers.add_parser("attack", help="attack sub-menu help")
    attack_parser.add_argument(
        "--lr", type=float, default=MyGlobals.lr, help="Learning rate"
    )
    attack_parser.add_argument("--dropout", type=float, default=MyGlobals.dropout)
    attack_parser.add_argument("--model_path", type=str, required=True)
    attack_parser.add_argument("--influence", type=float, default=MyGlobals.influence)
    attack_parser.add_argument(
        "--sample_type",
        type=str,
        default="balanced",
        choices=[
            "balanced",
            "unbalanced",
            "unbalanced_lo",
            "unbalanced_hi",
            "balanced_full",
        ],
    )
    attack_parser.add_argument(
        "--save_epoch",
        type=int,
        default=MyGlobals.save_epoch,
        help="Save at every save_epoch",
    )
    attack_parser.add_argument("--approx", action="store_true", default=False)
    attack_parser.add_argument(
        "--attack_mode",
        type=str,
        default="efficient",
        choices=["efficient", "naive", "baseline", "baseline-feat"],
    )
    attack_parser.add_argument(
        "--mode",
        type=str,
        default="vanilla-clean",
        help="[ vanilla | vanilla-clean | clusteradj | clusteradj-clean ] ",
    )
    attack_parser.add_argument(
        "--n-test", type=int, default=MyGlobals.n_test, help="The number of nodes"
    )
    attack_parser.set_defaults(func=attack)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
