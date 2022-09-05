from audioop import avg
import matplotlib.pyplot as plt
import numpy as np
import torch
import operator
import pickle as pkl
from data import LoadData
from utils import Dataset, Architecture
from graph_utils import getCommunityCountsMP
from main import load_model
import os
import trainer
import utils

def plot_avg_norm_dv(dvs, labels, filename="dv_arg_norm.png"):
    cmap = {0:'b', 1:'r', 2:'g', 3:'y', 4:'c', 5:'m', 6:'k', 7:'slateblue', 8:'orange', 9:'grey'}
    C = [cmap[label] for label in labels]
    comm_dict = {}
    for i in range(len(dvs)):
        if not labels[i] in comm_dict:
            comm_dict[labels[i]] = []
        comm_dict[labels[i]].append(i)
    comms = [val for key, val in sorted(comm_dict.items(), key=operator.itemgetter(0))]
    dvs_nor_np = torch.nn.functional.normalize(dvs, p=1.0, dim=1).cpu().numpy()
    for j in range(len(comms)):
        comm = comms[j]
        col = cmap[j]
        avg_dvs_nor_np = dvs_nor_np[comm].mean(axis=0)
        plt.bar(range(len(comms)), avg_dvs_nor_np, color=col)
    plt.savefig(filename, dpi=300)

def plot_avg_norm_dv_1(avg_dvs_nor_np, filename="dv_arg_norm.png"):
    cmap = {0:'b', 1:'r', 2:'g', 3:'y', 4:'c', 5:'m', 6:'k', 7:'slateblue', 8:'orange', 9:'grey'}
    for j in range(len(avg_dvs_nor_np)):
        col = cmap[j]
        plt.ylim(0.0, 1.0)
        plt.bar(range(len(avg_dvs_nor_np)), avg_dvs_nor_np[j], color=col)
    plt.savefig(filename, dpi=300)
    plt.close()

def get_avg_nor_given_labels(dvs, labels, act_lables=[]):
    comm_dict = {}
    for i in range(len(dvs)):
        if not labels[i] in comm_dict:
            comm_dict[labels[i]] = []
        comm_dict[labels[i]].append(i)
    
    for key in act_lables:
        if not key in comm_dict:
            comm_dict[key] = []
    comms = [val for key, val in sorted(comm_dict.items(), key=operator.itemgetter(0))]
    dvs_nor_np = torch.nn.functional.normalize(dvs, p=1.0, dim=1).cpu().numpy()
    avg_dvs_nor_np = []
    for j in range(len(comms)):
        comm = comms[j]
        if len(comm)==0:
            m = np.array([0.0]*len(dvs_nor_np[0]))
        else:
            m = dvs_nor_np[comm].mean(axis=0)
        avg_dvs_nor_np.append(m)
    return avg_dvs_nor_np

def get_avg_dvs_nor_test(dataset, eps, outdir, arch=None, nl=1, nh=2, hs=256, dropout=0.5, is_inductive=False, seeds = [], lr = 0.01):

    if arch is None:
        rng = np.random.default_rng(seeds[0])
        test_dataset = None
        if 'twitch' in dataset.value:
            test_dataset = dataset
            # dataset = Dataset.TwitchES
        data_loader = LoadData(dataset, test_dataset=test_dataset, rng=rng)
        features = data_loader.features
        labels = data_loader.labels
        num_classes = data_loader.num_classes
        orig_adj_np = data_loader.train_adj_orig_csr.toarray()
        np_labels = labels.cpu().numpy()
        dvs = getCommunityCountsMP(orig_adj_np, np_labels, num_classes, rng, eps=0.0)
        dvs_tor = torch.from_numpy(
                    np.array([dvs[j] for j in dvs])).type(torch.float32)
        return get_avg_nor_given_labels(dvs_tor, np_labels)
    else:
        avg_dvs_nor_per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            dp = False
            if eps > 0.0:
                dp = True
            test_dataset = None
            if 'twitch'  in dataset.value:
                test_dataset = dataset
                data_loader = LoadData(Dataset.TwitchES, dp=dp, test_dataset=test_dataset, rng=rng)
            else:
                data_loader = LoadData(dataset, dp=dp, test_dataset=test_dataset, rng=rng)
            features = data_loader.features
            labels = data_loader.labels
            adjacency_matrix = data_loader.train_adj_csr
            num_classes = data_loader.num_classes
            orig_adj_np = data_loader.train_adj_orig_csr.toarray()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # arch_mmlp_2_2-dataset_PubMed-seed_972-lr_0.001-hidden_size_256-num_hidden_2-dropout_0.1-eps_5.0
            dirpath = os.path.join(outdir, 'models')
            t_str  = f"{dataset.name}" if not 'twitch' in dataset.value else f"TwitchES_{test_dataset.name}"
            if arch==Architecture.MMLP:
                if not os.path.isfile(dirpath+'/'+f"arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth"):
                    print("Not found: ", dirpath+'/'+f"arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-lr_{lr}-eps_{eps}.pth")
                    continue
                model_paths = ",".join([f"arch_mmlp_{nl}_{j}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_mmlp_{nl}_{j}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth" for j in range(0, nl+1)])
            else:
                arch_name = arch.value if arch.value=="mlp" else "2layergcn" if nh==2 else "3layergcn" 
                if not os.path.isfile(dirpath + '/'+f"arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth"):
                    print("Not found: ", dirpath + '/'+f"arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth")
                    continue
                model_paths = f"arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth"

            test_nodes = np.array(range(len(features)))
            # if is_inductive:
            #     test_nodes = np.array(range(len(data_loader.test_features)))
            run_config = trainer.RunConfig(
                eps=eps,
                nl=nl if arch == Architecture.MMLP else 0,
                num_hidden=int(model_paths.split("num_hidden_")[1][0]),
                hidden_size=int(model_paths.split("hidden_size_")[1].split("-num")[0])
            )
            model = load_model(arch, run_config, model_paths.split(","), device, data_loader, outdir, rng)
            if not arch==Architecture.MMLP:
                model  = model.to(device)
            features = features.to(device)
            adjacency_matrix = adjacency_matrix.to(device)
            labels = labels.to(device)
            if arch == Architecture.GCN:
                outputs, _ = model(features, adjacency_matrix, labels)
            else:
                outputs, _ = model(features, labels)
            # get labels
            pred_labels = torch.max(outputs, dim=1)[1].detach().cpu().numpy()
            mislabeled = np.where(pred_labels-labels.cpu().numpy() !=0)[0]
            # get degree vectors
            dvs = getCommunityCountsMP(orig_adj_np, pred_labels, num_classes, rng, eps=eps)
            dvs_tor = torch.from_numpy(
                np.array([dvs[j] for j in dvs])).type(torch.float32)
            avg_dvs_nor_per_seed.append(get_avg_nor_given_labels(dvs_tor[test_nodes], pred_labels[test_nodes], labels.cpu().numpy()[test_nodes]))

        return [
                np.stack([
                        avg_dvs_nor_per_seed[k][j] 
                        for k in range(len(avg_dvs_nor_per_seed))
                        ]).mean(axis=0) 
                    for j in range(len(avg_dvs_nor_per_seed[0]))
                ]

def get_per_class_stats(dataset, eps, outdir, arch=None, nl=1, nh=2, hs=256, dropout=0.5, is_inductive=False, seeds = [], lr = 0.01):

    if arch is None:
        rng = np.random.default_rng(seeds[0])
        test_dataset = None
        if 'twitch' in dataset.value:
            test_dataset = dataset
            # dataset = Dataset.TwitchES
        data_loader = LoadData(dataset, test_dataset=test_dataset, rng=rng)
        features = data_loader.features
        labels = data_loader.labels
        num_classes = data_loader.num_classes
        orig_adj_np = data_loader.train_adj_orig_csr.toarray()
        np_labels = labels.cpu().numpy()
        label_counts = [len(np.where(np_labels==label)[0]) for label in range(np.max(np_labels)+1)]
        assert(np.sum(label_counts)==len(np_labels))
        return label_counts
    else:
        res_per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            dp = False
            if eps > 0.0:
                dp = True
            test_dataset = None
            if 'twitch'  in dataset.value:
                test_dataset = dataset
                data_loader = LoadData(Dataset.TwitchES, dp=dp, test_dataset=test_dataset, rng=rng)
            else:
                data_loader = LoadData(dataset, dp=dp, test_dataset=test_dataset, rng=rng)
            features = data_loader.features
            labels = data_loader.labels
            if is_inductive:
                features = data_loader.test_features
                labels = data_loader.test_labels
            np_labels = labels.cpu().numpy()
            adjacency_matrix = data_loader.train_adj_csr
            if is_inductive:
                adjacency_matrix = data_loader.test_adj_csr

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # arch_mmlp_2_2-dataset_PubMed-seed_972-lr_0.001-hidden_size_256-num_hidden_2-dropout_0.1-eps_5.0
            dirpath = os.path.join(outdir, 'models')
            t_str  = f"{dataset.name}" if not 'twitch' in dataset.value else f"TwitchES_{test_dataset.name}"
            if arch==Architecture.MMLP:
                if not os.path.isfile(dirpath+'/'+f"arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth"):
                    print("Not found: ", dirpath+'/'+f"arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_mmlp_{nl}_{1}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-lr_{lr}-eps_{eps}.pth")
                    continue
                model_paths = ",".join([f"arch_mmlp_{nl}_{j}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_mmlp_{nl}_{j}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth" for j in range(0, nl+1)])
            else:
                arch_name = arch.value if arch.value=="mlp" else "2layergcn" if nh==2 else "3layergcn" 
                if not os.path.isfile(dirpath + '/'+f"arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth"):
                    print("Not found: ", dirpath + '/'+f"arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth")
                    continue
                model_paths = f"arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}/arch_{arch_name}-dataset_{t_str}-seed_{seed}-lr_{lr}-hidden_size_{hs}-num_hidden_{nh}-dropout_{dropout}-eps_{eps}.pth"

            # test_nodes = np.array(range(len(features)))
            test_labels = data_loader.test_labels
            ignore_label = torch.nn.CrossEntropyLoss().ignore_index
            test_nodes = np.where(test_labels.cpu().numpy()!=ignore_label)[0]
            # if is_inductive:
            #     test_nodes = np.array(range(len(data_loader.test_features)))
            run_config = trainer.RunConfig(
                eps=eps,
                nl=nl if arch == Architecture.MMLP else 0,
                num_hidden=int(model_paths.split("num_hidden_")[1][0]),
                hidden_size=int(model_paths.split("hidden_size_")[1].split("-num")[0])
            )
            model = load_model(arch, run_config, model_paths.split(","), device, data_loader, outdir, rng)
            if not arch==Architecture.MMLP:
                model  = model.to(device)
            features = features.to(device)
            adjacency_matrix = adjacency_matrix.to(device)
            labels = labels.to(device)
            if arch == Architecture.GCN:
                outputs, _ = model(features, adjacency_matrix, labels)
            else:
                outputs, _ = model(features, labels)
            # get labels
            pred_labels = torch.max(outputs, dim=1)[1].detach().cpu().numpy()
            mislabeled = np.where(pred_labels-labels.cpu().numpy() !=0)[0]

            accs, precs, recs, f1s, n_labels = [],[],[],[],[]
            for label in range(np.max(np_labels)+1):
                n_l = np.where(test_labels.cpu().numpy()==label)[0]
                acc = len(np.where(pred_labels[n_l]==label)[0]) / len(n_l)
                if len(np.where(pred_labels==label)[0])==0:
                    prec = 0.0
                else:
                    prec = len(np.where(pred_labels[n_l]==label)[0]) / len(np.where(pred_labels==label)[0])
                rec = acc
                f1 = 2 * prec * rec / (prec + rec + 0.000000001)
                n_labels.append(len(n_l))
                accs.append(acc)
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
            res_per_seed.append([n_labels, accs, precs, recs, f1s])
        
        return [
                np.stack([
                        res_per_seed[k][j] 
                        for k in range(len(res_per_seed))
                        ]).mean(axis=0) 
                    for j in range(len(res_per_seed[0]))
                ]


        #     # get degree vectors
        #     dvs = getCommunityCountsMP(orig_adj_np, pred_labels, num_classes, rng, eps=eps)
        #     dvs_tor = torch.from_numpy(
        #         np.array([dvs[j] for j in dvs])).type(torch.float32)
        #     avg_dvs_nor_per_seed.append(get_avg_nor_given_labels(dvs_tor[test_nodes], pred_labels[test_nodes], labels.cpu().numpy()[test_nodes]))

        # return [
        #         np.stack([
        #                 avg_dvs_nor_per_seed[k][j] 
        #                 for k in range(len(avg_dvs_nor_per_seed))
        #                 ]).mean(axis=0) 
        #             for j in range(len(avg_dvs_nor_per_seed[0]))
        #         ]


def get_lr_nh_hs_dr(dataset=None, arch=None, config_filename=None, nl=0):
    if arch==None:
        return -1, -1, -1, -1
    if 'flickr' in dataset.value:
        return 0.0005, 2, 256, 0.2
    with open(config_filename, "rb") as fp:
        a = pkl.load(fp)
    key1 = dataset
    if 'twitch' in dataset.value:
        key1 = (Dataset.TwitchES, Dataset.TwitchFR)
    key2 = arch
    if 'mmlp' in arch.value:
        key2 = f'MMLP_nl{nl}'    
    t = a[key1][0.0][key2][0].split("-")
    lr, hs, nh, dr = float(t[0]), int(t[1]), int(t[2]), float(t[3])
    return (lr, nh, hs, dr)
        


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=str, help="Folder to parse with all the result files (csv files)", required=True
    )
    parser.add_argument(
        "--outdir", type=str, help="Folder to paste all the tex files", required=True, default='../plots/'
    )
    parser.add_argument(
        "--best_config_file", type=str, help="Pkl file with bets hyperparameters", default=None
    )
    parser.add_argument(
        "--only_per_class", action="store_true", help="True for only computing stats per class", default=False
    )
    parser.add_argument(
        "--dataset",
        type=utils.Dataset,
        choices=utils.Dataset,
        default=utils.Dataset.Cora,
    )

    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        print(f"Creating out directory {args.outdir}")
        os.makedirs(args.outdir)

    # seeds = [129,144,178,235,252,254,276,281,357,37,390,398,468,490,508,583,645,668,715,72,749,767,847,905,907,908,914,925,960,972]
    seeds = [37, 72]
    datasets = [args.dataset]
    # datasets = [Dataset.Cora]
    # epss = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    epss = [0.0]
    nls = [1]
    # archs = [None]
    # archs = [Architecture.MLP]
    # archs =[Architecture.MMLP]
    # archs = [Architecture.TwoLayerGCN]
    archs = [None, Architecture.MLP, Architecture.GCN, Architecture.MMLP]
    from itertools import product
    configs = product(datasets, epss, nls, archs)
    for config in configs:
        dataset = config[0]
        eps = config[1]
        nl = config[2]
        arch = config[3]
        is_inductive = False
        if 'twitch' in dataset.value or 'flickr' in dataset.value:
            is_inductive=True
        lr, nh, hs, dr = get_lr_nh_hs_dr(dataset, arch, args.best_config_file, nl=nl) # Currently hardcoded. Please change
        if (eps>0 and arch == Architecture.MLP) or (nl>1 and arch == Architecture.MLP):
            continue
        if (eps>0 and arch is None) or (nl>1 and arch is None):
            continue
        if nl>1 and arch == Architecture.GCN:
            continue

        print("Computing for: ", dataset, eps, nl, arch)

        if not args.only_per_class:
            avg_dvs_nor = get_avg_dvs_nor_test(dataset, eps, args.results_dir, arch=arch, nl=nl, nh=nh, hs=hs, dropout=dr, is_inductive=is_inductive, seeds=seeds, lr = lr)
            max_avg_dvs_nor = []
            print(max_avg_dvs_nor)
            for i in range(len(avg_dvs_nor)):
                ind = np.where(avg_dvs_nor[i]==np.max(avg_dvs_nor[i]))[0][0]
                print(ind)
                ar = [0.0]*len(avg_dvs_nor[i])
                ar[ind] = avg_dvs_nor[i][ind]
                max_avg_dvs_nor.append(ar)
            print(max_avg_dvs_nor)
            max_avg_dvs_nor=np.stack(max_avg_dvs_nor)
            if arch is None:
                filename = os.path.join(args.outdir, f"dv_avg_norm_1_test_{dataset.name}_orig.png")
            else:
                filename = os.path.join(args.outdir, f"dv_avg_norm_1_test_{dataset.name}_arch_{arch.name}_eps_{eps}.png")
            if arch==Architecture.MMLP:
                filename = os.path.join(args.outdir, f"dv_avg_norm_1_test_{dataset.name}_arch_{arch.name}_nl_{nl}_eps_{eps}.png")
            print(f"Plotting in file {filename}")
            fmt_str = "ind,"+",".join([str(x) for x in range(len(max_avg_dvs_nor))]) 
            np.savetxt(filename.rstrip('.png')+'.csv', np.array([list(range(len(max_avg_dvs_nor)))] + list(max_avg_dvs_nor)).T, header= fmt_str, delimiter=',', comments='')
            np.savetxt(filename.rstrip('.png')+'_avg_.csv', np.array([list(range(len(avg_dvs_nor)))] + list(avg_dvs_nor)).T, header= fmt_str, delimiter=',', comments='')
            plot_avg_norm_dv_1(avg_dvs_nor, filename=filename)

        else:
            avg_per_class_res = get_per_class_stats(dataset, eps, args.results_dir, arch=arch, nl=nl, nh=nh, hs=hs, dropout=dr, is_inductive=is_inductive, seeds=seeds, lr = lr)
            if arch is None:
                filename = os.path.join(args.outdir, f"dv_avg_norm_1_test_{dataset.name}_orig.png")
            else:
                filename = os.path.join(args.outdir, f"dv_avg_norm_1_test_{dataset.name}_arch_{arch.name}_eps_{eps}.png")
            if arch==Architecture.MMLP:
                filename = os.path.join(args.outdir, f"dv_avg_norm_1_test_{dataset.name}_arch_{arch.name}_nl_{nl}_eps_{eps}.png")
            if arch == None:
                fmt_str = "ind,"+"1"
                print(avg_per_class_res)
                np.savetxt(filename.rstrip('.png')+'_nlabels_per_class_.csv', np.array([list(range(len(avg_per_class_res)))] + [list(avg_per_class_res)]).T, header= fmt_str, delimiter=',', comments='')
            else:
                fmt_str = "ind,"+",".join([str(x) for x in range(len(avg_per_class_res))])
                # print(avg_per_class_res)
                print(np.array([list(range(len(avg_per_class_res[0])))] + list(avg_per_class_res)).T)
                np.savetxt(filename.rstrip('.png')+'_stats_per_class_.csv', np.array([list(range(len(avg_per_class_res[0])))] + list(avg_per_class_res)).T, header= fmt_str, delimiter=',', comments='')
        
