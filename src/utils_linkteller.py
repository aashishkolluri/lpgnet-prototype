from collections import defaultdict
import itertools
import json
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from globals import MyGlobals


def twitch_feature_reader(
    dataset="twitch/", scale="large", train_ratio=0.5, feature_size=-1
):
    identifier = dataset[dataset.find("/") + 1 :]
    filename = os.path.join(
        MyGlobals.LK_DATA, "{}/musae_{}_features.json".format(dataset, identifier)
    )
    with open(filename) as f:
        data = json.load(f)
        n_nodes = len(data)

        items = sorted(set(itertools.chain.from_iterable(data.values())))
        n_features = 3170 if dataset.startswith("twitch") else max(items) + 1

        features = np.zeros((n_nodes, n_features))
        for idx, elem in data.items():
            features[int(idx), elem] = 1

    data = pd.read_csv(
        os.path.join(
            MyGlobals.LK_DATA, "{}/musae_{}_target.csv".format(dataset, identifier)
        )
    )
    mature = list(map(int, data["mature"].values))
    new_id = list(map(int, data["new_id"].values))
    idx_map = {elem: i for i, elem in enumerate(new_id)}
    labels = [mature[idx_map[idx]] for idx in range(n_nodes)]

    labels = torch.LongTensor(labels)
    return features, labels


def flickr_feature_reader(
    dataset="flickr", scale="large", train_ratio=0.5, feature_size=-1
):
    # role
    role = json.load(
        open(os.path.join(MyGlobals.LK_DATA, "{}/role.json".format(dataset)))
    )
    idx_train = np.asarray(sorted(role["tr"]))
    idx_valid = np.asarray(sorted(role["va"]))
    idx_test = np.asarray(sorted(role["te"]))

    # features
    features = np.load(os.path.join(MyGlobals.LK_DATA, "{}/feats.npy".format(dataset)))
    features_train = features[idx_train]

    scaler = StandardScaler()
    scaler.fit(features_train)
    features = scaler.transform(features)
    features = torch.FloatTensor(features)
    features_train = features[idx_train]

    n_nodes = len(features)

    # label
    class_map = json.load(
        open(os.path.join(MyGlobals.LK_DATA, "{}/class_map.json".format(dataset)))
    )

    multi_label = 1
    for key, value in class_map.items():
        if type(value) == list:
            multi_label = len(value)  # single-label vs multi-label
        break

    labels = np.zeros((n_nodes, multi_label))
    for key, value in class_map.items():
        labels[int(key)] = value
    labels = torch.LongTensor(labels)
    # labels = torch.from_numpy(labels)

    return features, features_train, labels, idx_train, idx_valid, idx_test


def construct_balanced_edge_sets(dataset, sample_type, adj, n_samples, rng):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    dic = defaultdict(list)
    for u in range(n_nodes):
        begg, endd = indptr[u : u + 2]
        dic[u] = indices[begg:endd]

    edge_set = []
    nonedge_set = []

    # construct edge set
    for u in range(n_nodes):
        for v in dic[u]:
            if v > u:
                edge_set.append((u, v))
    n_samples = len(edge_set)

    # random sample equal number of pairs to compose a nonoedge set
    while 1:
        u = rng.choice(n_nodes)
        v = rng.choice(n_nodes)
        if v not in dic[u] and u not in dic[v]:
            nonedge_set.append((u, v))
            if len(nonedge_set) == n_samples:
                break

    print(
        f"sampling done! len(edge_set) = {len(edge_set)}, len(nonedge_set) = {len(nonedge_set)}"
    )

    return (edge_set, nonedge_set), list(range(n_nodes))


def construct_edge_sets(dataset, sample_type, adj, n_samples, rng):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    # construct edge set
    edge_set = []
    while 1:
        u = rng.choice(n_nodes)
        begg, endd = indptr[u : u + 2]
        v_range = indices[begg:endd]
        if len(v_range):
            v = rng.choice(v_range)
            edge_set.append((u, v))
            if len(edge_set) == n_samples:
                break

    # construct non-edge set
    nonedge_set = []

    # randomly select non-neighbors
    for _ in tqdm(range(n_samples)):
        u = rng.choice(n_nodes)
        begg, endd = indptr[u : u + 2]
        v_range = indices[begg:endd]
        while 1:
            v = rng.choice(n_nodes)
            if v not in v_range:
                nonedge_set.append((u, v))
                break
    list1, list2 = zip(*(edge_set + nonedge_set))
    return (edge_set, nonedge_set), list(set(list1 + list2))


def _get_edge_sets_among_nodes(indices, indptr, nodes):
    # construct edge list for each node
    dic = defaultdict(list)

    for u in nodes:
        begg, endd = indptr[u : u + 2]
        dic[u] = indices[begg:endd]

    n_nodes = len(nodes)
    edge_set = []
    nonedge_set = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            u, v = nodes[i], nodes[j]
            if v in dic[u]:
                edge_set.append((u, v))
            else:
                nonedge_set.append((u, v))

    print("#nodes =", len(nodes))
    print("#edges_set =", len(edge_set))
    print("#nonedge_set =", len(nonedge_set))
    return edge_set, nonedge_set


def _get_degree(n_nodes, indptr):
    deg = np.zeros(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        deg[i] = indptr[i + 1] - indptr[i]

    ind = np.argsort(deg)
    return deg, ind


def construct_edge_sets_from_random_subgraph(dataset, sample_type, adj, n_samples, rng):
    indices = adj.indices
    indptr = adj.indptr
    n_nodes = adj.shape[0]

    if sample_type == "unbalanced":
        indice_all = range(n_nodes)

    else:
        deg, ind = _get_degree(n_nodes, indptr)
        if dataset.startswith("twitch"):
            lo = 5 if "PTBR" not in dataset else 10
            hi = 10
        elif dataset in ("flickr", "ppi"):
            lo = 15
            hi = 30
        elif dataset in ("cora"):
            lo = 3
            hi = 4
        elif dataset in ("citeseer"):
            lo = 3
            hi = 3
        elif dataset in ("pubmed"):
            lo = 10
            hi = 10
        else:
            raise NotImplementedError(f"lo and hi for dataset = {dataset} not set!")

        if sample_type == "unbalanced_lo":
            indice_all = np.where(deg <= lo)[0]
        else:
            indice_all = np.where(deg >= hi)[0]

    print("#indice =", len(indice_all))

    # choose from low degree nodes
    nodes = rng.choice(indice_all, n_samples, replace=False)

    return _get_edge_sets_among_nodes(indices, indptr, nodes), nodes
