import scipy
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def read_dataset(d_name):
    dataset = DglGraphPropPredDataset(name=d_name)
    # split_idx = dataset.get_idx_split()
    # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    # valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    # test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    return dataset


def show_num_edges(d_name, num_bins=200, stat='density'):
    dataset = read_dataset(d_name)
    edge_distrib = []
    for item in dataset.graphs:
        # print("------")
        # print(item.num_nodes())
        # print(dir(item))
        edge_distrib.append(item.num_edges())
    edge_distrib.sort()
    sns.histplot(edge_distrib, kde=False, stat=stat, label='Number of Edges', bins=num_bins)

    # sns.histplot(data=edge_distrib)
    # sns.displot(edge_distrib, color="red")
    plt.title('Distribution of Number of Edges')
    plt.xlabel('Number of Edges')
    # plt.ylabel('Frequency')
    # plt.show()
    try:
        plt.savefig('./datasets/' + d_name + '/num_edges_' + stat + '_' + d_name + '.png')
    except FileNotFoundError:
        os.makedirs('./datasets/' + d_name + '/')
        plt.savefig('./datasets/' + d_name + '/num_edges_' + stat + '_' + d_name + '.png')

    # print(np.quantile(edge_distrib, 0.0))
    # print(np.quantile(edge_distrib, 0.25))
    # print(np.quantile(edge_distrib, 0.50))
    # print(np.quantile(edge_distrib, 0.75))
    # print(np.quantile(edge_distrib, 1.0))
    plt.clf()
    return edge_distrib


def show_num_nodes(d_name, num_bins=200, stat='density'):
    dataset = read_dataset(d_name)
    node_distrib = []
    for item in dataset.graphs:
        # print("------")
        # print(item.num_nodes())
        # print(dir(item))
        node_distrib.append(item.num_nodes())
    # print(read_dataset("ogbg-molhiv").graphs[0].num_nodes)
    sns.histplot(node_distrib, kde=False, stat=stat, label='Number of Nodes', bins=num_bins)
    plt.title('Distribution of Number of Nodes')
    plt.xlabel('Number of Nodes')
    # plt.show()
    try:
        plt.savefig('./datasets/' + d_name + '/num_nodes_' + stat + '_' + d_name + '.png')
    except FileNotFoundError:
        os.makedirs('./datasets/' + d_name + '/')
        plt.savefig('./datasets/' + d_name + '/num_nodes_' + stat + '_' + d_name + '.png')
    plt.clf()
    return node_distrib


def save_quantile(data_dict, name):
    df = pd.DataFrame(data_dict)
    ax = sns.boxplot(data=df, orient="v", whis=[5,95])
    plt.title("Distribution of " + name)
    try:
        plt.savefig('./datasets/' + name + '_quantile.png')
    except FileNotFoundError:
        os.makedirs('datasets/')
        plt.savefig('./datasets/' + name + '_quantile.png')
    # plt.show()
    plt.clf()
    return df.describe()


if __name__ == "__main__":
    dataset = read_dataset("ogbg-molhiv")
    for item in dataset.graphs:
        print("------")
        # print(item.num_nodes())
        # print(dir(item))
        print(item.edges())
        # print(item.ntypes)


    # raw_data = ["ogbg-molhiv"]
    # edge_dict = {}
    # node_dict = {}
    # for data in raw_data:
    #     num_edges = show_num_edges(data, stat='count')
    #     num_nodes = show_num_nodes(data, stat='count')
    #     show_num_edges(data, stat='density')
    #     show_num_nodes(data, stat='density')
    #     edge_dict[data] = num_edges
    #     node_dict[data] = num_nodes
    # save_quantile(edge_dict, "num_edges").to_csv("./datasets/num_edges_describe.csv")
    # save_quantile(node_dict, "num_nodes").to_csv("./datasets/num_edges_describe.csv")



# sns.scatterplot(x="total_bill", y="tip")
