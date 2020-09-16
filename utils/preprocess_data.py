import os
import pickle as pkl
import sys

from sklearn.utils import shuffle

import networkx as nx
import numpy as np
import scipy.sparse as sp


def parse_index_file(filename):

    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_citeseer():

    all_data = []
    all_edges = []

    for root,dirs,files in os.walk('./citeseer'):
            for file in files:
                if '.content' in file:
                    with open(os.path.join(root,file),'r') as f:
                        all_data.extend(f.read().splitlines())
                

                    
    #Shuffle the data because the raw data is ordered based on the label
    random_state = 77
    all_data = shuffle(all_data,random_state=random_state)


    #parse the data
    labels_citeseer = []
    nodes = []
    X = []

    for i,data in enumerate(all_data):
        print(data)
        elements = data.split('\t')
        labels_citeseer.append(elements[-1])
        X.append(elements[1:-1])
        nodes.append(elements[0])

    X = np.array(X,dtype=int)
    N = X.shape[0] #the number of nodes
    F = X.shape[1] #the size of node features
    # print('X shape: ', X.shape)


    dataset_str = 'citeseer'
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
      with open("{}ind.{}.{}".format('./citeseer/', dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format('./citeseer/', dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return N, labels, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()



def limit_data(labels,limit=20,val_num=500,test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1
        
        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break
    
    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    return train_idx, val_idx,test_idx

    # print('X shape: ', X.shape)
def load_data_cora():

    #loading the data
    all_data = []
    all_edges = []

    for root,dirs,files in os.walk('./cora'):
        for file in files:
            if '.content' in file:
                with open(os.path.join(root,file),'r') as f:
                    all_data.extend(f.read().splitlines())
            elif 'cites' in file:
                with open(os.path.join(root,file),'r') as f:
                    all_edges.extend(f.read().splitlines())

                
    #Shuffle the data because the raw data is ordered based on the label
    random_state = 77
    all_data = shuffle(all_data,random_state=random_state)


    #parse the data
    labels = []
    nodes = []
    X = []

    for i,data in enumerate(all_data):
        elements = data.split('\t')
        labels.append(elements[-1])
        X.append(elements[1:-1])
        nodes.append(elements[0])

    X = np.array(X,dtype=int)
    N = X.shape[0] #the number of nodes
    F = X.shape[1] #the size of node features


    #parse the edge
    edge_list=[]
    for edge in all_edges:
        e = edge.split('\t')
        edge_list.append((e[0],e[1]))


    train_idx,val_idx,test_idx = limit_data(labels)

    #set the mask
    train_mask = np.zeros((N,),dtype=bool)
    train_mask[train_idx] = True

    val_mask = np.zeros((N,),dtype=bool)
    val_mask[val_idx] = True

    test_mask = np.zeros((N,),dtype=bool)
    test_mask[test_idx] = True

    #build the graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    #obtain the adjacency matrix (A)
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = A + np.eye(A.shape[0])
    # print('Graph info: ', nx.info(G))

    return N, labels, A, X, train_mask, val_mask, test_mask
