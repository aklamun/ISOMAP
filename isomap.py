# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 08:44:02 2019

@author: aklamun
"""

import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd

def knn(data,k):
    '''output knn graph, edge weight = euclidean distance
    data = each row is a vector data point'''
    d_mat = sp.spatial.distance.cdist(data,data,metric='euclidean')
    knn_graph = nx.Graph()
    
    for i in range(data.shape[0]):
        inds = np.argsort(d_mat[i,:])[1:k+1]
        for j in inds:
            knn_graph.add_edge(i,j, weight=d_mat[i,j])
    
    #choose the maximal component subgraph (delete isolated points as suggested in paper)
    comps = list(nx.connected_component_subgraphs(knn_graph))
    comp_lens = [c.number_of_nodes() for c in comps]
    i = np.argmax(comp_lens)
    max_comp = comps[i]
    return max_comp

def mds(D,d):
    '''multi-dimensional scaling, D = distance matrix, d = final dimensionality'''
    S = np.multiply(D,D) #squared distances matrix
    n = S.shape[0]
    H = np.eye(n) - (1./n)*np.ones((n,n))
    tau = -0.5* np.dot( H, np.dot(S,H) )
    
    w,v = np.linalg.eigh(tau)
    #reverse order of w,v (order from greatest to least eigval)
    w = np.flip(w,axis=0)
    v = np.flip(v,axis=1)
    
    #select d dimensions
    w = w[:d]
    v = v[:,:d]
    
    #form embedding coordinate matrix
    X = np.dot( np.diag(np.sqrt(w)), np.transpose(v) )
    return X

def isomap(data,k,d):
    '''isomap embedding, k=# neightbords for knn, d=embedding dimensions
    returns X = columns are embedding coordinates of data points that were not removed in process'''
    knn_graph = knn(data,k)
    sp_mat = nx.floyd_warshall_numpy(knn_graph, weight='weight') #shortest path matrix
    X = mds(sp_mat,d)
    return X, sp_mat

def calc_residuals(data, max_d, k):
    '''calculate residuals for isomap vs. MDS as in ISOMAP paper'''
    df_res = pd.DataFrame(index=list(range(1,max_d+1)), columns=['Isomap','MDS'])
    for d in range(1,max_d+1):
        X, D_M = isomap(rets_data, k, d)
        D_M = np.array(D_M)
        D_Y = sp.spatial.distance_matrix(np.transpose(X), np.transpose(X), 2)
        
        D_M2 = np.array(sp.spatial.distance_matrix(data,data,2))
        X2 = mds(D_M2, d)
        D_Y2 = sp.spatial.distance_matrix(np.transpose(X2),np.transpose(X2), 2)
        
        R = np.corrcoef(D_Y.flatten(), D_M.flatten())[0,1]
        R_2 = np.corrcoef(D_Y2.flatten(), D_M2.flatten())[0,1]
        
        df_res.loc[d,'Isomap'] = R**2
        df_res.loc[d,'MDS'] = R_2**2
        
    return df_res

