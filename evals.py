import numpy as np

import json 

import numpy as np
import os
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr


def read_wordpair(file):
    """
    load file whose line template is
    token1 token2 score
    """
    pairs = []
    with open(file, 'r') as f:
        for line in f:
            pair = line.split()
            pair[-1] = float(pair[-1])
            pairs.append(pair)
    return pairs

def get_similarity_score(sim_file, dict_token_idx = None, embeddings = None, token_embedding = None):
    pairs = read_wordpair(sim_file)
    human_sim = []
    vec_sim = []
    cnt = 0
    total = len(pairs)
    
    if token_embedding:
        for pair in pairs:
            w1 = pair[0]
            w2 = pair[1]

            if w1 in token_embedding and w2 in token_embedding:
                cnt += 1
                wv1 = token_embedding[w1]
                wv2 = token_embedding[w2]

                scale = np.linalg.norm(wv1)*np.linalg.norm(wv2)
                vsim = wv1.dot(wv2.T)/scale
                human_sim.append(pair[2])
                vec_sim.append(vsim)
                
    else:
        for pair in pairs:
            w1 = pair[0]
            w2 = pair[1]

            if w1 in dict_token_idx and w2 in dict_token_idx:
                cnt += 1
                id1 = dict_token_idx[w1]
                id2 = dict_token_idx[w2]
                wv1 = embeddings[id1]
                qv2 = embeddings[id2]
                scale = np.linalg.norm(wv1)*np.linalg.norm(wv2)
                vsim = wv1.dot(wv2.T)/scale
                human_sim.append(pair[2])
                vec_sim.append(vsim)
    print(cnt, '/', total, ' word pairs appeared in the training dictionary')
    score = spearmanr(human_sim, vec_sim)
    print(sim_file, ':', score)
    
