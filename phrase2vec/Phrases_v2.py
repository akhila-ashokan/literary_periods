from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import pandas as pd 
import logging, time, sys, argparse, re, gensim, math, faulthandler
import pickle as pkl
import numpy as np
from os.path import exists
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt

"""
New Phrase2Vec Code 
Update: 6/22/2022
"""

def train_phrases_model():
    start_time = time.time()
    context_windows = read_file('../data/full_corpus/context_windows.pickle')
    context_windows_by_sense = sum(context_windows.values(), [])
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(context_windows_by_sense)]
    model = Doc2Vec(tagged_data, vector_size = 20, window = 4, min_count = 1, epochs = 100)
    model.save('../data/full_corpus/phrase_embedding_model.pickle')
    end_time = time.time()
    logging.info(end_time - start_time) 

def read_file(input_path):
        logging.info('Opened: ' + input_path)
        with open(input_path, "rb") as f:
            input_obj = pkl.load(f)
            return input_obj     
        
def get_word_vectors(model, phrases_df):
        vectors = []
        dv = model.dv
        not_found = 0
        for idx, row in phrases_df.iterrows():
            try:
                vectors.append(dv[idx])
            except KeyError:
                logging.info(key, 'not found')
                not_found += 1
                continue

        n = len(vectors)
        logging.info("n: " + str(n))
        logging.info("not_found: " + str(not_found))

        return vectors       
    
def get_similarity_matrix(vectors):
    n = len(vectors)
    distances = np.zeros((len(vectors[:n]), len(vectors[:n])))

    for idx1, vec1 in enumerate(tqdm(vectors[:n])):
        for idx2, vec2 in enumerate(vectors[idx1:n]):
            if idx2 == 0:
                distances[idx1][idx1] = 0
                continue
            # calculate the pearson correlation 
            p = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            # calculate the distance between vectors 
            dist = abs(0.5 * (1 - p))

            distances[idx1][idx1 + idx2] = dist
            distances[idx1 + idx2][idx1] = dist

    return distances

def run_pca(self, distances, k):
    dist_matrix = pd.DataFrame(distances)
    x = dist_matrix.loc[:, ].values
    x = StandardScaler().fit_transform(x)
    reduction_model = PCA(n_components=k)
    pca = reduction_model.fit_transform(x)
    total_var = reduction_model.explained_variance_ratio_.sum() * 100
    logging.info("Total variance explained:" + str(total_var))
    return pca

model = Doc2Vec.load('../data/full_corpus/phrase_embedding_model.pickle')
context_windows = read_file('../data/full_corpus/context_windows.pickle')

phrases_df = pd.DataFrame(columns=['modality', 'descriptor'])
for modality, lst in context_windows.items():
    descriptor_list = [[modality, phrase] for phrase in lst]
    phrases_df = phrases_df.append(pd.DataFrame(descriptor_list, columns = ['modality', 'descriptor']))
    phrases_df = phrases_df.reset_index(drop=True)

phrase_vectors = get_word_vectors(model, phrases_df)
get_similarity_matrix(phrase_vectors)