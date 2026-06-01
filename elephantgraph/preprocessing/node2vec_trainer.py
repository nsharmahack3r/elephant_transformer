import numpy as np
import pickle
from node2vec import Node2Vec

NODE2VEC_DIM = 64


class Node2VecTrainer:
    def __init__(self, dimensions=NODE2VEC_DIM, walk_length=40, num_walks=200,
                 p=1, q=4, workers=4, window=10):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window = window
        self.model = None

    def fit(self, G):
        node2vec = Node2Vec(
            G,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,
            quiet=True
        )
        self.model = node2vec.fit(window=self.window, min_count=1)
        return self.model

    def get_embeddings(self):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        embeddings = {}
        for node in self.model.wv.index_to_key:
            embeddings[node] = self.model.wv[node]
        return embeddings

    def save(self, path):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        data = {
            'embeddings': {node: self.model.wv[node]
                           for node in self.model.wv.index_to_key}
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['embeddings']
