from __future__ import division, print_function

import os, sys
import numpy as np

from sklearn.decomposition import NMF, MiniBatchDictionaryLearning, MiniBatchSparsePCA
from sklearn.base import BaseEstimator
from sklearn.grid_search import ParameterGrid

import librosa

import evaluation
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("--n_train", type=int, default=10)
args = parser.parse_args()

SR = 16000

class Reconstructor(BaseEstimator):
    def __init__(self, n_comp=128):
        self.n_comp = n_comp

    def _fit(self, X, y=None):
        stft = librosa.stft(y=X)
        self.stft_abs = np.abs(stft)
        self.stft_ang = np.angle(stft)
        self.decomposer.fit(self.stft_abs.transpose())

    def fit(self, X, y=None):
        self._fit(X)
        
    def fit_transform(self, X, y=None):
        self._fit(X)
        code = self.decomposer.transform(self.stft_abs.transpose())
        stft_recon = np.dot(code, self.decomposer.components_)
        return librosa.istft(stft_recon.transpose() * np.exp(self.stft_ang*1j))
    
    def transform(self, X):
        stft = librosa.stft(y=X)
        code = self.decomposer.transform(np.abs(stft).transpose())
        stft_recon = np.dot(code, self.decomposer.components_)
        return librosa.istft(stft_recon.transpose() * np.exp(self.stft_ang*1j))

    def score(self, X, y=None):
        reconstructed = self.transform(X)
        # TODO: connect to evaluation script
        return evaluation.score(X, reconstructed)

    
class NMFReconstructor(Reconstructor):
    def __init__(self, n_comp=128, alpha=0.0, l1_ratio=0.0):
        self.n_comp = n_comp
        self.decomposer = NMF(n_components=n_comp, alpha=alpha, l1_ratio=l1_ratio)


class SparsePCAReconstructor(Reconstructor):
    def __init__(self, n_comp=128, alpha=1):
        self.n_comp = n_comp
        self.decomposer = MiniBatchSparsePCA(n_components=n_comp, alpha=alpha)
        

class DictionaryLearningReconstructor(Reconstructor):
    def __init__(self, n_comp=128, alpha=1):
        self.n_comp = n_comp
        self.decomposer = MiniBatchDictionaryLearning(n_components=n_comp, alpha=alpha)
        
        
class Experiments:
    def __init__(self, data):
        """data needs to be an iterable of signal examples"""
        self._data = data

    def run(self, reconstructor, param_grid):
        self.grid_scores_ = []
        for params in list(ParameterGrid(param_grid)):
            rec = reconstructor(**params)
            scores = []
            for d in self._data:
                rec.fit(d)
                scores.append(rec.score(d))
            self.grid_scores_.append((params, scores))
        self.best_params_, self.best_scores_ = max(self.grid_scores_, key=lambda x: np.mean(x[1]))

        
def main(args):
    data_dir = args.data_dir
    if os.path.isdir(data_dir):    # data directory
        filenames = os.listdir(data_dir)
    else:    # data file
        filenames = [data_dir]
        data_dir = ""
        
    tr = []
    i = 0

    for fn in filenames:
        if fn.endswith(".wav"):
            filepath = os.path.join(data_dir, fn)
            data, sr = librosa.load(filepath, SR)
            tr.append(data)
            i += 1
            if i == args.n_train:
                break
        
    print("{0} examples loaded!".format(i))

    experiments = Experiments(tr)

    reconstructors = [NMFReconstructor, SparsePCAReconstructor, DictionaryLearningReconstructor]
    parameters = {"n_comp": [8, 16, 32, 64]}

    for rec in reconstructors:
        
        experiments.run(rec, parameters)
        print("Reconstructor: ", rec)
        print("Best params: ", experiments.best_params_)

        
if __name__ == "__main__":
    main(args)
    
