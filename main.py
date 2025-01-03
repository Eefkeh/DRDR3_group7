from DRD3_model import drd3_model
from DRD3_model import model_UMAP
from DRD3_model import umap
from sklearn.neural_network import MLPClassifier
from umap import UMAP

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os

train_file = 'train.csv'
test_file = 'test.csv'

model = MLPClassifier(hidden_layer_sizes=(210, 105, 52, 26, 13, 6, 3, 1), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001,
                            batch_size=100, 
                            learning_rate='constant', 
                            max_iter=500, 
                            random_state=1)

model_umap = MLPClassifier(hidden_layer_sizes=(50, 100, 25), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001,
                            batch_size=100, 
                            learning_rate='constant', 
                            max_iter=500, 
                            random_state=1)

model_umap2 = MLPClassifier(hidden_layer_sizes=(100, 50, 25), 
                        activation='tanh', 
                        solver='adam', 
                        alpha=0.001,
                            batch_size=64, 
                            learning_rate='adaptive', 
                            max_iter=1000, 
                            random_state=1)

descriptors_train, descriptors_test = drd3_model(train_file, test_file, model)

umap_data_train = umap(descriptors_train, 'Train', n_components = 10)
umap_data_test = umap(descriptors_test, 'Test', n_components = 10)

model_UMAP(umap_data_train, umap_data_test, model_umap2, train_file, test_file, descriptors_train, descriptors_test)



