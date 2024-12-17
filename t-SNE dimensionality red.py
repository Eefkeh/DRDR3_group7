
#libraries for visualiation
import matplotlib as plt
import pandas as pd
from data_processing import file_preparation,get_descriptors #loads the SMILES data + computes molecular descriptors
from sklearn.manifold import TSNE

def tsne(descriptors):
    tsne_model=TSNE(n_components=2,perplexity=30,init='pca',random_state=None)
    tsne_results=tsne_model.fit_transform(descriptors)

    #visualization
    plt.figure(figsize=(10,7))
    plt.scatter(tsne_results[:0],tsne_results[:1])
    plt.title('t-SNE visualization with PCA initialization')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.show()

    return tsne_results 


if __name__=='__main__':
    file='dataset.csv'

    #loading smiles data
    print('loading smiles data')
    smiles_data=file_preparation(file)
    