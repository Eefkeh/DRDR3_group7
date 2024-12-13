import rdkit
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import pandas as pd


rdkit.__version__

def file_preparation(data):
    """
    This function reads the file and extracts the smiles
    """
    with open(data, 'r') as file:
        reader = csv.reader(file)
        next(reader)                            #skip first row (the header)
        smiles = [row[0] for row in reader]
        train_data = smiles[:10]
    return train_data

def calculate_descriptors(data):
    data = file_preparation(data)
    molecules = []
    dict = {}
    desc_list = [n[0] for n in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    outputcsv = 'data.csv'
    scaler = MinMaxScaler()
    for row in data:
        descriptors = []
        mol = Chem.MolFromSmiles(row)
        descriptors.append(calc.CalcDescriptors(mol))
        #normalized_data = scaler.fit_transform(descriptors)
        normalized_data = normalize(descriptors, norm = 'l2')
        dict[row]= normalized_data
    pd.DataFrame.from_dict(dict)
    #with open("data.csv", "w", newline="") as f:
       # w = csv.DictWriter(f, dict.keys())
        #w.writeheader()
        #w.writerow(dict)

    return dict

print(calculate_descriptors('/Users/stefaniekip/Documents/BMT jaar 4/Q2 - Advanced programming/Groeps opdracht/drd-3-binder-quest (1)/train.csv'))