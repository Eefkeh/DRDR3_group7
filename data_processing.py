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

file = "dataset.csv"

def file_preparation(file):
    """
    This function reads the file and extracts the smiles
    """
    with open(file, 'r') as file:
        data = pd.read_csv(file, skiprows=1)

        smiles = data[data.columns[0]]
        train_data = smiles[:10]


    return train_data


def get_descriptors(data):
    descriptor_dict = {} # Initializes a dictionary that will hold each molecule with its descriptors
    desc_list = [n[0] for n in Descriptors._descList] # Finds all possible descriptors and stores these in desc_list
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list) # Initializes the calculater with the wanted descriptors

    for row in data:
        mol = Chem.MolFromSmiles(row) # Converts SMILES molecule object to RDKit molecule object
        mol_descriptors = calc.CalcDescriptors(mol) # Gets all descriptors for a molecule
        descriptors_normalized = normalize([mol_descriptors], norm='l2')
        descriptor_dict[mol] = descriptors_normalized
    
    pd.DataFrame.from_dict([descriptor_dict])
    print(descriptor_data)

data = file_preparation(file)

get_descriptors(data)
