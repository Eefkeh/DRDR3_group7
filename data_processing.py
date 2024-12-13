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
    descriptor_dict = {} # Initializes a dictionary that will hold each molecule with its descriptors
    desc_list = [n[0] for n in Descriptors._descList] # Finds all possible descriptors and stores these in desc_list
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list) # Initializes the calculater with the wanted descriptors

    for row in data:
        descriptor_values = [] # list to hold all the descriptor values
        mol = Chem.MolFromSmiles(row) # Converts SMILES molecule object to RDKit molecule object
        mol_descriptors =calc.CalcDescriptors(mol) # Gets all descriptors for a molecule
        descriptor_values.append(mol_descriptors) # append the descriptors to the descriptors list


print(calculate_descriptors('/Users/stefaniekip/Documents/BMT jaar 4/Q2 - Advanced programming/Groeps opdracht/drd-3-binder-quest (1)/train.csv'))

data = file_preparation(data)
