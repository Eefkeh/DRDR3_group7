import rdkit
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

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
    for row in data:
        mol = Chem.MolFromSmiles(row)
        dict[row]= [calc.CalcDescriptors(mol)]
    
    return dict

print(calculate_descriptors('/Users/stefaniekip/Documents/BMT jaar 4/Q2 - Advanced programming/Groeps opdracht/drd-3-binder-quest (1)/train.csv'))