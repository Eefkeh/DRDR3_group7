import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os

rdkit.__version__


train_file = r"C:\Users\20234383\OneDrive - TU Eindhoven\Desktop\8BA103_Project\Assignment_3\dataset.csv"
test_file = r"C:\Users\20234383\OneDrive - TU Eindhoven\Desktop\8BA103_Project\Assignment_3\test.csv"


def file_preparation_train(file):
    """
    This function reads the file and extracts the smiles
    """
    data = pd.read_csv(file)

    # Clean the dataset
    # Remove duplicate molecules from dataset
    cleaned_data = data.drop_duplicates(subset='SMILES_canonical', keep='first') # Finds the duplicates in smiles and keeps the first instance

    # Make sure that the results are 0 or 1 otherwise remove molecule
    cleaned_data = cleaned_data[cleaned_data['target_feature'].isin([0, 1])]

    # Check for invalid smiles
    invalid_smiles = []

    for index, row in cleaned_data.iterrows(): # Iterate over the rows of the dataframe
        molecule = row['SMILES_canonical']
        mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
        if mol is None: # If the SMILES cannot be converted to an RDKit Molecule append to invalid_smiles
            invalid_smiles.append(row['SMILES_canonical']) 
  
    cleaned_data = cleaned_data.loc[~cleaned_data['SMILES_canonical'].isin(invalid_smiles)] # Take out all molecules with invalid smiles


    return cleaned_data

def file_preparation_test(file):
    """
    This function reads the file and extracts the smiles
    """
    data = pd.read_csv(file)

    # Clean the dataset
    # Remove duplicate molecules from dataset
    cleaned_data = data.drop_duplicates(subset='SMILES_canonical', keep='first') # Finds the duplicates in smiles and keeps the first instance

    # Check for invalid smiles
    invalid_smiles = []

    for index, row in cleaned_data.iterrows(): # Iterate over the rows of the dataframe
        molecule = row['SMILES_canonical']
        mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
        if mol is None: # If the SMILES cannot be converted to an RDKit Molecule append to invalid_smiles
            invalid_smiles.append(row['SMILES_canonical']) 
  
    cleaned_data = cleaned_data.loc[~cleaned_data['SMILES_canonical'].isin(invalid_smiles)] # Take out all molecules with invalid smiles

    return cleaned_data

def get_descriptors_train(train_data):
    descriptor_data = [] # Initializes a list that will hold all descriptors

    descriptor_names = [n[0] for n in Descriptors._descList] # Finds all possible descriptors and stores these in descriptor_names
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names) # Initializes the calculater with the wanted descriptors

    # Calculate the descriptors based on the SMILES
    for index, row in train_data.iterrows(): # Iterate over the rows of the dataframe
        molecule = row['SMILES_canonical']
        mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
        if mol is not None: # If the SMILES was succesfully converted to an RDKit Molecule
            mol_descriptors = calc.CalcDescriptors(mol) # Gets all descriptors for a molecule
            descriptors_dict = {f"Descriptor_{i}": mol_descriptors[i] for i in range(len(mol_descriptors))} # Create a dictionary with all descriptors
            descriptors_dict['SMILES_canonical'] = row['SMILES_canonical'] # Add the SMILES  of the molecule to the dictionary
            descriptors_dict['target_feature'] = row['target_feature'] # Add the target feature of the molecule to the dictionary
            descriptor_data.append(descriptors_dict) # Append the dictionary for each molecule to a list to be able to create a dataframe further on
     
    # Create a new dataframe including the descriptors
    descriptors_train = pd.DataFrame(descriptor_data) 

    # Check whether all descriptors were calculated accurately --> what do we want to do?
    empty_descriptors = descriptors_train.columns[descriptors_train.isnull().any()] # finds columns with missing values
    descriptors_train = descriptors_train.dropna(subset=empty_descriptors) # Removes all molecules with missing values in the descriptors


    return(descriptors_train)


def normalize_data_train(descriptors_train):
    # check which columns should be excluded from normalization
    NOT_normalize_columns = []
    for column in descriptors_train.columns:
        if descriptors_train[column].isin([0, 1]).all() or column == 'SMILES_canonical': # Binary columns and label columns are excluded
            NOT_normalize_columns.append(column)

    # Select columns that are not binary
    columns_to_normalize = []
    for column in descriptors_train.columns:
        if column not in NOT_normalize_columns: # If the columns are not part of the not normalize columns they should be normalized
            columns_to_normalize.append(column)

    descriptors_train[columns_to_normalize] = normalize(descriptors_train[columns_to_normalize], axis=0, norm='l2')
 
    return descriptors_train


def get_descriptors_test(test_data):
    descriptor_data = [] # Initializes a list that will hold all descriptors

    descriptor_names = [n[0] for n in Descriptors._descList] # Finds all possible descriptors and stores these in descriptor_names
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names) # Initializes the calculater with the wanted descriptors

    # Calculate the descriptors based on the SMILES
    for index, row in test_data.iterrows(): # Iterate over the rows of the dataframe
        molecule = row['SMILES_canonical']
        mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
        if mol is not None: # If the SMILES was succesfully converted to an RDKit Molecule
            mol_descriptors = calc.CalcDescriptors(mol) # Gets all descriptors for a molecule
            descriptors_dict = {f"Descriptor_{i}": mol_descriptors[i] for i in range(len(mol_descriptors))} # Create a dictionary with all descriptors
            descriptors_dict['Unique_ID'] = row['Unique_ID'] # Add label for the molecule to the dictionary
            descriptors_dict['SMILES_canonical'] = row['SMILES_canonical'] # Add the SMILES  of the molecule to the dictionary
            descriptor_data.append(descriptors_dict) # Append the dictionary for each molecule to a list to be able to create a dataframe further on
     
    # Create a new dataframe including the descriptors
    test_descriptors = pd.DataFrame(descriptor_data) 

    # Check whether all descriptors were calculated accurately --> what do we want to do?
    empty_descriptors = test_descriptors.columns[test_descriptors.isnull().any()] # finds columns with missing values
    test_descriptors = test_descriptors.dropna(subset=empty_descriptors) # Removes all molecules with missing values in the descriptors

    print(test_descriptors)
    return(test_descriptors)

def normalize_data_test(test_descriptors):
    NOT_normalize_columns = []
    for column in test_descriptors.columns:
        if test_descriptors[column].isin([0, 1]).all() or column == 'Unique_ID' or column == 'SMILES_canonical': # Binary columns and label columns are excluded
            NOT_normalize_columns.append(column)

    # Select columns that are not binary
    columns_to_normalize = []
    for column in test_descriptors.columns:
        if column not in NOT_normalize_columns: # If the columns are not part of the not normalize columns they should be normalized
            columns_to_normalize.append(column)

    test_descriptors[columns_to_normalize] = normalize(test_descriptors[columns_to_normalize], axis=0, norm='l2')
 
    return test_descriptors

def train_network(descriptors_train, test_descriptors):
    X = descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1) # Get all the feature data
    y = descriptors_train['target_feature'] # Get all the target values

    # Initialize a neural network with wanted parameters
    nn = MLPClassifier(hidden_layer_sizes=(210, 105, 52, 26, 13, 6, 3, 1), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001,
                            batch_size=100, 
                            learning_rate='constant', 
                            max_iter=500, 
                            random_state=1)
        
    # Fit the neural network and predict outcomes
    nn.fit(X, y) 

    X_test = test_descriptors.drop(['Unique_ID', 'SMILES_canonical'], axis=1) # Get all the feature data
    test_predictions = nn.predict(X_test)


    # Create a csv file from the predictions to test in Kaggle
    results = pd.DataFrame({'Unique_ID': test_descriptors['Unique_ID'], 'prediction': test_predictions})
    results.to_csv("molecule_predictions.csv", index=False)
    print("File saved at:", os.path.abspath("molecule_predictions.csv"))




train_data = file_preparation_train(train_file)
train_descriptors = get_descriptors_train(train_data)
train_descriptors = normalize_data_train(train_descriptors)

test_data = file_preparation_test(test_file)
test_descriptors = get_descriptors_test(test_data)
test_descriptors = normalize_data_test(test_descriptors)

train_network(train_descriptors, test_descriptors)