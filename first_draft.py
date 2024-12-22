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


file = r"C:\Users\20234383\OneDrive - TU Eindhoven\Desktop\8BA103_Project\Assignment_3\dataset.csv"


def file_preparation(file):
    """
    This function reads the file and extracts the smiles
    """
    data = pd.read_csv(file)

    # Give each SMILES an original number label to check for correct processing
    data['Label'] = pd.RangeIndex(start=1, stop=len(data) + 1, step=1)

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


def get_descriptors(cleaned_data):
    descriptor_data = [] # Initializes a list that will hold all descriptors

    descriptor_names = [n[0] for n in Descriptors._descList] # Finds all possible descriptors and stores these in descriptor_names
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names) # Initializes the calculater with the wanted descriptors

    # Calculate the descriptors based on the SMILES
    for index, row in cleaned_data.iterrows(): # Iterate over the rows of the dataframe
        molecule = row['SMILES_canonical']
        mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
        if mol is not None: # If the SMILES was succesfully converted to an RDKit Molecule
            mol_descriptors = calc.CalcDescriptors(mol) # Gets all descriptors for a molecule
            descriptors_dict = {f"Descriptor_{i}": mol_descriptors[i] for i in range(len(mol_descriptors))} # Create a dictionary with all descriptors
            descriptors_dict['Label'] = row['Label'] # Add label for the molecule to the dictionary
            descriptors_dict['SMILES_canonical'] = row['SMILES_canonical'] # Add the SMILES  of the molecule to the dictionary
            descriptors_dict['target_feature'] = row['target_feature'] # Add the target feature of the molecule to the dictionary
            descriptor_data.append(descriptors_dict) # Append the dictionary for each molecule to a list to be able to create a dataframe further on
     
    # Create a new dataframe including the descriptors
    descriptors = pd.DataFrame(descriptor_data) 

    # Check whether all descriptors were calculated accurately --> what do we want to do?
    empty_descriptors = descriptors.columns[descriptors.isnull().any()] # finds columns with missing values
    descriptors = descriptors.dropna(subset=empty_descriptors) # Removes all molecules with missing values in the descriptors


    return(descriptors)


def normalize_data(descriptors):
    # check which columns should be excluded from normalization
    NOT_normalize_columns = []
    for column in descriptors.columns:
        if descriptors[column].isin([0, 1]).all() or column == 'Label' or column == 'SMILES_canonical': # Binary columns and label columns are excluded
            NOT_normalize_columns.append(column)

    # Select columns that are not binary
    columns_to_normalize = []
    for column in descriptors.columns:
        if column not in NOT_normalize_columns: # If the columns are not part of the not normalize columns they should be normalized
            columns_to_normalize.append(column)

    descriptors[columns_to_normalize] = normalize(descriptors[columns_to_normalize], axis=0, norm='l2')
 
    return descriptors


def neural_network(descriptors):
    X = descriptors.drop(['target_feature', 'Label', 'SMILES_canonical'], axis=1) # Get all the feature data
    y = descriptors['target_feature'] # Get all the target values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # Split the data, random state can be removed for final hand in

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
    nn.fit(X_train, y_train) 
    predictions = nn.predict(X_test)
    test_predictions = nn.predict(X)

    # Print Accuracy of model to be able to check efficacy of the model
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Create a csv file from the predictions to test in Kaggle
    results = pd.DataFrame({'SMILES_canonical': descriptors['SMILES_canonical'], 'prediction': test_predictions})
    results.to_csv("molecule_predictions.csv", index=False)
    print("File saved at:", os.path.abspath("molecule_predictions.csv"))

    return predictions




SMILES = file_preparation(file)
descriptors = get_descriptors(SMILES)
descriptors = normalize_data(descriptors)
neural_network(descriptors)
