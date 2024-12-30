import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors 
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
from umap import UMAP

rdkit.__version__


#train_file = r"C:\Users\20234383\OneDrive - TU Eindhoven\Desktop\8BA103_Project\Assignment_3\dataset.csv"
#test_file = r"C:\Users\20234383\OneDrive - TU Eindhoven\Desktop\8BA103_Project\Assignment_3\test.csv"
train_file = '/Users/stefaniekip/Documents/BMT jaar 4/Q2 - Advanced programming/Groeps opdracht/drd-3-binder-quest (1)/train.csv'
test_file = '/Users/stefaniekip/Documents/BMT jaar 4/Q2 - Advanced programming/Groeps opdracht/drd-3-binder-quest (1)/test.csv'

class data_preparation:
    def __init__(self, file, train_or_test, train_data = None, test_data = None, descriptors = None):
        self.file = file
        self.train_or_test = train_or_test
        self.train_data = train_data
        self.test_data = test_data
        self.descriptors = descriptors

    def file_preparation(self):
        """
        This function reads the file and extracts the smiles
        """
        data = pd.read_csv(self.file)

        # Clean the dataset
        # Remove duplicate molecules from dataset
        cleaned_data = data.drop_duplicates(subset='SMILES_canonical', keep='first') # Finds the duplicates in smiles and keeps the first instance

        if self.train_or_test == 'Train':
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

    def get_descriptors(self):
        descriptor_data = [] # Initializes a list that will hold all descriptors

        descriptor_names = [n[0] for n in Descriptors._descList] # Finds all possible descriptors and stores these in descriptor_names
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names) # Initializes the calculater with the wanted descriptors

        if self.train_or_test == 'Test':
            data = self.test_data
        else:
            data = self.train_data

        # Calculate the descriptors based on the SMILES
        for index, row in data.iterrows(): # Iterate over the rows of the dataframe
            molecule = row['SMILES_canonical']
            mol = Chem.MolFromSmiles(molecule) # Converts SMILES molecule object to RDKit molecule object
            if mol is not None: # If the SMILES was succesfully converted to an RDKit Molecule
                mol_descriptors = calc.CalcDescriptors(mol) # Gets all descriptors for a molecule
                descriptors_dict = {f"Descriptor_{i}": mol_descriptors[i] for i in range(len(mol_descriptors))} # Create a dictionary with all descriptors
                descriptors_dict['SMILES_canonical'] = row['SMILES_canonical'] # Add the SMILES  of the molecule to the dictionary
                if self.train_or_test == 'Train':
                    descriptors_dict['target_feature'] = row['target_feature'] # Add the target feature of the molecule to the dictionary
                else:
                    descriptors_dict['Unique_ID'] = row['Unique_ID'] # Add label for the molecule to the dictionary
                descriptor_data.append(descriptors_dict) # Append the dictionary for each molecule to a list to be able to create a dataframe further on
     
        # Create a new dataframe including the descriptors
        descriptors = pd.DataFrame(descriptor_data) 

        # Check whether all descriptors were calculated accurately --> what do we want to do?
        empty_descriptors = descriptors.columns[descriptors.isnull().any()] # finds columns with missing values
        descriptors = descriptors.dropna(subset=empty_descriptors) # Removes all molecules with missing values in the descriptors

        return(descriptors)
    
    def normalize_data(self):
        # check which columns should be excluded from normalization
        NOT_normalize_columns = []
        for column in self.descriptors.columns:
            if self.train_or_test == 'Train':
                if self.descriptors[column].isin([0, 1]).all() or column == 'SMILES_canonical': # Binary columns and label columns are excluded
                    NOT_normalize_columns.append(column)
            else:
                if self.descriptors[column].isin([0, 1]).all() or column == 'Unique_ID' or column == 'SMILES_canonical': # Binary columns and label columns are excluded
                    NOT_normalize_columns.append(column)

        # Select columns that are not binary
        columns_to_normalize = []
        for column in self.descriptors.columns:
            if column not in NOT_normalize_columns: # If the columns are not part of the not normalize columns they should be normalized
                columns_to_normalize.append(column)

        self.descriptors[columns_to_normalize] = normalize(self.descriptors[columns_to_normalize], axis=0, norm='l2')
 
        return self.descriptors

class neural_network:
    def __init__(self, descriptors_train, descriptors_test, model, umap_data_test = None, umap_data_train = None):
        self.descriptors_train = descriptors_train
        self.descriptors_test = descriptors_test
        self.model = model
        self.umap_data_test = umap_data_test
        self.umap_data_train = umap_data_train

    def train_model(self):
        X = self.descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1) # Get all the feature data
        y = self.descriptors_train['target_feature'] # Get all the target values

        # Fit the neural network and predict outcomes
        self.model.fit(X,y)

    def train_model_umap(self):
        X = self.umap_data_train.drop(['target_feature'], axis = 1)
        y = self.descriptors_train['target_feature'] # Get all the target values

        self.model.fit(X,y)

    def predict_outcome(self):
        X_test = self.descriptors_test.drop(['Unique_ID', 'SMILES_canonical'], axis=1) # Get all the feature data
        test_predictions = self.model.predict(X_test)

        # Create a csv file from the predictions to test in Kaggle
        results = pd.DataFrame({'Unique_ID': self.descriptors_test['Unique_ID'], 'prediction': test_predictions})
        #results.to_csv("molecule_predictions.csv", index=False)
        #print("File saved at:", os.path.abspath("molecule_predictions.csv"))

    def predict_outcome_umap(self):
        X_test = self.umap_data_test.drop(['Unique_ID'], axis = 1)
        test_predictions = self.model.predict(X_test)

        # Create a csv file from the predictions to test in Kaggle
        results = pd.DataFrame({'Unique_ID': self.descriptors_test['Unique_ID'], 'prediction': test_predictions})
        results.to_csv("molecule_predictions_umap.csv", index=False)
        print("File saved at:", os.path.abspath("molecule_predictions_umap.csv"))

    def accuracy_train_data(self):
        X = self.descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1) # Get all the feature data
        y = self.descriptors_train['target_feature'] # Get all the target values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # Split the data, random state can be removed for final hand in

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, predictions))

def umap(descriptors, n_components = 2):
    # X_train = descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1) # Get all the feature data
    # y_train = descriptors_train['target_feature'] # Get all the target values

    # X_test = descriptors_test.drop(['Unique_ID', 'SMILES_canonical'], axis=1) # Get all the feature data
    # y_test = descriptors_test['target_feature'] # Get all the target values

    # # Classification with a linear SVM
    # svc = LinearSVC(dual=False, random_state=123)
    # params_grid = {"C": [10**k for k in range(-3, 4)]}
    # clf = GridSearchCV(svc, params_grid)
    # clf.fit(X_train, y_train)
    # print("Accuracy on the test set with raw data: {:.3f}".format(clf.score(X_test, y_test)))

    # # Transformation with UMAP followed by classification with a linear SVM
    # umap = UMAP(random_state=456)
    # pipeline = Pipeline([("umap", umap), ("svc", svc)])
    # params_grid_pipeline = {"umap__n_neighbors": [5, 20],"umap__n_components": [15, 25, 50],"svc__C": [10**k for k in range(-3, 4)]}

    # clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline)
    # clf_pipeline.fit(X_train, y_train)
    # print("Accuracy on the test set with UMAP transformation: {:.3f}".format(clf_pipeline.score(X_test, y_test)))
    reducer = UMAP(n_components=n_components)
    train_data = descriptors_train.drop(['target_feature', 'SMILES_canonical'], axis=1)
    scaled_train_data = StandardScaler().fit_transform(train_data)

    embedding = reducer.fit_transform(scaled_train_data)
    print(embedding.shape)
    return embedding


def drd3_model(train_file, test_file, model):
    train_data_prep = data_preparation(train_file, 'Train')
    train_data = train_data_prep.file_preparation()
    test_data_prep = data_preparation(test_file, 'Test')
    test_data = test_data_prep.file_preparation()

    descriptors_train_prep = data_preparation(train_file, 'Train', train_data = train_data, test_data = test_data)
    descriptors_train = descriptors_train_prep.get_descriptors()
    descriptors_test_prep = data_preparation(test_file, 'Test', train_data = train_data, test_data=test_data)
    descriptors_test = descriptors_test_prep.get_descriptors()

    normalized_train_prep = data_preparation(train_file, 'Train', descriptors = descriptors_train)
    normalized_train_data = normalized_train_prep.normalize_data()
    normalized_test_prep = data_preparation(test_file, 'Test', descriptors=descriptors_test)
    normalized_test_data = normalized_test_prep.normalize_data()

    neural_network_prep = neural_network(normalized_train_data, normalized_test_data, model)
    train_model = neural_network_prep.train_model()
    predict_outcome = neural_network_prep.predict_outcome()

    return normalized_train_data, normalized_test_data

model = MLPClassifier(hidden_layer_sizes=(210, 105, 52, 26, 13, 6, 3, 1), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001,
                            batch_size=100, 
                            learning_rate='constant', 
                            max_iter=500, 
                            random_state=1)

descriptors_train, descriptors_test = drd3_model(train_file, test_file, model)

model_umap = MLPClassifier(hidden_layer_sizes=(5, 10, 5), activation='relu', solver='adam', alpha=0.0001,batch_size=100, learning_rate='constant', max_iter=500, random_state=1)

#model_umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

umap_data_train = umap(descriptors_train, n_components = 5)
umap_data_test = umap(descriptors_test, n_components = 5)

print(umap_data_test)

def model_UMAP(data_train, data_test, model_umap, train_file, test_file, descriptors_train , descriptors_test):
    # Convert UMAP data to DataFrame for compatibility with neural network
    umap_train_df = pd.DataFrame(data_train, columns=[f"UMAP_{i}" for i in range(data_train.shape[1])])
    umap_train_df['target_feature'] = descriptors_train['target_feature'].values

    umap_test_df = pd.DataFrame(data_test, columns=[f"UMAP_{i}" for i in range(data_test.shape[1])])

    if descriptors_test.shape[0] != umap_test_df.shape[0]:
        umap_test_df = umap_test_df.iloc[:descriptors_test.shape[0]]
    
    umap_test_df['Unique_ID'] = descriptors_test['Unique_ID'].values  # Add Unique_ID for results

    # Train the model using the UMAP-transformed training data
    neural_network_prep = neural_network(descriptors_train, descriptors_test, model_umap, umap_data_test=umap_test_df, umap_data_train=umap_train_df)        #er staat hier model ipv model_umap?
    neural_network_prep.train_model_umap()

    # Make predictions using the UMAP-transformed test data
    neural_network_prep.predict_outcome_umap()

    accuracy_prep = neural_network(descriptors_train, descriptors_test, model_umap)
    accuracy_prep.accuracy_train_data()

model_UMAP(umap_data_train, umap_data_test, model_umap, train_file, test_file, descriptors_train, descriptors_test)


#drd3_model(train_file, test_file, model)
#drd3_model(umap_data_train, umap_data_test, model)