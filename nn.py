import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import json
import tensorflow as tf 
import numpy as np 
from keras import Input
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Activation 
from keras.layers import LeakyReLU 
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC, F1Score, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, PrecisionAtRecall, SensitivityAtSpecificity


def pretty_print_dict(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 2)
        else:
            print(value)


def handle_skew(X):
    skewed_cols = X.columns[X.skew() > 0.5]
    X[skewed_cols] = X[skewed_cols].apply(lambda x: np.log(x + 1))
    return X



def remove_outliers_iforest(X, y, contamination=0.05, random_state=42):
    """
    Remove outliers from the input DataFrame using Isolation Forest.
    
    Args:
        X (pd.DataFrame): Input DataFrame containing the features.
        y (pd.Series): Target labels corresponding to the input features.
        contamination (float): The proportion of outliers in the dataset. Default is 0.05.
        random_state (int): Random seed for reproducibility. Default is 42.
        
    Returns:
        tuple: A tuple containing the cleaned input features (X_clean) and cleaned target labels (y_clean).
    """
    # Create an instance of Isolation Forest
    iforest = IsolationForest(contamination=contamination, random_state=random_state)
    
    # Fit the model and predict outlier labels
    outlier_labels = iforest.fit_predict(X)
    
    # Create a boolean mask for filtering outliers
    mask = outlier_labels != -1
    
    # Filter the input features and target labels to remove outliers
    X_clean = X[mask]
    y_clean = y[mask]
    
    # Print the number of outliers removed
    num_outliers = len(X) - len(X_clean)
    print(f"Number of outliers removed: {num_outliers}")
    
    return X_clean, y_clean


def run_smote(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def impute_missing_values(X_res):
    imputer = SimpleImputer(strategy='mean')
    X_res = pd.DataFrame(imputer.fit_transform(X_res), columns=X_res.columns)
    return X_res


def remap_target_values(y):
    # Assuming "0" represents failed companies and "1" represents alive companies
    y = y.replace({"alive": 0, "failed": 1})
    return y


def build_model_true(X_train, X_test, X_val, y_train, y_test, y_val, input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        #Dropout(0.2),
        Dense(64, activation='relu'),
        #Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[BinaryAccuracy(name='accuracy'),
                           Precision(name='precision'),
                           Recall(name='recall'),
                           AUC(name='auc'),
                           TruePositives(name='true_positives'),
                           TrueNegatives(name='true_negatives'),
                           FalsePositives(name='false_positives'),
                           FalseNegatives(name='false_negatives')
                           ])
    
    #model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
    model.summary()

    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    loss_and_metrics = model.evaluate(X_test, y_test, return_dict=True)
    pretty_print_dict(loss_and_metrics)
    from keras.utils import plot_model

    # Plot the model architecture
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model


def save_model_metrics(model_name, metrics, file_path):
    """
    Save the accuracy and performance metrics of a neural network model to a JSON file.
    
    Args:
        model_name (str): Name of the model.
        metrics (dict): Dictionary containing the accuracy and performance metrics.
        file_path (str): Path to the JSON file to save the metrics.
    """
    # Create a dictionary to store the model metrics
    model_metrics = {
        'model_name': model_name,
        'metrics': metrics
    }
    
    try:
        # Load existing metrics from the file, if available
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, initialize an empty list
        data = []
    
    # Append the new model metrics to the list
    data.append(model_metrics)
    
    # Save the updated metrics to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Metrics for model '{model_name}' saved to '{file_path}'.")

def main():
    #func_med()

    filename = 'Data/Taiwan/taiwan.csv'
    df = pd.read_csv(filename)

    X = df.drop(labels=['Bankrupt?'], axis=1)
    y = 1 - df['Bankrupt?']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X = handle_skew(X)

    X_clean, y_clean = remove_outliers_iforest(X, y, contamination=0.05, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    build_model_true(X_train, X_test, X_val, y_train, y_test, y_val, np.shape(X_train)[1])
    
    pass


if __name__ == '__main__':
    main()


"""
#TODO 
    #TODO TODAY
    ! implement ratio calculations
    ! tidy up the file
    

    #TODO SUNDAY 
    ! implement model stat saving
    ! batching
    ! data validation
    ? cross validation - k fold
    ? SMOTE
    ! any other preprocessing
        ! basic models
        ! NN

    ! create way to implement/test different 
        ! architectures  
        ! hyperparameters
        ! cost functions 
        ! activiation functions
    
    ! feature scaling
    ! Utilize automated model selection techniques like grid search or randomized search to find the optimal hyperparameters for each model.

    ! SHapley Additive exPlanations

    ! random forest visualiser
    ! exploratory data vizualisation
    ! data input
        ! work out the stages needed for that
    ! Work out how to use a pre trained model in some code - tf.keras.Model.save 
    ! packages all models together
    ! create basic UI
    ! documentation can come from well made docstrings 



"""