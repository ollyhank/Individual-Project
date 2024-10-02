import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import warnings
import csv

from scipy import stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, fbeta_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LeakyReLU, PReLU
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from keras.utils import plot_model
from keras.callbacks import LambdaCallback

warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


filename = "/Model_metrics/yes_smote_yes_feature"

def pretty_print_dict(d, indent=0):
    """
    Recursively pretty-prints a nested dictionary with indentation.

    Args:
        d (dict): The dictionary to be pretty-printed.
        indent (int, optional): The number of spaces to indent each level of nesting.
            Default is 0.

    Returns:
        None. The function prints the formatted dictionary to the console.
    """
    for key, value in d.items():
        print(' ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            pretty_print_dict(value, indent + 2)
        else:
            print(value)


def remove_outliers_iforest(X, y=None, contamination=0.05, n_estimators=50):
    """
    Remove outliers from the input DataFrame using Isolation Forest.

    Args:
        X (pd.DataFrame): Input DataFrame containing the features.
        y (pd.Series, optional): Target labels corresponding to the input features. Default is None.
        contamination (float): The proportion of outliers in the dataset. Default is 0.05.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: Cleaned input features (X_clean) with outliers removed.
        pd.Series (optional): Cleaned target labels (y_clean) if y is provided, otherwise None.
    """
    # Create an instance of Isolation Forest
    iforest = IsolationForest(contamination=contamination, random_state=42, n_estimators=n_estimators)
    outlier_labels = iforest.fit_predict(X)

    # Create a boolean mask for filtering outliers and filter the outliers
    mask = outlier_labels != -1
    X_clean = X[mask]

    num_outliers = len(X) - len(X_clean)
    logger.info(f"Number of outliers removed: {num_outliers}...")

    if y is not None:
        # Filter the target labels to remove outliers
        y_clean = y[mask]
        return X_clean, y_clean
    else:
        return X_clean


def handle_skew_new(X):
    """
    Handle skewness in the input features.
    
    Args:
        X (pandas.DataFrame): The input dataframe containing the features.
    
    Returns:
        pandas.DataFrame: The transformed dataframe with skewness handled.
    """
    cap_value = 1e10
    skewed_cols = X.columns[X.skew() > 0.5]    

    for column in skewed_cols:
        X[column] = np.log(X[column] + 1)
        X[column] = np.clip(X[column], -cap_value, cap_value)
    
    return X


def select_features(X, y, k=10):
    """
    Select the top k most relevant features from a dataset using the specified Anova-F scoring method.
    
    Parameters:
    X (numpy.ndarray or pandas.DataFrame): The input features.
    y (numpy.ndarray or pandas.Series): The target variable.
    k (int): The number of top features to select.
    
    Returns:
    numpy.ndarray: The selected features.
    """
    constant_filter = VarianceThreshold(threshold=0)
    X_filter = constant_filter.fit_transform(X)

    selector = SelectKBest(score_func=f_classif, k=k)

    selector.fit(X_filter, y)
    selected_mask = selector.get_support()
    selected_mask_indices = selector.get_support(indices=True)

    X_selected = X_filter[:, selected_mask]

    column_names = [X.columns[i] for i in selected_mask_indices]
    column_names = pd.DataFrame(column_names)
    column_names.to_csv("Data/selected_features.csv", header=False, index=False)
    logger.info("Written selection...")

    return X_selected


def run_smote(X, y):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

    Args:
        X (pandas.DataFrame): The input features.
        y (pandas.Series): The target variable.

    Returns:
        tuple: A tuple containing the resampled features (X_res) and target variable (y_res).
    """
    smote = SMOTEENN(sampling_strategy='minority', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Perform random undersampling
    undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    X_res, y_res = undersampler.fit_resample(X_res, y_res)
    
    return X_res, y_res


def impute_test_data(train_data, test_data):
    """
    Impute missing values in a test dataset based on the corresponding values from a training dataset.

    Args:
        train_data (pandas.DataFrame): Training data.
        test_data (pandas.DataFrame): Test Data.

    Returns:
        test_data (pandas.DataFrame): Test data with the imputed values present.
    """
    # Identify the columns with missing values in the test dataset
    missing_cols = test_data.columns[(test_data.isnull() | test_data.eq('')).any()].tolist()    
    # Create an imputer object
    imputer = SimpleImputer(strategy='mean')
    
    # Fit the imputer on the training dataset
    imputer.fit(train_data[missing_cols])
    
    # Transform the test dataset to impute missing values
    test_data[missing_cols] = imputer.transform(test_data[missing_cols])

    return test_data


def run_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Generates a logistic regression model and trains on given data.
    
    Args:
        X_train (pd.DataFrame): Training input features.
        X_test (pd.DataFrame): Testing input features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.

    Returns:
        lr (Sklearn model object): Trained model
    """

    lr = LogisticRegression(random_state=42, 
                            C=10, 
                            solver="liblinear")

    lr.fit(X_train, y_train)

    # Evaluate model
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    """# Save evaluation results to a file
    with open(f"Data/{filename}_results.txt", "a") as file:
        file.write("\n\n\nLogistic Regression Evaluation Results:\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"F2 Score: {f2:.4f}\n")
        file.write(f"ROC AUC Score: {roc_auc:.4f}\n")
        file.write("\nConfusion Matrix:\n")
        file.write(np.array2string(confusion_mat, separator=', '))
        file.write("\n\nClassification Report:\n")
        file.write(classification_report(y_test, y_pred))"""

    return lr


def run_random_forest(X_train, X_test, y_train, y_test):
    """
    Generates a random forest model and trains on given data.
    
    Args:
        X_train (pd.DataFrame): Training input features.
        X_test (pd.DataFrame): Testing input features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.

    Returns:
        rf (Sklearn model object): Trained model
    """

    # Random Forest model with grid search
    rf = RandomForestClassifier(random_state=42, 
                                class_weight=None, 
                                min_samples_split=2, 
                                n_estimators=100)
    
    rf = rf.fit(X_train, y_train)
        
    # Evaluate model
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    """# Save evaluation results to a file
    with open(f"Data/{filename}_results.txt", "a") as file:
        file.write("\n\n\nRandom Forest Evaluation Results:\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"F2 Score: {f2:.4f}\n")
        file.write(f"ROC AUC Score: {roc_auc:.4f}\n")
        file.write("\nConfusion Matrix:\n")
        file.write(np.array2string(confusion_mat, separator=', '))
        file.write("\n\nClassification Report:\n")
        file.write(classification_report(y_test, y_pred))"""

    return rf


def run_svm(X_train, X_test, y_train, y_test):
    """
    Generates a support vector machine model and trains on given data.
    
    Args:
        X_train (pd.DataFrame): Training input features.
        X_test (pd.DataFrame): Testing input features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.

    Returns:
        svm (Sklearn model object): Trained model
    """
    # SVM model with grid search
    svm = SVC(probability=True, 
                random_state=42, 
                C=10, 
                kernel="rbf", 
                gamma="auto")
    
    svm.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = svm.predict(X_test)
    y_pred_proba = svm.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    """# Save evaluation results to a file
    with open(f"Data/{filename}_results.txt", "a") as file:
        file.write("\n\n\nSVM Evaluation Results:\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"F2 Score: {f2:.4f}\n")
        file.write(f"ROC AUC Score: {roc_auc:.4f}\n")
        file.write("\nConfusion Matrix:\n")
        file.write(np.array2string(confusion_mat, separator=', '))
        file.write("\n\nClassification Report:\n")
        file.write(classification_report(y_test, y_pred))
    """
    return svm


def run_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Generates a gradient boosting model and trains on given data.
    
    Args:
        X_train (pd.DataFrame): Training input features.
        X_test (pd.DataFrame): Testing input features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.

    Returns:
        gb (Sklearn model object): Trained model
    """
    
    # Gradient Boosting model with grid search
    gb = GradientBoostingClassifier(random_state=42, 
                                    learning_rate=0.1, 
                                    max_depth=7, 
                                    max_features="log2", 
                                    n_estimators=300, 
                                    subsample=0.8)
    
    gb.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = gb.predict(X_test)
    y_pred_proba = gb.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    """# Save evaluation results to a file
    with open(f"Data/{filename}_results.txt", "a") as file:
        file.write("\n\n\nGradient Boosting Evaluation Results:\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"F2 Score: {f2:.4f}\n")
        file.write(f"ROC AUC Score: {roc_auc:.4f}\n")
        file.write("\nConfusion Matrix:\n")
        file.write(np.array2string(confusion_mat, separator=', '))
        file.write("\n\nClassification Report:\n")
        file.write(classification_report(y_test, y_pred))"""

    return gb


def run_models(X_train, X_test, y_train, y_test, models):
    """
    Runs all traditional statistical models from Sklearn library
    
    Args:
        X_train (pd.DataFrame): Training input features.
        X_test (pd.DataFrame): Testing input features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.
        models (Dict): Dictionary of model names and resulty.
        
    Returns:
        models (Dict): Dictionary of model names and resulty.
    """
    logger.info("Training Logistic Regression...")
    lr = run_logistic_regression(X_train, X_test, y_train, y_test)
    models["Logistic Regression"] = lr

    logger.info("Training Random Forest...")
    rf = run_random_forest(X_train, X_test, y_train, y_test)
    models["Random Forest"] = rf

    logger.info("Training Support Vector Machine...")
    svm = run_svm(X_train, X_test, y_train, y_test)
    models["Support Vector Machine"] = svm

    logger.info("Training Gradient Boosting...")
    gb = run_gradient_boosting(X_train, X_test, y_train, y_test)
    models["Gradient Boosting"] = gb

    return models


def calculate_financial_ratios(df):
    """
    Calculate financial ratios from the input dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the required variables.
        
    Returns:
        pd.DataFrame: A new dataframe with the calculated financial ratios.
    """
    try:

        df.columns = ['_'.join(col.split()).strip() for col in df.columns]

        calculated_ratios = pd.DataFrame()
        # Calculate BV/TA - Book value / Total assets
        calculated_ratios['BV/TA'] = df['Net_Value_Per_Share_(A)'] / df['Total_Asset_Turnover']
        
        # Calculate CF/TA - Cashflow / Total assets
        calculated_ratios['CF/TA'] = df['Cash_Flow_to_Total_Assets']
        
        # Calculate GOI/TA - Gross operating income / Total assets
        calculated_ratios['GOI/TA'] = df['Operating_Profit_Per_Share_(Yuan_¥)'] / df['Total_Asset_Turnover']
        
        # Calculate P/CF - Price / Cashflow
        calculated_ratios['P/CF'] = df['Net_Value_Per_Share_(A)'] / df['Cash_Flow_Per_Share']
        
        return calculated_ratios
    
    except KeyError as e:
        logger.error(f"Required variable not found in the dataframe: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Error during financial ratio calculation: {str(e)}")
        raise


def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing the loaded DataFrame and the target variable.
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = ['_'.join(col.split()).strip() for col in df.columns]
        logger.info("Calculating financial ratios...")

        X = df.drop(labels=['Bankrupt?'], axis=1)
        y = 1 - df['Bankrupt?']

        return X, y
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except KeyError as e:
        logger.error(f"KeyError: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


def preprocess_data(X, y):
    """
    Preprocess the input data.

    Args:
        X (pd.DataFrame): Input features.
        y (pd.Series): Target variable.

    Returns:
        tuple: A tuple containing the preprocessed features and target variable.
    """
    try:
        logger.info("Handling skewed features...")
        X = handle_skew_new(X)

        logger.info("Removing outliers...")
        X_clean, y_clean = remove_outliers_iforest(X, y, contamination=0.05)

        X_expanded = calculate_financial_ratios(X_clean)

        logger.info("Selecting best features...")
        X_selected = select_features(X_clean, y_clean)

        X_selected = np.concatenate((X_selected, X_expanded), axis=1)

        logger.info("Applying SMOTE to handle class imbalance...")
        X_resampled, y_resampled = run_smote(X_selected, y_clean)

        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise


def build_model(X_train, X_test, y_train, y_test, input_dim):
    """
    Build and train a neural network model.

    Args:
        X_train (pd.DataFrame): Training input features.
        X_test (pd.DataFrame): Testing input features.
        y_train (pd.Series): Training target labels.
        y_test (pd.Series): Testing target labels.
        input_dim (int): Input dimension of the model.
        filename (str): File to save the evaluation results.

    Returns:
        keras.Model: Trained neural network model.
    """
    try:
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation=PReLU(), name='Input'),
            Dense(256, activation=LeakyReLU(alpha=0.05), name='Hidden1'),
            Dropout(0.2, name='Dropout1'),
            Dense(128, activation=LeakyReLU(alpha=0.05), name='Hidden2'),
            Dropout(0.2, name='Dropout2'),
            Dense(1, activation='sigmoid', name='Ouput')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            TruePositives(name='true_positives'),
            TrueNegatives(name='true_negatives'),
            FalsePositives(name='false_positives'),
            FalseNegatives(name='false_negatives')
        ])

        # Create a callback to print training progress
        print_progress = LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"Epoch {epoch+1}/{5} - "
                f"loss: {logs['loss']:.4f} - "
                f"accuracy: {logs['accuracy']:.4f}"
            )
        )

        # Train the model with the print_progress callback
        model.fit(X_train, y_train, epochs=5, callbacks=[print_progress], verbose=0)

        # Evaluate the model on the test set
        loss_and_metrics = model.evaluate(X_test, y_test, return_dict=True, verbose=0)

        # Get the predicted probabilities for the test set
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate additional evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        confusion_mat = confusion_matrix(y_test, y_pred)

        """# Save evaluation results to the same file
        with open(f"Data/{filename}_results.txt", "a") as file:
            file.write("Neural Network Evaluation Results:\n\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n")
            file.write(f"F1 Score: {f1:.4f}\n")
            file.write(f"F2 Score: {f2:.4f}\n")
            file.write(f"ROC AUC Score: {roc_auc:.4f}\n")
            file.write("\nConfusion Matrix:\n")
            file.write(np.array2string(confusion_mat, separator=', '))
            file.write("\n\nClassification Report:\n")
            file.write(classification_report(y_test, y_pred))"""

    except Exception as e:
        logger.error(f"Error during model building and training: {str(e)}")
        raise

    return model


def predict_single_company(file_path, models, X):
    """
    Predict the bankruptcy status of a single company using the trained models.

    Args:
        file_path (str): Path to the CSV file containing the company's financial data.
        models (dict): Dictionary of trained models.

    Returns:
        dict: Dictionary of predictions from each model.
    """
    try:
        # Load the company's data from the CSV file
        imput_flag = input("Are all columns in the data entered? (y/n): ").lower().strip()
        df = pd.read_csv(file_path)
        df.columns = ['_'.join(col.split()).strip() for col in df.columns]

        if imput_flag == 'n':
            X_selected_test = impute_test_data(X, df)

        # Preprocess the data
        X_test = df
        X_test = handle_skew_new(X_test)

        selected_columns_names = np.genfromtxt('Data/selected_features.csv', delimiter='\n', dtype=str)
        X_selected_test = X_test.filter(selected_columns_names, axis=1)
        
        X_expanded_test = calculate_financial_ratios(X_test)
        X_selected_test = np.concatenate((X_selected_test, X_expanded_test), axis=1)

        # Make predictions using each model
        predictions = {}
        for model_name, model in models.items():
            if model_name == 'Neural Network':
                y_pred = model.predict(X_selected_test)
                y_pred = (y_pred > 0.5).astype(int)[0]
            else:
                y_pred = model.predict(X_selected_test)
            predictions[model_name] = y_pred[0]

        return predictions

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error during prediction for single company: {str(e)}")
        raise


def terminal_interface(models, X):
    """
    Terminal-based interface for users to input a CSV file and view predictions for a single company.

    Args:
        models (dict): Dictionary of trained models.
    """
    print("\n\nWelcome to the Bankruptcy Prediction System!")
    print("Please provide the path to the CSV file containing the financial data of a single company.")

    while True:
        file_path = input("Enter the file path (or 'q' to quit): ")

        if file_path.lower() == 'q':
            print("Thank you for using the Bankruptcy Prediction System. Goodbye!")
            break

        try:
            # Make predictions for the single company
            single_company_predictions = predict_single_company(file_path, models, X)

            # Print the predictions
            print("\nSingle Company Predictions:")
            max_model_name_length = max(len(model_name) for model_name in single_company_predictions.keys())

            for model_name, prediction in single_company_predictions.items():
                result = 'Bankrupt' if prediction == 1 else 'Not Bankrupt'
                print(f"\t{model_name:<{max_model_name_length}}\t\t{result}")

            predictions_np = np.array(list(single_company_predictions.values()), dtype=int)
            majority_voting_result = 'Bankrupt' if st.mode(predictions_np)[0] == 1 else 'Not Bankrupt'

            print("\t" + ("-" * 40))
            print(f"\tMajority voting decision:\t{majority_voting_result}")

        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        print()


def main():
    try:
        # Load data
        logger.info("Loading data...")
        X, y = load_data('Data/Taiwan/taiwan.csv')

        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(X, y)

        models = {}

        # Build and train the model
        logger.info("Building and training the model...")
        models["Neural Network"] = build_model(X_train, X_test, y_train, y_test, np.shape(X_train)[1])

        # Run all models
        models = run_models(X_train, X_test, y_train, y_test, models)

        terminal_interface(models, X)

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == '__main__':
    main()
