# Imports
# standard imports
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def random_forest_model(df, readme):
    # Prepare the features and labels
    X = df[readme]
    y = df['language']
    # Split the data into train, validate, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    # Train the random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_vec, y_train)
    # Make predictions on the validation set
    y_val_pred = rf_classifier.predict(X_val_vec)
    # Evaluate the model on validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print('Validation Accuracy:', val_accuracy)
    # Make predictions on the test set
    y_test_pred = rf_classifier.predict(X_test_vec)
    # Evaluate the model on test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print('Test Accuracy:', test_accuracy)
    return val_accuracy, test_accuracy



def create_models_df(readme, model, val_accuracy, test_accuracy):
    metric_df = pd.DataFrame(data=[
        {'Readme': readme,
         'Model': model,
         'validate_accuracy': val_accuracy,
         'test_accuracy': test_accuracy,
    }]
    )
    return metric_df



def add_to_models_df(model, readme, metric_df, val_accuracy, test_accuracy):
    metric_df = metric_df.append(
        {'Readme': readme,
         'Model': model,
         'validate_accuracy': val_accuracy,
         'test_accuracy': test_accuracy,}, ignore_index=True)
    return metric_df




