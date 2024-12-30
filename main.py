import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dask import dataframe as dd
import time

def load_dataset(single_processor=True):
    if single_processor:
        return pd.read_csv("leetcode_dataset.csv")
    else:
        return dd.read_csv("leetcode_dataset.csv")

def preprocess_data(data):
    data = data.dropna()

    # Feature engineering
    data['likes_to_dislikes_ratio'] = data['likes'] / (data['dislikes'] + 1)

    # Drop irrelevant or text-heavy columns
    irrelevant_columns = ['id', 'title', 'description', 'solution_link', 'url', 'similar_questions']
    data = data.drop(columns=[col for col in irrelevant_columns if col in data.columns], errors='ignore')

    def convert_shorthand(value):
        if isinstance(value, str) and value[-1] == 'K':
            return float(value[:-1]) * 1000
        if isinstance(value, str) and value[-1] == 'M':
            return float(value[:-1]) * 1_000_000
        return float(value)

    numeric_shorthand_columns = ['accepted', 'submissions']  # Add more columns if needed
    for col in numeric_shorthand_columns:
        if col in data.columns:
            data[col] = data[col].apply(convert_shorthand)

    if 'related_topics' in data.columns:
        data = pd.get_dummies(data, columns=['related_topics'], drop_first=True)
    if 'companies' in data.columns:
        data = pd.get_dummies(data, columns=['companies'], drop_first=True)
    if 'difficulty' in data.columns:
        difficulty_mapping = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        data['difficulty'] = data['difficulty'].map(difficulty_mapping)

    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    data[numerical_features] = (data[numerical_features] - data[numerical_features].mean()) / data[numerical_features].std()

    return data

def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return mae, rmse

def compare_processing():
    # Single Processor
    start_time = time.time()
    data = load_dataset(single_processor=True)
    data = preprocess_data(data)
    X = data.drop(['acceptance_rate'], axis=1)
    y = data['acceptance_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    mae, rmse = evaluate_model(model, X_test, y_test)
    single_time = time.time() - start_time

    print(f"Single Processor:\nMAE: {mae}\nRMSE: {rmse}\nTime Taken: {single_time:.2f} seconds\n")

    # Parallel Processing
    start_time = time.time()
    data = load_dataset(single_processor=False).compute()
    data = preprocess_data(data)
    X = data.drop(['acceptance_rate'], axis=1)
    y = data['acceptance_rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    mae, rmse = evaluate_model(model, X_test, y_test)
    parallel_time = time.time() - start_time

    print(f"Parallel Processing:\nMAE: {mae}\nRMSE: {rmse}\nTime Taken: {parallel_time:.2f} seconds\n")

if __name__ == "__main__":
    compare_processing()