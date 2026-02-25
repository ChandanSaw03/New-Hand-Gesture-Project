import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_and_save_model(data_dir='data', model_path='models/gesture_model.pkl'):
    print("Loading data from:", data_dir)
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        return
        
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}. Please run data collection first.")
        return
        
    df_list = []
    for f in csv_files:
        filepath = os.path.join(data_dir, f)
        df_list.append(pd.read_csv(filepath))
        
    # Combine all data
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Split features and labels
    X = combined_df.drop('label', axis=1)
    y = combined_df['label']
    
    print(f"Total samples: {len(X)}")
    print(f"Gestures found: {y.unique()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\nModel successfully saved to: {model_path}")

if __name__ == "__main__":
    train_and_save_model()
