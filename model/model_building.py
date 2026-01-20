"""
Breast Cancer Prediction System - Model Building
CSC415 Holiday Assignment - Project 5 (Part A)
Author: SOMADE TOLUWANI (22CH032062)

This script builds an SVM Classifier to predict tumor malignancy
using the Wisconsin Breast Cancer Dataset with 5 selected features.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

def build_and_train_model():
    print("=" * 60)
    print("🎗️ Breast Cancer Prediction - Model Building")
    print("=" * 60)
    
    # Step 1: Load Dataset
    print("\n📊 Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    X_full = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target # 0: Malignant, 1: Benign
    
    # Step 2: Feature Selection (Select 5 features as per assignment)
    # Recommended: radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
    # compactness_mean, concavity_mean, symmetry_mean
    selected_features = [
        'mean radius', 
        'mean texture', 
        'mean area', 
        'mean smoothness', 
        'mean concavity'
    ]
    X = X_full[selected_features]
    print(f"   Selected Features: {selected_features}")
    
    # Step 3: Feature Scaling (Mandatory for SVM)
    print("\n📏 Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 5: Implement SVM Algorithm
    print("\n🤖 Training Support Vector Machine (SVM) Classifier...")
    model = SVC(kernel='rbit', probability=True, random_state=42)
    # Using 'rbf' kernel for non-linear boundary
    model = SVC(C=1.0, kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 6: Evaluation
    y_pred = model.predict(X_test)
    print("\n📈 Performance Metrics:")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
    
    # Step 7: Save Model
    print("\n💾 Saving model to disk...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': selected_features,
        'target_names': ['Malignant', 'Benign']
    }
    joblib.dump(model_data, 'breast_cancer_model.pkl')
    print("   ✅ Saved as 'breast_cancer_model.pkl'")
    
    print("\n" + "=" * 60)
    print("✅ Model Building Complete!")

if __name__ == '__main__':
    build_and_train_model()
