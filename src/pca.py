import numpy as np
import joblib
import logging
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from simple_model import (
    process_sessions, 
    create_labels, 
    print_label_distribution, 
    FINGER_GROUPS, 
    JOINT_ANGLE_THRESHOLDS, 
    SELECTED_CHANNELS
)

# Configuration Parameters
FRAME_SIZES = [128, 64]
N_COMPONENTS = 10  # Reduced number of PCA components for efficiency
training_session_dirs = ['EMG Hand Data 20241030_232021', 'EMG Hand Data 20241030_233323']  # Add your paths
validation_session_dirs = ['EMG Hand Data 20241031_001704']
test_session_dirs = ['EMG Hand Data 20241031_002827']

def perform_pca(X_train, X_val, X_test, n_components):
    """
    Perform PCA on the input feature data.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    logging.info(f"PCA completed with {n_components} components.")
    return X_train_pca, X_val_pca, X_test_pca, pca

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model for multi-label classification using OneVsRestClassifier.
    """
    # Random Forest with OneVsRestClassifier for multi-label support
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Adjust n_estimators if needed
    model = OneVsRestClassifier(rf)
    model.fit(X_train, y_train)
    logging.info("Multi-label Random Forest model training completed.")
    return model

def evaluate_model(model, X, y, group_names, label='Validation Set'):
    """
    Evaluate the model and print classification metrics.
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"{label} Accuracy: {accuracy * 100:.2f}%")

    # Classification report for each label
    print(f"\n{label} Classification Report:")
    print(classification_report(y, y_pred, target_names=group_names, zero_division=0))

    # Confusion matrix for each label
    for i, finger_group in enumerate(group_names):
        print(f"\nConfusion Matrix for {finger_group} ({label}):")
        cm = confusion_matrix(y[:, i], y_pred[:, i])
        print(cm)

def main():
    # Process training sessions and scale the data
    X_train_full, y_train_full, skipped_train, emg_means, emg_stds = process_sessions(
        training_session_dirs,
        use_existing_scaler=False
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)

    # Process and scale validation data
    X_val, y_val, skipped_val = process_sessions(
        validation_session_dirs,
        emg_means=emg_means,
        emg_stds=emg_stds,
        use_existing_scaler=True
    )
    X_val_scaled = scaler.transform(X_val)

    # Process and scale test data
    X_test, y_test, skipped_test = process_sessions(
        test_session_dirs,
        emg_means=emg_means,
        emg_stds=emg_stds,
        use_existing_scaler=True
    )
    X_test_scaled = scaler.transform(X_test)

    # Perform PCA
    X_train_pca, X_val_pca, X_test_pca, pca = perform_pca(X_train_scaled, X_val_scaled, X_test_scaled, N_COMPONENTS)
    
    # Optionally, reduce training data size for testing
    X_train_pca, y_train_full = X_train_pca[:10000], y_train_full[:10000]

    # Train Random Forest model
    model = train_random_forest(X_train_pca, y_train_full)

    # Evaluate on validation and test sets
    evaluate_model(model, X_val_pca, y_val, list(FINGER_GROUPS.keys()), label='Validation Set')
    evaluate_model(model, X_test_pca, y_test, list(FINGER_GROUPS.keys()), label='Test Set')

    # Save the PCA model and scaler
    joblib.dump(pca, 'pca_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    logging.info("PCA model and scaler saved.")

if __name__ == "__main__":
    main()
