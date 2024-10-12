import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time

# Load the saved model
model = load_model('emg_cnn_model.keras')

# Load the validation data
data = np.load('cnn_data_custom_channels.npz')
X_val, y_val = data['X_frames'], data['y_frames']

# Make predictions
y_pred = model.predict(X_val)

# Function to plot predicted vs. actual values for a few samples
def plot_predictions(y_true, y_pred, sample_indices):
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i] if num_samples > 1 else axes
        ax.plot(y_true[idx], label='True', marker='o')
        ax.plot(y_pred[idx], label='Predicted', marker='x')
        ax.set_title(f'Sample {idx+1}')
        ax.set_xlabel('Finger')
        ax.set_ylabel('Percentage')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# Loop to display random 5 predictions every 2 seconds
try:
    while True:
        # Randomly select 5 indices
        random_indices = np.random.choice(len(y_val), 5, replace=False)
        # Plot the selected samples
        plot_predictions(y_val, y_pred, random_indices)
        # Pause for 2 seconds
        time.sleep(2)
except KeyboardInterrupt:
    print("Stopped displaying samples.")
