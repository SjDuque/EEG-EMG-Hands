import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import time
import numpy as np
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

def collect_data(board, duration, label):
    print(f"Collecting data for {label} state...")
    data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        sample = board.get_current_board_data(10)  # Get the last 10 samples
        if sample.shape[1] > 0:
            data.append(sample)
        time.sleep(0.01)  # Adjust the sleep time as needed for your data rate
    if data:
        data = np.concatenate(data, axis=1)  # Concatenate along the second axis to ensure consistent shape
    return data

def train_model(model, focus_data, non_focus_data):
    print("Training model...")
    X = np.concatenate((focus_data.T, non_focus_data.T), axis=0)  # Transpose to get samples as rows
    y = np.concatenate((np.ones(focus_data.shape[1]), np.zeros(non_focus_data.shape[1])), axis=0)
    
    model.partial_fit(X, y, classes=np.array([0, 1]))

    # Evaluate the model
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model updated with accuracy: {accuracy * 100:.2f}%")
    
    return model

def infer(model, board, duration):
    print(f"Running inference for {duration} seconds...")
    start_time = time.time()
    while time.time() - start_time < duration:
        data = board.get_current_board_data(10)  # Get the last 10 samples
        if data.shape[1] > 0:
            sample = data[:, :data.shape[0]].reshape(1, -1)  # Reshape for prediction
            prediction = model.predict(sample)
            print(f"Focus: {prediction[0]}")
        time.sleep(0.5)  # Adjust the prediction frequency as needed
    print("Inference completed.")

def main():
    params = BrainFlowInputParams()
    params.serial_port = '/dev/cu.usbserial-D200PQ3N'
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    model = SGDClassifier(loss='hinge', random_state=42)

    try:
        while True:
            user_input = input("Press 'T' to enter training mode, 'I' to run inference, or 'Q' to quit: ").strip().upper()
            
            if user_input == 'T':
                print("Training mode activated")
                focus_data = collect_data(board, 2.5, 'focus')
                non_focus_data = collect_data(board, 2.5, 'non-focus')
                focus_data = np.concatenate((focus_data, collect_data(board, 2.5, 'focus')), axis=1)
                non_focus_data = np.concatenate((non_focus_data, collect_data(board, 2.5, 'non-focus')), axis=1)
                model = train_model(model, focus_data, non_focus_data)
                print("Training completed.")

            elif user_input == 'I':
                if model is not None:
                    infer(model, board, 5)
                else:
                    print("Model has not been trained yet. Please enter training mode first by pressing 'T'.")

            elif user_input == 'Q':
                break

            else:
                print("Invalid input. Please press 'T' to enter training mode, 'I' to run inference, or 'Q' to quit.")

    except KeyboardInterrupt:
        print("Program interrupted")

    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()
