import numpy as np
import pandas as pd
import mediapipe as mp
import logging
import os
import warnings

class FingerDataProcessor:
    """
    A class to process hand landmark data and calculate finger angles from a CSV file.
    """

    def __init__(self, 
                 angle_thresholds={
                    'thumb_ip':    (155, 162.5),
                    'index':        (90, 120),
                    'middle':       (90, 145),
                    'ring':         (70, 145),
                    'pinky':        (80, 157.5)
                 }
                ):
        """
        Initialize the FingerDataProcessor with optional angle thresholds.

        Parameters:
            angle_thresholds (dict, optional): A dictionary specifying angle thresholds for each finger.
                                               If None, default thresholds are used.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        # Initialize MediaPipe HandLandmark
        self.HandLandmark = mp.solutions.hands.HandLandmark

        # Generate hand_landmarks list from HandLandmark enum
        self.hand_landmarks = [landmark.name for landmark in self.HandLandmark]

        # Map each landmark to its index using HandLandmark enum
        self.hand_landmarks_idx_dict = {landmark.name: landmark.value for landmark in self.HandLandmark}

        # Define joint sets for each finger using landmark names
        self.joint_sets = {
            'thumb_ip':    [("WRIST", "THUMB_IP",          "THUMB_TIP")],
            'index_mcp':    [("WRIST", "INDEX_FINGER_MCP",  "INDEX_FINGER_TIP")],
            'middle_mcp':   [("WRIST", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_TIP")],
            'ring_mcp':     [("WRIST", "RING_FINGER_MCP",   "RING_FINGER_TIP")],
            'pinky_mcp':    [("WRIST", "PINKY_MCP",         "PINKY_TIP")],
        }
        
        # Define angle thresholds for each finger
        self.angle_thresholds = angle_thresholds

        # Precompute the indices for each joint set
        self.precomputed_joints = {}
        for finger, joints in self.joint_sets.items():
            self.precomputed_joints[finger] = []
            for j in joints:
                name_a, name_b, name_c = j
                idx_a = self.hand_landmarks_idx_dict.get(name_a)
                idx_b = self.hand_landmarks_idx_dict.get(name_b)
                idx_c = self.hand_landmarks_idx_dict.get(name_c)
                if idx_a is None or idx_b is None or idx_c is None:
                    logging.warning(f"Landmark names {name_a}, {name_b}, or {name_c} not found.")
                    self.precomputed_joints[finger].append((np.nan, np.nan, np.nan))
                else:
                    self.precomputed_joints[finger].append((idx_a, idx_b, idx_c))

    @staticmethod
    def calculate_angle_vectorized(a, b, c):
        """
        Calculate the angle between three points a, b, and c in degrees using vectorized operations.

        Parameters:
            a (np.ndarray): Array of shape (n, 3) for point a.
            b (np.ndarray): Array of shape (n, 3) for point b (vertex).
            c (np.ndarray): Array of shape (n, 3) for point c.

        Returns:
            np.ndarray: Array of angles in degrees. Returns np.nan where calculation is not possible.
        """
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba, axis=1)
        norm_bc = np.linalg.norm(bc, axis=1)

        # Avoid division by zero
        valid = (norm_ba != 0) & (norm_bc != 0)
        cosine_angle = np.einsum('ij,ij->i', ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine_angle))
        angles[~valid] = np.nan
        return angles

    def get_percentage_angles_vectorized(self, angles_dict):
        """
        Calculate the percentage of angles within the threshold for each finger using vectorized operations.

        Parameters:
            angles_dict (dict): A dictionary with finger names as keys and their corresponding angles as np.ndarray.

        Returns:
            pd.DataFrame: DataFrame with percentage of angles within the threshold for each finger.
        """
        percentage_data = {}
        for finger, angles in angles_dict.items():
            if finger in self.angle_thresholds:
                min_angle, max_angle = self.angle_thresholds[finger]
                with np.errstate(invalid='ignore'):
                    percentage = (angles - min_angle) / (max_angle - min_angle)
                    percentage = np.clip(percentage, 0, 1)
                percentage_data[finger] = percentage
            else:
                logging.warning(f"No angle thresholds defined for finger {finger}.")
                percentage_data[finger] = np.full_like(next(iter(angles_dict.values())), np.nan)
        return pd.DataFrame(percentage_data)

    def get_calculated_angles_vectorized(self, landmarks_array):
        """
        Calculate the angles for each finger based on hand landmarks using vectorized operations.

        Parameters:
            landmarks_array (np.ndarray): Array of shape (n, 21, 3) containing (x, y, z) coordinates.

        Returns:
            dict: A dictionary with finger names as keys and their corresponding angles as np.ndarray.
        """
        angles_dict = {}
        for finger, joints in self.precomputed_joints.items():
            angles = []
            for idx_a, idx_b, idx_c in joints:
                if np.isnan(idx_a) or np.isnan(idx_b) or np.isnan(idx_c):
                    angles.append(np.full(landmarks_array.shape[0], np.nan))
                else:
                    a = landmarks_array[:, int(idx_a), :]
                    b = landmarks_array[:, int(idx_b), :]
                    c = landmarks_array[:, int(idx_c), :]
                    angle = self.calculate_angle_vectorized(a, b, c)
                    angles.append(angle)
            # Average angle for the finger
            if angles:
                angles_stack = np.vstack(angles)
                # Ignore 'RuntimeWarning: Mean of empty slice' when all angles are NaN
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    angles_mean = np.nanmean(angles_stack, axis=0)
                angles_dict[finger] = angles_mean
            else:
                angles_dict[finger] = np.full(landmarks_array.shape[0], np.nan)
        return angles_dict

    def load_and_process_csv(self, csv_file_path):
        """
        Load the CSV file and process to calculate finger angles using vectorized operations.

        Parameters:
            csv_file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing timestamps and corresponding finger angles.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path, usecols=lambda col: col.startswith('landmark_') or col == 'timestamp')
        except FileNotFoundError:
            logging.error(f"File '{csv_file_path}' not found.")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            logging.error(f"File '{csv_file_path}' is empty.")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing CSV file: {e}")
            return pd.DataFrame()

        # Define expected columns
        expected_columns = ['timestamp'] + [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logging.error(f"The following required columns are missing in the CSV: {missing_columns}")
            return pd.DataFrame()

        total_rows = len(df)
        logging.info(f"Total rows: {total_rows}")

        if df.empty:
            logging.warning("No data available in the CSV file.")
            return pd.DataFrame()

        # Extract landmark data into a NumPy array of shape (n, 21, 3)
        landmark_cols = [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
        landmarks = df[landmark_cols].to_numpy().reshape(-1, 21, 3)

        # Calculate angles using vectorized operations
        angles_dict = self.get_calculated_angles_vectorized(landmarks)

        # Calculate percentage angles
        percentage_df = self.get_percentage_angles_vectorized(angles_dict)

        # Combine with timestamp
        processed_df = pd.concat([df['timestamp'], percentage_df], axis=1)

        logging.info(f"Processed {len(processed_df)} rows.")
        
        return processed_df

    def save_processed_data(self, processed_data, output_csv_path):
        """
        Save the processed data to a new CSV file.

        Parameters:
            processed_data (pd.DataFrame): The processed data to save.
            output_csv_path (str): Path to the output CSV file.
        """
        if processed_data.empty:
            logging.warning("No data to save.")
            return
        
        try:
            processed_data.to_csv(output_csv_path, index=False)
            logging.info(f"Processed data saved to '{output_csv_path}'.")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")

# Example Usage
if __name__ == "__main__":

    # Path to the data_session_2 directory
    base_folder = 'data'

    # Initialize the processor
    processor = FingerDataProcessor()

    # Iterate over each 'EMG Hand Data' folder in base_folder
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path) and 'EMG Hand Data' in folder_name:
            csv_file = os.path.join(folder_path, 'fingers.csv')
            output_csv = os.path.join(folder_path, 'finger_angles.csv')
            
            if os.path.exists(csv_file):
                # Process the CSV and get the angles data as a DataFrame
                angles_df = processor.load_and_process_csv(csv_file)
                
                # Example: Print the first few entries
                if not angles_df.empty:
                    print(f"First few processed entries in {folder_name}:")
                    print(angles_df.head())
                else:
                    print(f"No valid data found in the CSV file in {folder_name}.")

                # Optionally, save the processed data to a new CSV
                processor.save_processed_data(angles_df, output_csv)
            else:
                print(f"'fingers.csv' not found in {folder_name}")
