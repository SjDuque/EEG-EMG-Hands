import numpy as np
import pandas as pd
import mediapipe as mp
import logging
import os

class FingerDataProcessor:
    """
    A class to process hand landmark data and calculate finger angles from a CSV file.
    """

    def __init__(self, angle_thresholds=None):
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

        # Define fingers
        self.fingers = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']

        # Generate hand_landmarks list from HandLandmark enum
        self.hand_landmarks = [landmark.name for landmark in self.HandLandmark]

        # Map each landmark to its index using HandLandmark enum
        self.hand_landmarks_idx_dict = {landmark.name: landmark.value for landmark in self.HandLandmark}

        # Define joint sets for each finger using landmark names
        self.joint_sets = {
            'THUMB':    [("WRIST", "THUMB_MCP",         "THUMB_TIP")],
            'INDEX':    [("WRIST", "INDEX_FINGER_MCP",  "INDEX_FINGER_TIP")],
            'MIDDLE':   [("WRIST", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_TIP")],
            'RING':     [("WRIST", "RING_FINGER_MCP",   "RING_FINGER_TIP")],
            'PINKY':    [("WRIST", "PINKY_MCP",         "PINKY_TIP")],
        }

        # Define angle thresholds (optional)
        if angle_thresholds is not None:
            self.joint_angle_thresholds = angle_thresholds
        else:
            self.joint_angle_thresholds = {
                'THUMB':    (132.5, 157.5),
                'INDEX':    (60, 160),
                'MIDDLE':   (40, 167.5),
                'RING':     (30, 167.5),
                'PINKY':    (30, 167.5)
            }
            

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculate the angle between three points a, b, and c in degrees.

        Parameters:
            a (tuple): (x, y, z) coordinates of point a.
            b (tuple): (x, y, z) coordinates of point b (vertex).
            c (tuple): (x, y, z) coordinates of point c.

        Returns:
            float: The angle in degrees. Returns np.nan if calculation is not possible.
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            # Avoid division by zero
            return np.nan
        
        # Compute the cosine of the angle using dot product
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        # Handle potential numerical issues
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        radians = np.arccos(cosine_angle)
        return np.degrees(radians)
    
    def get_percentage_angles(self, angles_dict):
        """
        Calculate the percentage of angles within the threshold for each finger.

        Parameters:
            angles_dict (dict): A dictionary with finger names as keys and their corresponding angles as values.

        Returns:
            dict: A dictionary with finger names as keys and their corresponding percentage of angles within the threshold.
        """
        percentage_dict = {}
        
        for finger, angles in angles_dict.items():
            if finger in self.joint_angle_thresholds:
                min_angle, max_angle = self.joint_angle_thresholds[finger]
                if np.isnan(angles):
                    percentage_dict[finger] = np.nan
                else:
                    percentage = (min(max_angle, max(min_angle, angles)) - min_angle) / (max_angle - min_angle) * 100
                    percentage_dict[finger] = percentage
            else:
                logging.warning(f"No angle thresholds defined for finger {finger}.")
                percentage_dict[finger] = np.nan
        
        return percentage_dict

    def get_calculated_angles(self, landmarks):
        """
        Calculate the angles for each finger based on hand landmarks.

        Parameters:
            landmarks (list of tuples): List containing (x, y, z) coordinates for each landmark.

        Returns:
            dict: A dictionary with finger names as keys and their corresponding angles as values.
        """
        angles_dict = {}
        
        for finger, joints in self.joint_sets.items():
            angles = []
            for j in joints:
                name_a, name_b, name_c = j
                
                # Map landmark names to indices
                idx_a = self.hand_landmarks_idx_dict.get(name_a)
                idx_b = self.hand_landmarks_idx_dict.get(name_b)
                idx_c = self.hand_landmarks_idx_dict.get(name_c)
                
                # Ensure all landmarks exist
                if idx_a is None or idx_b is None or idx_c is None:
                    logging.warning(f"Landmark names {name_a}, {name_b}, or {name_c} not found.")
                    angle = np.nan
                else:
                    a = landmarks[idx_a]
                    b = landmarks[idx_b]
                    c = landmarks[idx_c]
                    
                    angle = self.calculate_angle(a, b, c)
                angles.append(angle)
            
            # Average angle for the finger (useful if multiple joints per finger)
            # If all angles are NaN, the mean will be NaN
            if any(not np.isnan(angle) for angle in angles):
                angles_dict[finger] = np.nanmean(angles)
            else:
                angles_dict[finger] = np.nan
        
        return angles_dict

    def load_and_process_csv(self, csv_file_path):
        """
        Load the CSV file and process each row to calculate finger angles.

        Parameters:
            csv_file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing timestamps and corresponding finger angles.
        """
        try:
            # Read only required columns to optimize memory usage
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

        # Define a function to process each row
        def process_row(row):
            # Extract landmarks as a list of tuples (x, y, z)
            landmarks = [tuple(row.get(f'landmark_{i}_{coord}', np.nan) for coord in ['x', 'y', 'z']) for i in range(len(self.hand_landmarks))]
            return self.get_calculated_angles(landmarks)

        # Apply the function to each row
        angles_df = df.apply(process_row, axis=1, result_type='expand')

        # Combine the timestamp with the angles
        processed_df = pd.concat([df[['timestamp']].reset_index(drop=True), angles_df], axis=1)

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
