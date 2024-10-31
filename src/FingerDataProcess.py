import numpy as np
import pandas as pd
import mediapipe as mp

class FingerDataProcessor:
    """
    A class to process hand landmark data and calculate finger angles from a CSV file.
    """

    def __init__(self, angle_thresholds=None):
        """
        Initialize the HandAngleProcessor with optional angle thresholds.

        Parameters:
            angle_thresholds (dict, optional): A dictionary specifying angle thresholds for each finger.
                                               If None, default thresholds are used.
        """
        # Initialize MediaPipe HandLandmark
        self.mp_hands = mp.solutions.hands
        self.HandLandmark = self.mp_hands.HandLandmark

        # Define fingers
        self.fingers = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']

        # Generate hand_landmarks list from HandLandmark enum
        self.hand_landmarks = [landmark.name for landmark in self.HandLandmark]

        # Map each landmark to its index using HandLandmark enum
        self.hand_landmarks_idx_dict = {landmark.name: landmark.value for landmark in self.HandLandmark}

        # Define joint sets for each finger using landmark names
        self.joint_sets = {
            'THUMB':    [("WRIST", "THUMB_MCP", "THUMB_TIP")],
            'INDEX':    [("WRIST", "INDEX_FINGER_MCP", "INDEX_FINGER_TIP")],
            'MIDDLE':   [("WRIST", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_TIP")],
            'RING':     [("WRIST", "RING_FINGER_MCP", "RING_FINGER_TIP")],
            'PINKY':    [("WRIST", "PINKY_MCP", "PINKY_TIP")],
        }

        # Define angle thresholds (optional)
        if angle_thresholds is not None:
            self.joint_angle_thresholds = angle_thresholds
        else:
            self.joint_angle_thresholds = {
                'THUMB': (132.5, 157.5),
                'INDEX': (25, 165),
                'MIDDLE': (15, 167.5),
                'RING': (15, 167.5),
                'PINKY': (15, 167.5)
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
                
                # Check if all landmarks exist
                if idx_a is None or idx_b is None or idx_c is None:
                    print(f"Warning: Landmark names {name_a}, {name_b}, or {name_c} not found.")
                    angle = np.nan
                else:
                    a = landmarks[idx_a]
                    b = landmarks[idx_b]
                    c = landmarks[idx_c]
                    
                    # Check for NaN values in the landmarks
                    if any(np.isnan(coord) for coord in a + b + c):
                        angle = np.nan  # Assign NaN if any coordinate is missing
                    else:
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
        # Read the CSV file
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"Error: File '{csv_file_path}' not found.")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Error: File '{csv_file_path}' is empty.")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file: {e}")
            return pd.DataFrame()
        
        # Ensure all required columns are present
        expected_columns = ['timestamp'] + [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            print(f"Error: The following required columns are missing in the CSV: {missing_columns}")
            return pd.DataFrame()
        
        processed_data = []
        skipped_rows = 0
        total_rows = len(df)
        
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            timestamp = row['timestamp']
            landmarks = []
            row_has_nan = False  # Flag to indicate if the row has any NaN
            
            # Extract landmarks from the row
            for i in range(21):
                x = row.get(f'landmark_{i}_x', np.nan)
                y = row.get(f'landmark_{i}_y', np.nan)
                z = row.get(f'landmark_{i}_z', np.nan)
                
                # Check if any coordinate is NaN
                if pd.isna(x) or pd.isna(y) or pd.isna(z):
                    row_has_nan = True
                    break  # Exit the landmark loop immediately
                else:
                    landmarks.append((x, y, z))
            
            if row_has_nan:
                skipped_rows += 1
                continue  # Skip processing this row entirely
            
            # Calculate angles for the current set of landmarks
            angles = self.get_calculated_angles(landmarks)
            
            # Check if any angle is NaN (optional, based on calculation success)
            if any(np.isnan(angle) for angle in angles.values()):
                skipped_rows += 1
                continue  # Skip this row if any angle calculation failed
            
            # Create a dictionary entry with timestamp and angles
            data_entry = {'timestamp': timestamp}
            data_entry.update(angles)
            processed_data.append(data_entry)
        
        print(f"Processed {total_rows} rows.")
        print(f"Skipped {skipped_rows} rows due to missing or invalid data.")
        
        # Convert the list of dictionaries to a pandas DataFrame
        processed_df = pd.DataFrame(processed_data)
        
        return processed_df

    def save_processed_data(self, processed_data, output_csv_path):
        """
        Save the processed data to a new CSV file.

        Parameters:
            processed_data (pd.DataFrame): The processed data to save.
            output_csv_path (str): Path to the output CSV file.
        """
        if processed_data.empty:
            print("No data to save.")
            return
        
        try:
            processed_data.to_csv(output_csv_path, index=False)
            print(f"Processed data saved to '{output_csv_path}'.")
        except Exception as e:
            print(f"Error saving processed data: {e}")

# Example Usage
if __name__ == "__main__":
    # Path to your CSV file
    csv_file = 'EMG Hand Data 20241031_002827/fingers.csv'  # Replace with your actual CSV file path
    output_csv = 'EMG Hand Data 20241031_002827/finger_angles.csv'  # Desired output file path

    # Initialize the processor
    processor = FingerDataProcessor()

    # Process the CSV and get the angles data as a DataFrame
    angles_df = processor.load_and_process_csv(csv_file)

    # Example: Print the first few entries
    if not angles_df.empty:
        print("First few processed entries:")
        print(angles_df.head())
    else:
        print("No valid data found in the CSV file.")

    # Optionally, save the processed data to a new CSV
    processor.save_processed_data(angles_df, output_csv)
