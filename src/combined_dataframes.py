import os
import pandas as pd
from pathlib import Path

def combine_csv_files(input_folders, output_folder, file_names):
    """
    Combines CSV files with the same name from multiple folders into a single CSV per file type.

    Parameters:
    - input_folders (list of str): List of paths to input folders.
    - output_folder (str): Path to the output folder where combined CSVs will be saved.
    - file_names (list of str): List of CSV file names to combine (e.g., ['emg.csv', 'fingers.csv']).
    """
    
    # Ensure the output directory exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    print(f"Output directory is set to: {output_folder}")

    for file_name in file_names:
        combined_df = pd.DataFrame()
        print(f"\nProcessing file: {file_name}")

        for folder in input_folders:
            file_path = Path(folder) / file_name
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist. Skipping this file.")
                continue

            try:
                df = pd.read_csv(file_path)
                # Optionally, add a column to identify the source folder
                # df['source_folder'] = folder
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                print(f"Added data from: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        if not combined_df.empty:
            output_file_path = Path(output_folder) / file_name
            try:
                combined_df.to_csv(output_file_path, index=False)
                print(f"Combined data saved to: {output_file_path}")
            except Exception as e:
                print(f"Error saving combined file {output_file_path}: {e}")
        else:
            print(f"No data to save for {file_name}.")

def main():
    # ============================
    # Configuration Variables
    # ============================

    # List of input folder paths containing the CSV files
    input_folders = [
        'EMG Hand Data 20241022_215347',
        'EMG Hand Data 20241022_220748',
    ]

    # Path to the output folder where combined CSVs will be saved
    output_folder = 'emg_session_1'

    # List of CSV file names to combine
    file_names = ['emg.csv', 'fingers.csv']
    # If you have additional or different CSV files, modify the list accordingly
    # Example: file_names = ['emg.csv', 'fingers.csv', 'additional.csv']

    # ============================
    # End of Configuration
    # ============================

    # Validate input folders
    valid_folders = []
    for folder in input_folders:
        if Path(folder).is_dir():
            valid_folders.append(folder)
        else:
            print(f"Warning: {folder} is not a valid directory and will be skipped.")

    if not valid_folders:
        print("Error: No valid input folders provided. Exiting.")
        return

    combine_csv_files(valid_folders, output_folder, file_names)

if __name__ == "__main__":
    main()
