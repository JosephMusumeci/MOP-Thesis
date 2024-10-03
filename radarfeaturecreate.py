import os
import numpy as np
import pandas as pd

# Define the root directory and the output folder
root_dir = '{directoty}\\MOP'  # Replace this with the actual path of the root directory
wo_outlier_dir = os.path.join(root_dir, 'synced_data', 'woutlier')
output_dir = os.path.join(root_dir, 'featurewith')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Subjects to process (subject1, subject2, etc.)
subjects = ['subject1', 'subject2', 'subject3', 'subject4']
timesplit = 'timesplit'  # Assuming all subjects have the same folder structure

# CSV files specifically for Radar data (train, validate, test)
radar_files = {
    'train': 'radar_data_train.csv',
    'validate': 'radar_data_validate.csv',
    'test': 'radar_data_test.csv'
}

# Initialise dictionaries to store merged Radar data
merged_radar = {
    'train': [],
    'validate': [],
    'test': []
}

# Helper function to load, sort, and process radar CSV data
def load_and_process_radar_csv(file_path):
    # Load the data into a pandas DataFrame
    radar_data = pd.read_csv(file_path)

    # Extract frame numbers
    frames = radar_data['Frame #'].unique()
    
    # Initialise a list to store data per frame
    processed_data = []

    # Loop through each frame
    for frame in frames:
        # Filter data for the current frame
        frame_data = radar_data[radar_data['Frame #'] == frame]

        # Sort data by X, Y, Z coordinates
        frame_data = frame_data.sort_values(by=['X', 'Y', 'Z'], ascending=True)

        # Select first 64 rows (or less) and fill up with zeros if needed
        # Extract X, Y, Z, Doppler, Intensity columns (columns 3, 4, 5, 6, 7 in the CSV)
        radar_points = frame_data[['X', 'Y', 'Z', 'Doppler', 'Intensity']].to_numpy()

        # Pad with zeros if there are fewer than 64 objects
        if radar_points.shape[0] < 64:
            radar_points = np.pad(radar_points, ((0, 64 - radar_points.shape[0]), (0, 0)), mode='constant')

        # Reshape the data to (8, 8, 5)
        radar_points = radar_points.reshape(8, 8, 5)
        
        # Add processed frame data to the list
        processed_data.append(radar_points)

    # Stack all frame data into a single numpy array
    return np.array(processed_data, dtype=np.float64)

# Loop through each subject and process each Radar CSV file
for subject in subjects:
    subject_path = os.path.join(wo_outlier_dir, subject, timesplit)
    
    for phase, radar_file in radar_files.items():
        radar_data_path = os.path.join(subject_path, radar_file)
        
        if os.path.exists(radar_data_path):
            # Load and process the Radar CSV file
            processed_radar_data = load_and_process_radar_csv(radar_data_path)

            # Scale the Intensity values in the last column from 16 bit Int scale to 0 to 1
            processed_radar_data[:, :, :, 4] =  processed_radar_data[:, :, :, 4] / 65535.0

            # Append processed radar data to the corresponding phase (train/validate/test)
            merged_radar[phase].append(processed_radar_data)
        else:
            print(f"Missing Radar data for {subject}, phase {phase}.")

# Convert lists to NumPy arrays and save them as npy files with pickle enabled
for phase in ['train', 'validate', 'test']:
    if merged_radar[phase]:  # Ensure there's data to merge
        merged_radar[phase] = np.vstack(merged_radar[phase])  # Stack Radar data vertically
        
        # Save the merged radar data as .npy files with pickle enabled
        radar_output_path = os.path.join(output_dir, f'radar_data_{phase}.npy')

        np.save(radar_output_path, merged_radar[phase], allow_pickle=True)
        
        # Output the shape of the saved files for confirmation
        print(f"Saved radar_data_{phase}.npy with shape {merged_radar[phase].shape}")
    else:
        print(f"No Radar data to merge for {phase}, skipping.")
