import os
import numpy as np
import pandas as pd

# Define the root directory and the output folder
root_dir = '{directory}}\\MOP'  # Replace this with the actual path of the root directory
wo_outlier_dir = os.path.join(root_dir, 'synced_data', 'woutlier')
output_dir = os.path.join(root_dir, 'featurewith')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Subjects to process (subject1, subject2, etc.)
subjects = ['subject1', 'subject2', 'subject3', 'subject4']
timesplit = 'timesplit'  # Assuming all subjects have the same folder structure

# CSV files specifically for Kinect data (train, validate, test)
kinect_files = {
    'train': 'kinect_data_train.csv',
    'validate': 'kinect_data_validate.csv',
    'test': 'kinect_data_test.csv'
}

# Columns to remove (1-based index from the description)
columns_to_remove = [8, 12, 22, 23, 24, 25, 33, 37, 47, 48, 49, 50, 58, 62, 72, 73, 74, 75]
# Convert 1-based indexing to 0-based for pandas
columns_to_remove = [col - 1 for col in columns_to_remove]

# Initialise dictionaries to store merged Kinect data
merged_kinect = {
    'train': [],
    'validate': [],
    'test': []
}

# Helper function to load and process Kinect CSV data
def load_and_process_kinect_csv(file_path):
    # Load the data, skipping the first row (assuming it's label information)
    data = pd.read_csv(file_path, header=None).iloc[1:, :]  # Skip the first row
    
    # Drop the columns that need to be removed
    data = data.drop(columns=columns_to_remove, axis=1)
    
    # Convert the data to a NumPy array
    kinect_data = data.to_numpy()
    
    return np.array(kinect_data, dtype=np.float64)

# Loop through each subject and process each Kinect CSV file
for subject in subjects:
    subject_path = os.path.join(wo_outlier_dir, subject, timesplit)
    
    for phase, kinect_file in kinect_files.items():
        kinect_data_path = os.path.join(subject_path, kinect_file)
        
        if os.path.exists(kinect_data_path):
            # Load and process the Kinect CSV file
            processed_kinect_data = load_and_process_kinect_csv(kinect_data_path)
            
            # Append processed Kinect data to the corresponding phase (train/validate/test)
            merged_kinect[phase].append(processed_kinect_data)
        else:
            print(f"Missing Kinect data for {subject}, phase {phase}.")

# Convert lists to NumPy arrays and save them as npy files with pickle enabled
for phase in ['train', 'validate', 'test']:
    if merged_kinect[phase]:  # Ensure there's data to merge
        merged_kinect[phase] = np.vstack(merged_kinect[phase])  # Stack Kinect data vertically
        
        # Save the merged Kinect data as .npy files with pickle enabled
        kinect_output_path = os.path.join(output_dir, f'kinect_data_{phase}.npy')
        np.save(kinect_output_path, merged_kinect[phase], allow_pickle=True)
        
        # Output the shape of the saved files for confirmation
        print(f"Saved kinect_data_{phase}.npy with shape {merged_kinect[phase].shape}")
    else:
        print(f"No Kinect data to merge for {phase}, skipping.")

print("Kinect npy files created in 'featurewout' folder.")
