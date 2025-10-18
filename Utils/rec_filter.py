import pandas as pd
from .fHALF import fHALF
import os

# Get the current file's directory (Utils directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
csv_path = os.path.join(current_dir, '12koutput.csv')

# Read the CSV file
y = pd.read_csv(csv_path)

print("Columns in the DataFrame:", y.columns)

# Convert `organ` and `symptom` columns to lowercase
y['organ'] = y['organ'].str.lower()
y['symptom'] = y['symptom'].str.lower()

# Sort the dataset by `organ` and `symptom`
z = y.sort_values(by=['organ', 'symptom']).reset_index(drop=True)

# Group by `symptom` and `organ` to calculate the count
final_ds = (
    z.groupby(['symptom', 'organ'])
    .size()
    .reset_index(name='cnt')  # Add a `cnt` column for the count
)

# Sort by `cnt` in descending order
f = final_ds.sort_values(by='cnt', ascending=False).reset_index(drop=True)

# Remove rows with `unspecified` organ or invalid symptom values
fs = f[
    (f['organ'] != 'unspecified') &
    (f['symptom'] != 'organ') &
    (f['symptom'] != 'symptom')
].reset_index(drop=True)

# Display the final DataFrame
print(fs)

def rec_filter(input_string, filtered_data):
    # Use fHALF to extract all medical entities independently
    entities_list = fHALF(input_string)
    
    # Convert to DataFrame for easier handling
    entities_df = pd.DataFrame(entities_list, columns=['entity', 'duration', 'type'])
    # Keep duration as text (no conversion to integer)
    
    # Rename columns to match expected format
    entities_df = entities_df.rename(columns={'entity': 'symptom', 'type': 'organ'})
    
    # Convert the DataFrame to a list of tuples and return
    return list(entities_df[['symptom', 'duration', 'organ']].itertuples(index=False, name=None))

# # Example Usage
# fs_data = {
#     'symptom': ['symp1', 'symp2', 'symp3', 'symp1'],
#     'organ': ['org1', 'org2', 'org3', 'org1'],
#     'cnt': [10, 20, 30, 15]
# }
# fs = pd.DataFrame(fs_data)