import sys
import os
import numpy as np
import pandas as pd

def read_and_concat_psv_files(folder):
    all_dataframes = []
    for filename in os.listdir(folder):
        if filename.endswith(".psv") and filename.startswith("p"):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path, sep='|')
            # Extract PID from the filename
            pid = filename[2:-4]
            df['PID'] = pid
            all_dataframes.append(df)
    
    if not all_dataframes:
        print("No PSV files found in the specified folder.")
        sys.exit(1)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

def create_count_df(combined_df):
    # Count occurrences of SepsisLabel for each PID
    count_0 = combined_df[combined_df['SepsisLabel'] == 0].groupby('PID').size().reset_index(name='count_of_0')
    count_1 = combined_df[combined_df['SepsisLabel'] == 1].groupby('PID').size().reset_index(name='count_of_1')

    # Merge the counts
    count_df = pd.merge(count_0, count_1, on='PID', how='outer').fillna(0)
    
    # Ensure columns are integers
    count_df['count_of_0'] = count_df['count_of_0'].astype(int)
    count_df['count_of_1'] = count_df['count_of_1'].astype(int)
    
    count_df = count_df.sort_values(by='count_of_1', ascending=False)
    
    count_df['cumulative_count_of_1'] = count_df['count_of_1'].cumsum()

    return count_df

# find pids within each threshold group
def find_pids_crossing_threshold(count_df, thresholds=[0.7, 0.2, 0.1]):
    
    # Sort count_df by count_of_1 in descending order
    count_df = count_df.sort_values(by='count_of_1', ascending=False)
    
    # Initialize a dictionary to store the PIDs for each threshold
    pid_groups = {threshold: [] for threshold in thresholds}
    
    total_count_of_1 = count_df['count_of_1'].sum()
    cumulative_count = 0
    
    i = 0
    current_group = []
    threshold_value = total_count_of_1 * thresholds[i]

    for index, row in count_df.iterrows():
        cumulative_count += row['count_of_1']
        current_group.append(row['PID'])
        # print(cumulative_count, total_count_of_1, threshold_value)
        
        if cumulative_count >= threshold_value:
            pid_groups[thresholds[i]] = current_group
            current_group = []
            cumulative_count = 0
            i += 1
            threshold_value = total_count_of_1 * thresholds[i]

    if current_group:
        pid_groups[thresholds[i]] = current_group

    return pid_groups


def create_dataframes_for_groups(combined_df, pid_groups):
    train_df = combined_df[combined_df['PID'].isin(pid_groups[0.7])]
    batch_df = combined_df[combined_df['PID'].isin(pid_groups[0.2])]
    client_df = combined_df[combined_df['PID'].isin(pid_groups[0.1])]
    
    # Split batch_df equally into three DataFrames
    batch_dfs = np.array_split(batch_df, 3)
    
    return train_df, batch_dfs, client_df


def save_dataframes_to_psv(train_df, batch_dfs, client_df, output_dir):
    train_df.to_csv(os.path.join(output_dir, 'train_df.psv'), sep='|', index=False)
    for i, batch_df in enumerate(batch_dfs, 1):
        batch_df.to_csv(os.path.join(output_dir, f'batch_df_{i}.psv'), sep='|', index=False)
    client_df.to_csv(os.path.join(output_dir, 'client_df.psv'), sep='|', index=False)

