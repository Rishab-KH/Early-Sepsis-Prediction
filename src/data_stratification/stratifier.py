import sys
import os
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
    
    return train_df, batch_df, client_df


if __name__ == "__main__":
    # Check if the folder name is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_name>")
        sys.exit(1)
    
    # Get the folder name from the command-line arguments
    folder_name = sys.argv[1]
    
    # Read and concatenate all PSV files
    combined_dataframe = read_and_concat_psv_files(folder_name)
    
    # Create the count DataFrame
    count_dataframe = create_count_df(combined_dataframe)
    
    # Print the count DataFrame (or perform any other operations you need)
    print(count_dataframe)
    
    pid_groups = find_pids_crossing_threshold(count_dataframe, thresholds=[0.7, 0.2, 0.1])

     # Print the PIDs that fall within each threshold group
    for threshold, pids in pid_groups.items():
        print(f"Number of PIDs that fall within {threshold*100}% of total count of 1s: {len(pids)}")

    # Create train_df, batch_df, and client_df
    train_df, batch_df, client_df = create_dataframes_for_groups(combined_dataframe, pid_groups)
    
    # Print the shapes of the resulting dataframes
    print(f"train_df shape: {train_df.shape}")
    print(f"batch_df shape: {batch_df.shape}")
    print(f"client_df shape: {client_df.shape}")
    

