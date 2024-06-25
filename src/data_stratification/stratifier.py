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
    
    return count_df

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

    total_count_of_1 = count_dataframe['count_of_1'].sum()
    
    print(total_count_of_1)
    