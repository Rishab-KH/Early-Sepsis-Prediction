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



if __name__ == "__main__":
    # Check if the folder name is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_name>")
        sys.exit(1)
    
    # Get the folder name from the command-line arguments
    folder_name = sys.argv[1]
    
    # Read and concatenate all PSV files
    combined_dataframe = read_and_concat_psv_files(folder_name)

    print(combined_dataframe)