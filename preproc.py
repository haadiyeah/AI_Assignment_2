import os
import pandas as pd

def combine_data(data):
    combined_data = pd.DataFrame()
    for key in data:
        combined_data = pd.concat([combined_data, data[key]])
    return combined_data

def read_csv_files(directory):
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename), usecols=range(298, 638))    # 298 - 637 columns
            file_parts = filename.split('-')
            if file_parts[2] == '03':
                df['label'] = 1
            elif file_parts[2] == '04':
                df['label'] = 2
            data[filename] = df
    return combine_data(data)