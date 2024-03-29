import os
import pandas as pd

def combine_data(data):
    print("Combining data...");
    combined_data = pd.DataFrame()
    for key in data:
        combined_data = pd.concat([combined_data, data[key]])
    return combined_data

def read_csv_files(directory):
    # filtering data (columns 298 - 637)
    print("Reading data...");
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename), usecols=range(298, 638))    # 298 - 637 columns
            file_parts = filename.split('-')
            if file_parts[2] == '03':
                df['label'] = 0
            elif file_parts[2] == '04':
                df['label'] = 1
            data[filename] = df
    return combine_data(data)

def get_data(folder="data"):
    data = read_csv_files(folder);
    return data;

def save_best_chromosome(chromosome, fitness, iterations, population, mutation_rate):
    with open("best_chromosome.txt", "a") as f:
        f.write("Iteration: ");
        f.write(str(iterations));
        f.write(",\t");
        f.write("Population: ");
        f.write(str(population));
        f.write(",\t");
        f.write("Mutation Rate: ");
        f.write(str(mutation_rate));
        f.write("\n");
        f.write(str(chromosome));
        f.write("\n");
        f.write("Number of features: ");
        f.write(str(len(chromosome)));
        f.write("Fitness: ");
        f.write(str(fitness));
        f.write("\n\n");