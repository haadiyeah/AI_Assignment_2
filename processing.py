import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            
            if file_parts[2] != '03' and file_parts[2] != '04':
                continue;
            
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
        f.write(", ");
        f.write("Population: ");
        f.write(str(population));
        f.write(", ");
        f.write("Mutation Rate: ");
        f.write(str(mutation_rate));
        f.write("\n");
        f.write(str(chromosome));
        f.write("\n");
        f.write("Number of features: ");
        f.write(str(chromosome.count(1)))
        f.write("\n");
        f.write("Fitness: ");
        f.write(str(fitness));
        f.write("\n\n");
        
def plot_graph(local_bests):
    # plot the local best fitnesses
    plt.plot(range(1, len(local_bests) + 1), local_bests, color='red')  # line color
    plt.scatter(range(1, len(local_bests) + 1), local_bests, color='black')  # point color
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness of the best chromosome in each generation")
    plt.xticks(range(1, len(local_bests) + 1))  # Set the ticks on the x-axis

    # Set the ticks on the y-axis
    y_ticks = np.arange(0.9, 0.93, 0.005)
    plt.yticks(y_ticks)
    plt.ylim([0.9, 0.93])  # Set the range of the y-axis

    plt.show()