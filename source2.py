import random
from preproc import read_csv_files
from fitness import calc_fitness, create_population

# filtering data (columns 298 - 637)
print("Reading data...");
data = read_csv_files("data");
print("Data rows: ", len(data));   # 47319

num_features = len(data.columns) - 1;   # removing labels column
print("Number of features: ", num_features);    # 340

# creating population
pop = 100;
chromosomes = create_population(pop, num_features);
print("Population: ", pop);

# termination condition
iterations = random.randint(150, 200);
print("Iterations: ", iterations);

# feature selection
for i in range(iterations):
    data_df = data.copy();
    # reset indexes
    data_df.reset_index(drop=True, inplace=True);
    # calculating fitness
    # -----
    result = [];
    for j, chromosome in enumerate(chromosomes):
        print("it: ", i+1, "\nch: ", j+1)
        data_df, fitness = calc_fitness(data_df, chromosome, j, pop);
        result.append(fitness);
    
    print("Fitnesses: ", result);

    # selection
    # sorting chromosomes based on fitness
    result, chromosomes = zip(*sorted(zip(result, chromosomes), reverse=True))
    result = list(result)
    chromosomes = list(chromosomes)

    # chromosomes = chromosomes[:80];    # selecting top 80 chromosomes
    # -----

print ("Top 10 chromosomes: \n", chromosomes[:10], result[:10]);
