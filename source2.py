import random
from preproc import read_csv_files
from fitness import calc_fitness, create_population


# filtering data (columns 298 - 637)
data_df = read_csv_files("data");
print("Data rows: ", len(data_df));   # 47319

num_features = len(data_df.columns) - 1;   # removing labels column
print("Number of features: ", num_features);    # 340

# creating population
pop = 100;
chromosomes = create_population(pop, num_features);

# termination condition
iterations = random.randint(150, 200);

# feature selection
# for i in range(iterations):
    # calculating fitness
# -----
result = [];
for i, chromosome in enumerate(chromosomes):
    result.append(calc_fitness(data_df, chromosome, i, pop));
print("Fitnesses: ", result);

# selection
# sorting chromosomes based on fitness
result, chromosomes = zip(*sorted(zip(result, chromosomes), reverse=True))
result = list(result)
chromosomes = list(chromosomes)

# chromosomes = chromosomes[:80];    # selecting top 80 chromosomes
# -----

print ("Top 10 chromosomes: \n", chromosomes[:10], result[:10]);
