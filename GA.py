from processing import save_best_chromosome
from fitness import calc_fitness, create_population
from operators import crossover, mutation
from multiprocessing import Pool

def GA(data, iterations=10, population=10, mutation_rate=0.2):
    print("Data rows: ", len(data));   # 93707
    
    num_features = len(data.columns) - 1;   # removing labels column
    print("Number of features: ", num_features);    # 340

    # termination condition
    print("Iterations: ", iterations);

    # creating population
    chromosomes = create_population(population, num_features);
    print("Population: ", population);

    global_best = {"chromosome": [], "fitness": 0};
    local_bests = [];
    # feature selection
    for i in range(iterations):        
        data_df = data.copy();
        # reset indexes
        data_df.reset_index(drop=True, inplace=True);

        
        # calculating fitness using multiprocessing
        with Pool() as p:
            result = p.starmap(calc_fitness, [(data_df, chromosome) for chromosome in chromosomes])
        
        print("Iteration: ", i+1);
        print("Fitnesses: ", result);

        # sorting chromosomes based on fitness
        result, chromosomes = zip(*sorted(zip(result, chromosomes), reverse=True))
        result = list(result)
        chromosomes = list(chromosomes)

        local_bests.append(result[0])     
           
        # update global best
        if(result[0] > global_best["fitness"]):
            global_best["chromosome"] = chromosomes[0];
            global_best["fitness"] = result[0];
        
        print("Global best: ", global_best);
        
        # rank selection, select top 50% chromosomes
        chromosomes = chromosomes[:population//2];
        parents_count = len(chromosomes);
        
        # crossover
        crossover(chromosomes, parents_count, population);
        
        # mutation
        mutation(chromosomes, parents_count, mutation_rate);
        
    print("Global best: ", global_best);
    # save the best chromosome
    save_best_chromosome(global_best["chromosome"], global_best["fitness"], iterations, population, mutation_rate);
    
    return global_best, local_bests;