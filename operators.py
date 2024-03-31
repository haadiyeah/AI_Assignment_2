import random

def crossover(chromosomes, parents_count, population):
    for i in range(parents_count):
        parent1 = chromosomes[i]
        parent2 = chromosomes[i+1]
        
        random_index = random.randint(0, len(parent1)-1)
        child1 = parent1[:random_index] + parent2[random_index:]
        child2 = parent2[:random_index] + parent1[random_index:]
        
        chromosomes.append(child1)
        chromosomes.append(child2)
        
        i+=2    # next pair of parents excluding the ones just used
        
    # if chromosomes < population
    if len(chromosomes) < population:
        chromosomes.append(chromosomes[parents_count-1])    # append the last parent
    
def mutation(chromosomes, parents_count, mutation_rate):
    for i in range(parents_count, len(chromosomes)):
        if(random.random() < mutation_rate):
            random_index = random.randint(0, len(chromosomes[i])-1)
            chromosomes[i][random_index] = 1 - chromosomes[i][random_index]