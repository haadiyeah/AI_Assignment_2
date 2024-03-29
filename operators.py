import random

def crossover(chromosomes, population):
    children = []
    while len(children) + len(chromosomes) < population:
        # select 2 random chromosomes
        parent1 = random.choice(chromosomes)
        parent2 = random.choice(chromosomes)
        
        # select random crossover point
        crossover_point = random.randint(0, len(parent1)-1)
        
        # create new chromosome
        child = parent1[:crossover_point] + parent2[crossover_point:]
        children.append(child)
    
    chromosomes.extend(children)
    
def mutation(chromosomes, mutation_rate, parents_count):
    for i in range(parents_count, len(chromosomes)):
        if(random.random() < mutation_rate):
            random_index = random.randint(0, len(chromosomes[i])-1)
            chromosomes[i][random_index] = 1 - chromosomes[i][random_index]

        