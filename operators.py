import random

def crossover(chromosomes, parents_count):
    children = []
    for i in range(parents_count):
        parent1 = chromosomes[i]
        parent2 = chromosomes[i+1]
        
        random_index = random.randint(0, len(parent1)-1)
        child1 = parent1[:random_index] + parent2[random_index:]
        child2 = parent2[:random_index] + parent1[random_index:]
        
        chromosomes.append(child1)
        chromosomes.append(child2)
        
        i+=2    # next pair of parents excluding the ones just used
        
    if (parents_count % 2 != 0):    # if the parents_count is odd
        chromosomes.append(chromosomes[parents_count-1])    # append the last parent
    else:
        chromosomes.append(chromosomes[0])    # append the first parent (best chromosome)
        
    
def mutation(chromosomes, mutation_rate, parents_count):
    for i in range(parents_count, len(chromosomes)):
        if(random.random() < mutation_rate):
            random_index = random.randint(0, len(chromosomes[i])-1)
            chromosomes[i][random_index] = 1 - chromosomes[i][random_index]

        