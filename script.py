from GA import GA
from processing import get_data

data = get_data();
GA(data, iterations=10, population=10, mutation_rate=0.1);
GA(data, iterations=10, population=10, mutation_rate=0.2);
GA(data, iterations=10, population=14, mutation_rate=0.1);
GA(data, iterations=10, population=14, mutation_rate=0.2);