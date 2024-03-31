from GA import GA
from model import train_model
from processing import get_data, plot_graph
import matplotlib.pyplot as plt

def run():
    data = get_data();

    # run the GA with different parameters
    gb1, lbs1 = GA(data, iterations=10, population=10, mutation_rate=0.1);
    gb2, lbs2 = GA(data, iterations=10, population=10, mutation_rate=0.2);
    gb3, lbs3 = GA(data, iterations=10, population=14, mutation_rate=0.1);
    gb4, lbs4 = GA(data, iterations=10, population=14, mutation_rate=0.2);

    # zip the gb with the lbs
    gb1["local_bests"] = lbs1;
    gb2["local_bests"] = lbs2;
    gb3["local_bests"] = lbs3;
    gb4["local_bests"] = lbs4;

    # find the best global best chromosome
    global_bests = [gb1, gb2, gb3, gb4];    
    
    global_bests.sort(key=lambda x: x["fitness"], reverse=True);
    global_best = global_bests[0];

    local_bests = global_best["local_bests"];

    # plot the graph
    plot_graph(local_bests);

    # train the model with the best chromosome
    chromosome = global_best["chromosome"];
    train_model(data, chromosome);
    
if __name__ == "__main__":
    run();