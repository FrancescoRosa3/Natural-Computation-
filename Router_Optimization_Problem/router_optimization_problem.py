import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# class representing a router
class Router:

    def __init__(self, x, y, r_sig):
        self.pos = [x, y]
        self.r_sig = r_sig

    def compute_distance(self, router):
        return distance.euclidean(self.pos, router.pos)

class Clonalg:

    def __init__(self, max_it, n1, n2, p, beta, evaluation, filename_client, r_sig, c_w, c_ap, source_x, source_y):
        # algorithm parameters
        self.max_it = max_it
        self.N = n1
        self.n1 = n1
        self.n2 = n2
        self.beta = beta
        self.p = p
        self.nc = int(beta * self.N)  # Number of clones to be generate for each antibody
        self.evaluation = evaluation

        # support attribute
        self.filename_client = filename_client
        # optimization function parameters
        self.c_w = c_w
        self.c_ap = c_ap

        # Set clients
        self.clients_list = self.create_client_list(self.filename_client)
        self.num_clients = len(self.clients_list)
        print("Number of clients: ", self.num_clients)
        self.compute_position_limits()

        # Set Source
        self.source = Router(source_x, source_y, r_sig)

        # Set population
        self.population = self.create_population(self.x_min, self.x_max, self.y_min, self.y_max, r_sig, self.N, self.num_clients)
        #self.population = self.create_population(self.x_min, self.x_max, self.y_min, self.y_max, r_sig, self.N, 10)
        self.results = np.zeros((3, self.max_it))  # 3: max, min and average


    def create_client_list(self, filename):
        clients = []
        with open(filename) as file:
            for row in file:
                pos_str = row.split()
                clients.append([float(pos_str[0]), float(pos_str[1])])
        return clients

    def compute_position_limits(self):
        self.x_min = min(self.clients_list[:][0])
        self.x_max = max(self.clients_list[:][0])
        self.y_min = min(self.clients_list[:][1])
        self.y_max = max(self.clients_list[:][1])
        print("Limit positions: X_min, X_max, Y_min, Y_max: ", self.x_min, self.x_max, self.y_min, self.y_max)

    def create_population(self, x_min, x_max, y_min, y_max, r_sig, population_dim, max_number_router=10):
        population = []
        for i in range(population_dim):
            conf = []
            num_router = np.random.uniform(low=1, high=max_number_router)
            for j in range(int(num_router)):
                conf.append(self.create_router(x_min, x_max, y_min, y_max, r_sig))
            population.append(conf)
        return population

    def create_router(self, x_min, x_max, y_min, y_max, r_sig):
        x = np.random.uniform(low=x_min, high=x_max)
        y = np.random.uniform(low=y_min, high=y_max)
        return Router(x, y, r_sig)

    # Methods for the optimization procedure
    def normalize(self, d):
        dmax = np.amax(d)
        return np.apply_along_axis(lambda di: di / dmax, 0, d)

    def compute_affinity(self, population):
        affinities = []
        for configuration in population:
            configuration.append(self.source)
            affinities.append(self.evaluation(self.c_ap, self.c_w, configuration))
            configuration.pop(len(configuration)-1)
        return affinities

    def mutate_coordinate(self, coordinate, delta, lower_bound, upper_bound):
        pb = np.random.choice([0, 1])
        if pb == 1:
            coordinate = math.fmod(coordinate + delta, upper_bound)
        else:
            coordinate = math.fmod(coordinate - delta, abs(lower_bound))
        
        return coordinate

    def mutation(self, clones_to_mutate, affinities):
        for (i, config) in enumerate(clones_to_mutate):
            print(i, len(config))
            # compute mutation rate
            alpha = np.exp(-self.p * affinities[i])
            pb = np.random.uniform(0, 1)
            if (pb > alpha):
                continue
            
            # mutate coordinate for each router in config
            for router in config:
                delta =  router.pos[0] * alpha * np.random.choice([0.01, 100])
                router.pos[0] = self.mutate_coordinate(router.pos[0], delta, self.x_min, self.x_max)
                delta = router.pos[1] * alpha * np.random.choice([0.01, 100])
                router.pos[1] = self.mutate_coordinate(router.pos[1], delta, self.y_min, self.y_max)

            # add or remove router
            pb = np.random.choice([0, 1])
            if pb == 1:
                # add router
                config.append(self.create_router(self.x_min, self.x_max, self.y_min, self.y_max, config[0].r_sig))
            else:
                # remove router
                if(len(config) > 1):
                    index = int(np.random.uniform(len(config)-1))
                    config.pop(index)
        return clones_to_mutate


    def clonalg_opt(self):
        t = 1
        while t < self.max_it:
            print("Iteration ", t)
            # compute affinity
            self.affinities = self.compute_affinity(self.population)
            # store results
            self.results[0][t] = np.amin(self.affinities)
            self.results[1][t] = np.amax(self.affinities)
            self.results[2][t] = np.average(self.affinities)
            print("Best ", self.results[0][t], "\nAverage ",  self.results[2][t], "\nWorst ",  self.results[1][t])
            # selection

            # clone

            # mutation
            self.affinities = self.normalize(self.affinities)
            self.population = self.mutation(self.population, self.affinities)
            # replace

            t = t + 1

    def graph(self):
        plt.plot(self.results[0], label="Best evaluation")
        plt.plot(self.results[1], label="Worst evaluation")
        plt.plot(self.results[2], label="Average")
        plt.legend(loc='best')
        plt.show()

    def result(self):
        b = self.fitness.argmax(axis=0)
        return (self.population[b], self.fitness[b] * (-1))

def compute_total_wire_length(configuration):
    # compute the graph
    graph = []
    #print("Number of routers: ", len(configuration))
    for i in range(len(configuration)):
        connection_node = [0 for k in range(0, i)]
        for j in range(i, len(configuration)):
            connection_node.append(configuration[i].compute_distance(configuration[j]))
        graph.append(connection_node)
    #print(graph)
    # compute the minimum spanning tree
    mst = minimum_spanning_tree(graph).toarray().astype(float)
    #print(mst)
    # compute the total distance
    distance = 0
    for i in range(len(mst)):
        for j in range(len(mst[0])):
            distance = distance + mst[i][j]
    #print("Total distance: ", distance)
    return distance


def wires_cost(c_ap, c_w, configuration):
    # C = C_AP + C_W = n_ap * C_ap + C_w * Sum(L_ij)
    # first term
    n_ap = len(configuration)
    C_AP = n_ap * c_ap
    # second term
    total_wire_length = compute_total_wire_length(configuration)
    C_W = c_w * total_wire_length
    return (C_AP + C_W)


def main():
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))

    np.random.seed(1)
    clonalg = Clonalg(max_it=10, n1=50, n2=0, p=5, beta=0.2, evaluation=wires_cost, 
                     filename_client=dir_path+"/coord200.txt", r_sig = 30, c_w=1, c_ap=1,
                     source_x=0, source_y=0)
    clonalg.clonalg_opt()
    #print(clonalg.result())
    clonalg.graph()


if __name__ == '__main__':
    main()
