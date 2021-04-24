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
    def __init__(self, max_it, n1, n2, p, beta, evaluation, filename_client, r_sig, c_w, c_ap):
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

        # Set population
        #self.population = self.create_population(self.x_min, self.x_max, self.y_min, self.y_max, r_sig, self.N, self.num_clients)
        self.population = self.create_population(self.x_min, self.x_max, self.y_min, self.y_max, r_sig, self.N, 10)
        self.results = np.zeros((3, self.max_it))  # 3: max, min and average

        evaluation([self.c_ap, self.c_w, self.population[0]])

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
            num_router = np.random.choice(max_number_router)
            for j in range(num_router):
                x = np.random.uniform(low=x_min, high=x_max)
                y = np.random.uniform(low=y_min, high=y_max)
                router = Router(x, y, r_sig)
                conf.append(router)
            population.append(conf)
        return population

        

    def select(self, population, fitness):
        # if n1 is equal N, then no selection is required
        if self.N == self.n1:
            return population, fitness

        indexes = fitness.argsort()[-self.n1::][::-1]
        # select the n1 highest fitness
        return population[indexes], fitness[indexes]

    def select_clones(self, population, fitness):
        # multimodal: select the best clone for each antibody and generate new population
        for i in range(0, self.N, self.nc):
            best = np.argmax(fitness[i * self.nc: i * self.nc + self.nc])
            self.population[i] = population[best]

    def clone(self, antibodies, fitness):
        fitness_clones = np.zeros(len(antibodies) * self.nc)
        clones = []
        for i, antibody in enumerate(antibodies):
            for j in range(i * self.nc, i * self.nc + self.nc):
                clones.append(antibody)
                fitness_clones[j] = fitness[i]
            i += self.nc

        return clones, fitness_clones

    def normalize(self, d):
        dmax = np.amax(d)
        return np.apply_along_axis(lambda di: di / dmax, 0, d)

    def mutation(self, clones, fitness):
        for (i, clone) in enumerate(clones):
            alpha = np.exp(-self.p * fitness[i])
            pb = np.random.uniform(0, 1)
            if (pb > alpha):
                continue

            delta = clones[i][0] * alpha * np.random.choice([0.01, 1])
            clones[i][0] = math.fmod(clones[i][0] + delta, self.limits[1])
            delta = clones[i][1] * alpha * np.random.choice([0.01, 1])
            clones[i][1] = math.fmod(clones[i][1] + delta, self.limits[1])
        return clones

    def replace(self):
        if self.n2 == 0:
            return self.population

    def clonalg_opt(self):
        t = 1
        while t < self.max_it:
            self.fitness = np.apply_along_axis(self.evaluation, 1, self.population)

            self.results[0][t] = np.amax(self.fitness)
            self.results[1][t] = np.amin(self.fitness)
            self.results[2][t] = np.average(self.fitness)

            population_select, fitness_select = self.select(self.population, self.fitness)
            clones, fitness_clones = self.clone(population_select, fitness_select)
            fitness_clones_normalized = self.normalize(fitness_clones)
            clones_mutated = self.mutation(clones, fitness_clones_normalized)
            fitness_clones = np.apply_along_axis(self.evaluation, 1, clones_mutated)
            self.select_clones(clones_mutated, fitness_clones)
            self.replace()
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

def compute_total_distance(configuration):
    # compute the graph
    graph = []
    print("Number of routers: ", len(configuration))
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
    print("Total distance: ", distance)
    return distance


def wires_cost(args):
    configuration = args[2]
    # C = C_AP + C_W = n_ap * C_ap + C_w * Sum(L_ij)
    # first term
    n_ap = len(configuration)
    c_ap = args[0]
    C_AP = n_ap * c_ap
    # second term
    c_w = args[1]
    total_distance = compute_total_distance(configuration)
    C_W = c_w * total_distance
    return (C_AP + C_W) * (-1)


def main():
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))

    np.random.seed(1)
    clonalg = Clonalg(max_it=50, n1=50, n2=0, p=5, beta=0.2, evaluation=wires_cost, filename_client=dir_path+"/coord200.txt", r_sig = 30, c_w=1, c_ap=1)
    #clonalg.clonalg_opt()
    #print(clonalg.result())
    #clonalg.graph()


if __name__ == '__main__':
    main()
