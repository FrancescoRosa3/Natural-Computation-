import matplotlib.pyplot as plt
import numpy as np
import math
import networkx as nx

from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import copy 
import os
# class representing a router
class Router:

    def __init__(self, x, y, r_sig):
        self.pos = [x, y]
        self.r_sig = r_sig

    def compute_distance(self, router):
        return distance.euclidean(self.pos, router.pos)

class Clonalg:

    def __init__(self, max_it, n1, n2, n3, p, beta, evaluation, filename_client, r_sig, c_w, c_ap, source_x, source_y, seed):
        # algorithm parameters
        self.max_it = max_it
        self.N = n1
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.beta = beta
        self.p = p
        self.nc = int(beta * self.N)  # Number of clones to be generate for each antibody
        self.evaluation = evaluation

        # support attribute
        self.filename_client = filename_client
        # optimization function parameters
        self.c_w = c_w
        self.c_ap = c_ap
        self.r_sig = r_sig

        # Set clients
        self.clients_list = self.create_client_list(self.filename_client)
        self.num_clients = len(self.clients_list)
        print("Number of clients: ", self.num_clients)
        self.compute_position_limits()

        # Set Source
        self.source = Router(source_x, source_y, r_sig)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.dir_name = str(dir_path+"/../screens/best_conf_screens_"+str(seed)+"_"+str(max_it)
                                                                                    +"_"+str(n1)
                                                                                    +"_"+str(n2)
                                                                                    +"_"+str(n3)
                                                                                    +"_"+str(p)
                                                                                    +"_"+str(beta))
        os.mkdir(self.dir_name)
        # Set population
        # self.population = self.create_population(self.x_min, self.x_max, self.y_min, self.y_max, r_sig, self.N, self.num_clients)
        self.population = self.create_population(self.x_min, self.x_max, self.y_min, self.y_max, self.r_sig, self.N)
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
        dmin = np.amin(d)
        dmax = np.amax(d)
        return np.apply_along_axis(lambda di: 1-((di - dmin)/abs(dmax-dmin)), 0, d)

    def compute_affinity(self, population, population_affinity=None, clonated_flags=None):
        affinities = []
        for (i, configuration) in enumerate(population):
            # add source
            configuration.append(self.source)
            if(clonated_flags is not None and clonated_flags[i] == True):
                # clones that were mutated
                affinities.append(self.evaluation(self.c_ap, self.c_w, configuration, self.clients_list))
            elif clonated_flags is not None and clonated_flags[i] == False and population_affinity is not None:
                # clones that were not mutated
                affinities.append(population_affinity[i])
            else:
                # affinity for the beginnig population
                affinities.append(self.evaluation(self.c_ap, self.c_w, configuration, self.clients_list))
            # remove source
            configuration.pop(len(configuration)-1)
        return affinities

    def select(self, population, affinities, n):

        affinities = np.array(affinities)
        # select the n highest affinities
        indices = affinities.argsort()[:n]
        temp_pop = []
        for index in indices:
            temp_pop.append(population[index])
        print(affinities[indices])
        
        return temp_pop, affinities[indices]

    def select_pop(self, population, affinities):
        print("SELECTED POPULATION: ")
        if self.n1 == self.N and self.n2 == 0:
            # print("Return the whole population")
            print(affinities)
            return  population, affinities

        return self.select(population, affinities, self.n1)

    def select_clones(self, population, affinities):
        # print("SELECTED CLONES: ")
        if self.n2 == 0:
            print("Replace the best clones in the population")
            # multimodal: select the best clone for each antibody and generate new population
            for i in range(0, self.N):
                best = np.argmin(affinities[i * self.nc: i * self.nc + self.nc])
                self.population[i] = population[best+(i * self.nc)]
                self.affinities[i] = affinities[best+(i * self.nc)]
            return self.population, self.affinities

        return self.select(population, affinities, self.n2)

    def clone(self, population_to_clone, affinities_to_clone, affinities_to_clone_norm):

        if self.n1 == self.N and self.n2 == 0:
            # print("Generating clones for each element")
            affinities_clones = np.zeros(len(population_to_clone) * self.nc)
            clones = []
            for i, config_to_clone in enumerate(population_to_clone):
                for j in range(i * self.nc, i * self.nc + self.nc):
                    clones.append(copy.deepcopy(config_to_clone))
                    affinities_clones[j] = affinities_to_clone[i]
            return clones, affinities_clones
        
        affinities_clones = []
        clones = []
        for i, config_to_clone in enumerate(population_to_clone):
            #print(i)
            clones_per_conf = self.nc * affinities_to_clone_norm[i]
            for j in range(int(clones_per_conf)):
                clones.append(copy.deepcopy(config_to_clone))
                affinities_clones.append(affinities_to_clone[i])
            # i += self.nc

        return clones, affinities_clones

    def mutate_coordinate(self, coordinate, delta, lower_bound, upper_bound):
        pb = np.random.choice([0, 1])
        if pb == 1:
            coordinate = math.fmod(coordinate + delta, upper_bound)
        else:
            coordinate = math.fmod(coordinate - delta, abs(lower_bound))
        return coordinate

    def mutation(self, clones_to_mutate, affinities):
        mutated_flags = [False for i in range(len(clones_to_mutate))]
        for (i, config) in enumerate(clones_to_mutate):
            # print(i, len(config))
            # compute mutation rate
            alpha = np.exp(-self.p * affinities[i])
            pb = np.random.uniform(0, 1)
            #print(alpha, affinities[i])
            if (pb > alpha):
                continue
            
            # mutate coordinate for each router in config
            mutated_flags[i] = True
            for router in config:
                delta =  router.pos[0] * alpha * np.random.choice([10, 100])
                router.pos[0] = self.mutate_coordinate(router.pos[0], delta, self.x_min, self.x_max)
                delta = router.pos[1] * alpha * np.random.choice([10, 100])
                router.pos[1] = self.mutate_coordinate(router.pos[1], delta, self.y_min, self.y_max)

            # add or remove router
            pb = np.random.choice([0, 1, 2])
            if pb == 1:
                # add router
                config.append(self.create_router(self.x_min, self.x_max, self.y_min, self.y_max, config[0].r_sig))
            elif pb == 0:
                # remove router
                if(len(config) > 1):
                    index = int(np.random.uniform(len(config)-1))
                    config.pop(index)
        return clones_to_mutate, mutated_flags

    def replace(self, clones, clones_affinities, population, affinities):
        if self.n2 == 0:
            return self.population
        
        # replace clones
        for i in reversed(range(-self.n2, 0)):
            population[i] = clones[i]
            affinities[i] = clones_affinities[i]

        # replace random generated population
        population[-self.n2-self.n3: -self.n2] = self.create_population(self.x_min, self.x_max, self.y_min, self.y_max, self.r_sig, self.n3)
        affinities[-self.n2-self.n3: -self.n2] = self.compute_affinity(population[-self.n2-self.n3: -self.n2])
        
        
        print("RANDOM POP: ")
        print(len(affinities[-self.n2-self.n3: -self.n2]))
        print(affinities[-self.n2-self.n3: -self.n2])
        

        self.population = population
        self.affinities = affinities.tolist()
             
    def clonalg_opt(self):
        self.t = 1
        while self.t < self.max_it:
            print("Iteration ", self.t)
            # compute affinity
            if(self.t==1):
                self.affinities = self.compute_affinity(self.population)
            # store results
            self.results[0][self.t] = np.amin(self.affinities) # best
            self.results[1][self.t] = np.amax(self.affinities) # worst
            self.results[2][self.t] = np.average(self.affinities)
            print("Best ", self.results[0][self.t], "\nAverage ",  self.results[2][self.t], "\nWorst ",  self.results[1][self.t])
            # selection
            population_select, affinities_select = self.select_pop(self.population, self.affinities)
            # clone
            affinities_select_norm = self.normalize(affinities_select)
            clones, affinities_clones = self.clone(population_select, affinities_select, affinities_select_norm)
            # mutation
            affinities_clones_norm = self.normalize(affinities_clones)
            clones_mutated, clonated_flags = self.mutation(clones, affinities_clones_norm)
            # clone affinities
            affinities_clones = self.compute_affinity(clones_mutated, affinities_clones, clonated_flags)
            # select best mutated clones
            selected_clones, selected_clones_affinity = self.select_clones(clones_mutated, affinities_clones)
            print("population length: " + str(len(self.population)))
            # replace
            self.replace(selected_clones, selected_clones_affinity, population_select, affinities_select)
            self.ap_graph()
            self.t = self.t + 1

    def result_graph(self):
        fig, ax = plt.subplots()
        plt.plot(self.results[0], label="Best evaluation")
        plt.plot(self.results[1], label="Worst evaluation")
        plt.plot(self.results[2], label="Average")
        plt.legend(loc='best')
        fig.savefig(self.dir_name+"/final_" + str(self.t) + ".png")
        # plt.show()
        plt.close()
    def get_best_conf(self):
        best_conf_index = self.affinities.index(min(self.affinities))
        return self.population[best_conf_index], min(self.affinities), best_conf_index

    def ap_graph(self, configuration = None):

        fig, ax = plt.subplots()
    
        for client in self.clients_list:
            plt.scatter(client[0], client[1], s=4, c='red')
        
        conf, affinity, best_index = self.get_best_conf()
        plt.title(label = "Affinity: "+str(affinity)+" #Router: "+str(len(conf)) + " Configuration index: " + str(best_index))
        for router in conf:
            plt.scatter(router.pos[0], router.pos[1], c='green')
            circle = plt.Circle((router.pos[0], router.pos[1]), router.r_sig, color='b', fill=False)
            ax = plt.gca()
            ax.add_patch(circle)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        fig.savefig(self.dir_name+"/best_conf_" + str(self.t) + ".png")
        plt.close()
        
    def result(self):
        b = self.fitness.argmin(axis=0)
        return (self.population[b], self.affinities[b] * (-1))

# compute the list of clients that are not covered
def clients_not_covered(router, clients):
    clients_not_covered_by_router = []
    # for each client
    for client in clients:
        # compute the distance between router and client
        try:
            client_router_distance = distance.euclidean(client, router.pos)
        except:
            print(router.pos, client)
        if client_router_distance > router.r_sig:
            # client not covered
            clients_not_covered_by_router.append(client)

    return clients_not_covered_by_router

def compute_coverage(configuration, clients):
    # compute the graph
    graph = []
    #print("Number of routers: ", len(configuration))
    number_of_clients = len(clients)
    
    #print("conf len: " + str(len(configuration)))
    for i in range(len(configuration)):
        # find the clients that are not covered
        if i != len(configuration)-1:
            clients = clients_not_covered(configuration[i], clients)
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
    # number of clientes
    number_clients_convered = number_of_clients - len(clients)
    return distance, number_clients_convered


def cost_function(c_ap, c_w, configuration, clients):
    # C = C_AP + C_W = n_ap * C_ap + C_w * Sum(L_ij)
    # first term
    n_ap = len(configuration)
    C_AP = n_ap * c_ap
    # second term
    total_wire_length, number_clients_convered = compute_coverage(configuration, clients)
    C_W = c_w * total_wire_length
    # print("number_clients_convered: " +  str(number_clients_convered))
    C = C_AP + C_W - 10*number_clients_convered
    # print("cost function: " + str(C_AP) + " + " + str(C_W) + " - " + str(number_clients_convered) + " = " + str(C))
    return (C)


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    seed = 10
    for i in range(10):
        seed = seed + 100
        np.random.seed(seed)
        clonalg = Clonalg(max_it=20, n1=50, n2=0, n3=0, p=0.7, beta=0.3, evaluation=cost_function, 
                        filename_client=dir_path+"/coord200.txt", r_sig = 100, c_w=0.01, c_ap=5,
                        source_x=0, source_y=0, seed=seed)

        clonalg.clonalg_opt()
        #print(clonalg.result())
        clonalg.result_graph()
    


if __name__ == '__main__':
    main()
