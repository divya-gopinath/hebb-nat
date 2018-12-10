import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

N_NEURONS = 102  # Number of nodes in the graph
EPSILON = 0.01  # Epsilon value used for update
DISP_THRESH = 0.0  # Does nothing
SIM_ITERS = 120  # Number of iterations to run Hebb for
SIM_DISPLAY = 4  # Number of times we stop the simulation and plot the graph
Y_DIM = 2  # Display graphs in a SIM_DISPLAY by SIM_DISPLAY/Y_DIM grid (make sure it divides properly)
DISPLAY = False  # Display graphs

np.random.seed(1)


def initialize_adj(n):
    adj = np.random.random((n, n))
    adj[adj < 0] = 0
    np.fill_diagonal(adj, 0)
    adj[1, 0] = 0  # 0 does not point to 1 directly
    adj[0, :] = 0  # Nothing points to 0
    adj[:, 1] = 0  # 1 points to nothing
    adj = (adj.T / np.sum(adj, axis=1)).T
    adj[np.isnan(adj)] = 0
    return adj


class FastNet:
    def __init__(self, n_neurons):
        self.n = n_neurons  # Number of nodes in the graph
        # Adjacency matrix where self.adj[i,j] is the extra probability that i fires if j fired last time
        self.adj = initialize_adj(self.n)
        self.firing = np.random.choice([0, 1], self.n, p=[0.9, 0.1])  # self.firing[i] is 1 if i fired this round
        self.fired = np.random.choice([0, 1], self.n, p=[0.9, 0.1])  # self.fired[i] is 1 if i fired last round
        self.gr = nx.Graph()

    def force_fire(self, indices):
        for i in indices:
            self.firing[i] = 1

    def evolve(self):
        self.fired = self.firing

        # the dot product will have terms fire_probs[i] = sum_j adj[i,j]*firing[j] = prob that i fires this round
        fire_probs = np.dot(self.adj, self.firing)
        self.firing = (fire_probs > np.random.random(fire_probs.shape)).astype(int)

        # If j fired last time, and i fired this time, then adj[i,j] should be increased by a factor of EPSILON
        self.adj *= 1 + (EPSILON * np.outer(self.firing, self.fired))

        # normalize so the probability of a firing is at most 1
        self.adj = (self.adj.T / np.sum(self.adj, axis=1)).T
        self.adj[np.isnan(self.adj)] = 0
    def display(self, axarr, ind):
        plt.sca(axarr.item(ind))
        if False:
            plt.imshow(self.adj)
        if True:
            rows, cols = np.where(self.adj > DISP_THRESH)
            edges = zip(rows.tolist(), cols.tolist())
            cmap = list(map({
                        0: 'red',  # Did not fire
                        1: 'blue',  # Firing now
                        2: 'green',  # Fired last time
                        3: 'purple'  # Fired both this time and last time
                        }.get,
                        (self.firing + 2*self.fired).astype(int)))
            for edge in edges:
                self.gr.add_edge(*edge, weight=self.adj[edge])
            pos = nx.circular_layout(self.gr)
            plt.axis('off')
            nx.draw_networkx_nodes(self.gr, pos, node_color=cmap, with_labels=True)
            nx.draw_networkx_edges(self.gr, pos, width=[3*d['weight'] for (u, v, d) in self.gr.edges(data=True)])
            nx.draw_networkx_labels(self.gr, pos)


if False and __name__ == '__main__':

    net = FastNet(N_NEURONS)
    if DISPLAY:
        fig, axarr = plt.subplots(SIM_DISPLAY//Y_DIM, Y_DIM)
    for t in range(120):
        net.force_fire([0, 1])
        net.evolve()
    win, loss = 0, 0
    for t in range(100):
        net.force_fire([0])
        net.evolve()
        if net.fired[1]==1:
            win+=1
        else:
            loss+=1
    if DISPLAY:
        plt.show()

if True or __name__ == '__main__':
    accuracies = []
    for N in range(30,150,10):
        acc_loc=[]
        for T in range(500):
            net = FastNet(N)
            if DISPLAY:
                fig, axarr = plt.subplots(SIM_DISPLAY//Y_DIM, Y_DIM)
            for t in range(T):
                net.force_fire([0, 1])
                if DISPLAY and t % (SIM_ITERS//SIM_DISPLAY) == 0:
                    ind_disp = t // (SIM_ITERS//SIM_DISPLAY)
                    x_ind = ind_disp % (SIM_DISPLAY//Y_DIM)
                    y_ind = ind_disp // (SIM_DISPLAY//Y_DIM)
                    net.display(axarr, (x_ind, y_ind))
                net.evolve()
            win, loss = 0, 0
            for t in range(100):
                net.force_fire([0])
                if DISPLAY and t % (SIM_ITERS//SIM_DISPLAY) == 0:
                    ind_disp = t // (SIM_ITERS//SIM_DISPLAY)
                    x_ind = ind_disp % (SIM_DISPLAY//Y_DIM)
                    y_ind = ind_disp // (SIM_DISPLAY//Y_DIM)
                    net.display(axarr, (x_ind, y_ind))
                net.evolve()
                if net.fired[1]==1:
                    win+=1
                else:
                    loss+=1
            acc_loc.append(win/(win+loss))
            if DISPLAY:
                plt.show()
        accuracies.append(acc_loc)
