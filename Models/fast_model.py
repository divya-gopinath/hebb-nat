import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

N_NEURONS = 100  # Number of nodes in the graph
EPSILON = 0.01  # Epsilon value used for update
DISP_THRESH = 0  # Does nothing
SIM_ITERS = 120  # Number of iterations to run Hebb for
SIM_DISPLAY = 12  # Number of times we stop the simulation and plot the graph
Y_DIM = 3  # Display graphs in a SIM_DISPLAY by SIM_DISPLAY/Y_DIM grid (make sure it divides properly)


class FastNet:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.adj = np.random.random((self.n, self.n))
        self.firing = np.random.randint(0, 2, self.n)
        self.fired = np.random.randint(0, 2, self.n)
        self.gr = nx.Graph()

    def force_fire(self, indices):
        self.firing[indices] = 1

    def evolve(self):
        self.fired = self.firing
        fire_probs = np.dot(self.adj, self.firing)
        self.firing = fire_probs > np.random.random(fire_probs.shape)
        self.adj *= 1 + (EPSILON * np.outer(self.firing, self.fired))
        self.adj /= np.sum(self.adj, axis=1)

    def display(self, axarr, ind):
        plt.sca(axarr.item(ind))
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


if __name__ == '__main__':
    net = FastNet(N_NEURONS)
    fig, axarr = plt.subplots(SIM_DISPLAY//Y_DIM, Y_DIM)
    for t in range(SIM_ITERS):
        net.evolve()
        if t % (SIM_ITERS//SIM_DISPLAY) == 0:
            ind_disp = t // (SIM_ITERS//SIM_DISPLAY)
            x_ind = ind_disp % (SIM_DISPLAY//Y_DIM)
            y_ind = ind_disp // (SIM_DISPLAY//Y_DIM)
            net.display(axarr, (x_ind, y_ind))
    plt.show()
