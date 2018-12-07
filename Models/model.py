import random

import numpy as np
import matplotlib.pyplot as pl

UPDATE_EPSILON = 0.1
INITIAL_FIRE_PROB = 0.01
WEIGHT_INIT = 0.2
EDGE_PROB = 0.8
THRESHOLD = 0.1
SAVE_HISTOGRAM = False

random.seed(0)

class Neuron:
    def __init__(self):
        self.fire_next = False
        self.fire_now = random.random() < INITIAL_FIRE_PROB
        self.fire_prev = False
        self.preds = []

    def calc_fire(self):
        fire_prob = sum(syn.weight * syn.start.fire_now for syn in self.preds)
        self.fire_next = random.random() < fire_prob

    def update(self):
        self.fire_prev = self.fire_now
        self.fire_now = self.fire_next
        self.fire_next = False

    def normalize(self):
        sum_weights = sum(syn.weight for syn in self.preds)
        for syn in self.preds:
            syn.weight /= sum_weights


class Synapse:
    def __init__(self, s, t, w):
        self.start = s
        self.end = t
        self.weight = w
        self.updated = False

    def update_weight(self):
        if self.start.fire_prev and self.end.fire_now:
            self.weight *= (1 + UPDATE_EPSILON)
            self.updated = True
        else:
            self.updated = False

class HebbianNet:
    def __init__(self, n):
        self.n = n
        self.t = 0

        self.neurons = [Neuron() for _ in range(n)]

        self.synapses = []
        for i in range(n):
            for j in range(i+1):
                s, t = i, j
                if random.random() < 0.5:
                    s, t = t, s
                if random.random() < EDGE_PROB:
                    self.synapses.append(
                        Synapse(
                            self.neurons[s],
                            self.neurons[t],
                            WEIGHT_INIT
                        )
                    )

        for syn in self.synapses:
            syn.end.preds.append(syn)

    def update(self):
        self.t += 1
        for neuron in self.neurons:
            neuron.calc_fire()
        for neuron in self.neurons:
            neuron.update()
        for syn in self.synapses:
            syn.update_weight()
        for neuron in self.neurons:
            neuron.normalize()

    def get_updated_synapses(self):
        return set(i for i,synapse in enumerate(self.synapses) if synapse.updated)

    def get_firing_neurons(self):
        return set(i for i,neuron in enumerate(self.neurons) if neuron.fire_now)


def is_same(i,old,new,tag):
    try:
        ratio = len(old & new) / len(old | new)
        print ("{}: {} percent similar at {}th iteration".format(tag, ratio*100, i))
        return True if ratio == 1 else False
    except:
        # zero division
        return True

def save_weight_distribution(weights):
    fig = pl.hist(weights, normed=0, range=(.0042,.0225))
    pl.title('weights')
    pl.xlabel("weights")
    pl.ylabel("freq")
    pl.savefig("./img/w_{}.png".format(i))

if __name__ == '__main__':
    net = HebbianNet(300)

    prev_weights = [syn.weight for syn in net.synapses] # not used
    prev_fired = set()
    prev_updated = set()

    for i in range(3000):

        net.update()

        cur_weights = [syn.weight for syn in net.synapses]
        if SAVE_HISTOGRAM: save_weight_distribution(cur_weights)

        cur_fired = net.get_firing_neurons()
        cur_updated = net.get_updated_synapses()

        if is_same(i,prev_fired,cur_fired, "fired neurons"): break
        if is_same(i,prev_updated,cur_updated, "updated synapses"): break

        prev_weights = cur_weights # not used
        prev_fired = cur_fired
        prev_updated = cur_updated

        l = sum(syn.weight > THRESHOLD for syn in net.synapses)
        if l > 0: print("Edges over threshold: {} of {} ({}th iteration)".format(l, len(net.synapses),i))
