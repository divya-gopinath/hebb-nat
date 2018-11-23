import random

UPDATE_EPSILON = 0.1
INITIAL_FIRE_PROB = 0.01
WEIGHT_INIT = 0.2
EDGE_PROB = 0.8
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

    def update_weight(self):
        if self.start.fire_prev and self.end.fire_now:
            self.weight *= (1 + UPDATE_EPSILON)


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


if __name__ == '__main__':
    net = HebbianNet(300)
    for _ in range(100):
        net.update()
        print("Edges over threshold: {} of {}".format(
            sum(syn.weight > 0.99 for syn in net.synapses),
            len(net.synapses)
        ))


