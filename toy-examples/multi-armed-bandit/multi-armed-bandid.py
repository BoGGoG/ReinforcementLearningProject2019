import tensorflow as tf
import numpy as np

class Bandid:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def giveRandom(self):
        return 0.5


