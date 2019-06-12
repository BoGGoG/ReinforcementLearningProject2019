import tensorflow as tf

class Qnetwork():
    def __init__(self, inputLength, outputLength):
        """
        NOT IMPLEMENTED
        Qnetwork for uno
        :param inputLength: Size of the state, every element of the vector means how many of this card the agent has
        :param outpuLength: Number of actions, i.e. number of different cards in the game
        """
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.inputPlaceholder = tf.placeholder(shape = [1, inputLength], dtype=tf.float32)
        self.weights = tf.get_variable("weights", shape = [inputLength, outputLength])
        self.qout = tf.matmul(self.inputPlaceholder, self.weights)
        predict = tf.argmax(self.qout, 1)




