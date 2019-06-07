"""
Marco Knipfer, June 2019

Note: Not working properly. Needs some trys in the training to get a good result ...

run: 
python3 frozenlake_NN.py
python3 frozenlake_NN_load.py

slippery = False for no random moves

Links used:
- Tutorial https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
- Save and restore tensorflow https://www.tensorflow.org/guide/saved_model
"""

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


###### PARAMETERS ######
slippery = False
saveFilePath = "model.ckpt"
gamma = .99
epsilon = 0.1
numEpisodes = 10000
# numEpisodes = 20000
stepsPerEpisode = 200


env = gym.make("FrozenLake-v0", map_name = "4x4", is_slippery = slippery)

###### NEURAL NETWORK #####
tf.reset_default_graph()

# feed-forward NN to choose actions
inputs1 = tf.placeholder(shape = [1, 16], dtype = tf.float32, name = "inputs1")
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01), name = "W")
# W = tf.get_variable("W", [16,4])
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# calculate loss: squared differences
nextQ = tf.placeholder(shape = [1,4], dtype = tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
updateModel = trainer.minimize(loss)

# initialization and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()

###### TRAINING THE NETWORK ######
# init = tf.initialize_all_variables()
jList = []
rList = []

with tf.Session() as sess:
    print("Tensoflow session started")
    sess.run(init)
    for i in tqdm(range(numEpisodes)):
        state = env.reset()
        rAll = 0
        done = False
        for j in range(stepsPerEpisode):
            action, allQ = sess.run([predict, Qout], feed_dict = {inputs1:np.identity(16)[state:state+1]})
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()
            state1, reward, done, _ = env.step(action[0])
            Q1 = sess.run(Qout, feed_dict = {inputs1:np.identity(16)[state1:state1+1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, action[0]] = reward + gamma * maxQ1
            _, W1 = sess.run([updateModel, W], feed_dict = {inputs1: np.identity(16)[state:state+1], nextQ:targetQ})
            rAll += reward
            state = state1
            if done:
                epsilon = 1. / ((i / 50.) + 10)
                if reward != 1:
                    reward = -5
                break
        jList.append(j)
        rList.append(rAll)
    savePath = saver.save(sess, saveFilePath)
    print("Model saved in path: {}".format(savePath))
print("Percent of succesful episodes: {}".format(sum(rList) / numEpisodes))
plt.plot(rList)
plt.show()
            


