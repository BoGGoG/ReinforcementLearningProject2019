import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time
print(tf.__version__)

steps = 100

useLongTrainedModel = True
currentModelPath = "model.ckpt"
longTrainedModelPath = 'trainedModel/model.ckpt'
if useLongTrainedModel:
    saveFilePath = longTrainedModelPath
else:
    saveFilePath = currentModelPath

slippery = True
env = gym.make("FrozenLake-v0", map_name = "4x4", is_slippery = slippery)


###### NEURAL NETWORK #####
tf.reset_default_graph()

# feed-forward NN to choose actions
inputs1 = tf.placeholder(shape = [1, 16], dtype = tf.float32, name = "inputs1")
# W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
W = tf.get_variable("W", shape = [16,4])
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

def printAction(action):
    directions = ['left', 'down', 'right', 'up']
    print('wanted next direction ' + directions[action])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, saveFilePath)
    print("Model restored")
    print("W: {}".format(W.eval()))
    state = env.reset()
    for j in range(steps):
        action, allQ = sess.run([predict, Qout], feed_dict = {inputs1:np.identity(16)[state:state+1]})
        state, reward, done, _ = env.step(action[0])
        printAction(action[0])
        env.render()
        print("reward: {}".format(reward))
        time.sleep(0.3)
        if done:
            print("finished after {} steps".format(j))
            break



