import numpy as np

def rollingMean(gamesHistory, windowSize = 100, player = 0):
    """calculate rolling mean of player 0
    :param gamesHistory: numpy array [[1,0],[0,1],...] of wins
    :param windowSize
    """
    gamesHistory = np.array(list(map(lambda row: row[player], gamesHistory)))
    rollingMeanHistory = np.empty(gamesHistory.shape[0] - windowSize + 1)
    for i in range(windowSize, gamesHistory.shape[0] + 1):
        windowMean = gamesHistory[i - windowSize:i].mean()
        rollingMeanHistory[i - windowSize] = windowMean
    return(rollingMeanHistory)

def totalMean(gamesHistory, player = 0):
    gamesHistory = np.array(list(map(lambda row: row[player], gamesHistory)))
    return gamesHistory.mean()
