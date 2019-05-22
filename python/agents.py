import numpy as np
from python.arena import Agent


class RandomAgent(Agent):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def digest(self, game_info):
        if game_info['game_over']:
            return None

        legal_actions = game_info['legal_actions']
        if legal_actions.sum() <= 1:  # can only draw
            return self.action_dim-1
        else:
            p = game_info['legal_actions'][:-1]
            p = p / p.sum()
        return np.random.choice(self.action_dim-1, p=p)
