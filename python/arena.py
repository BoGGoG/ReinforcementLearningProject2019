"""
The class Arena provides a framework for a game played by two different opposing agents.
Upon instantiation, Arena takes two agents and one game environment as input parameters.
After that, all communication between the game environment and the two agents is handled by the Arena
instance's method 'step'.
In the main loop of training/ evaluating, only this step function should be used to alter the state of arena, agents,
and game environment.
The game environmanet and the agents must exchange information according to a well defined protocoll.
This protocoll is implicitely defined by the abstract classes 'Agent' and 'GameEnv'.
The Agent takes 'game_info' as an input, does its calculations, and returns an integer encoding his action.
The GameEnv takes this integer actions, evolves the game state, and returns the new 'game_info'.
This 'game_info' is a dictionary of the form defined in the documentation of the Agent class.
It has an attribure 'turn' that is used by the Arena to determine which Agent receives the game_info.

The specific encodings of actions and 'p_state' can be chosen freely as long as they obey the boundary conditions
defined by the abstract classes Agent and GameEnv.
Therefor, these classes should be inherited by the actual game environment and agent classes.
NOTE: there are restrictions to the encodings, e.g. the action-integer returned by the agent must be >=0 and
<= agent.get_state_dim(), where the latter variable is defined by the developers of the specifiv game.

This framework is intended for two player card games, although it should work for arbitrary turn based two player
games with discrete action spaces.
"""
import copy


class Arena:
    def __init__(self, agent_0, agent_1, game_env):
        """
        Framework to standardize game playing for two agents.
        The Goal is that independent teams can create their own Agents and Games independently and still be able
        to use the other team's game. Also, duelling of different agents should be possible.
        :param agent_0: should subclass the Agent class
        :param agent_1: " "
        :param game_env: should subclass the GameEnv class
        """
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        self.game_env = game_env

        # internal info about the game
        self.game_info = self.game_env.reset()

        # when one player wins, the other still has to receive the game_info in the next step to receive the reward.
        # we can reset the game only after both players received their rewards, so we need to keep track of for whom
        # the game ended separately
        self.agent_0_done = False
        self.agent_1_done = False

        # keep track of which agent performs best
        self.cum_agent_0_reward = 0.0
        self.cum_agent_1_reward = 0.0

    def step(self):
        """This Function is executed by the main loop of the training/ evaluation program.
        It wraps the whole agent-game dynamics.
        Execute one step, i.e. whichever agent's turn it is receives his game info (see Agent class) and plays one card
        :return: action taken by the agent whose turn it is (just for bookkeeping/ printing)
        """

        # keep track of whose turn it is and reward this agent with the reward given by game_env
        if self.game_info['turn'] == 0:
            agent = self.agent_0
            self.cum_agent_0_reward += self.game_info['reward']
        else:
            agent = self.agent_1
            self.cum_agent_1_reward += self.game_info['reward']

        # if both agents are done with this round, reset the game
        if self.agent_0_done and self.agent_1_done:
            self.game_info = self.game_env.reset()
            self.agent_0_done = False
            self.agent_1_done = False
            return None

        # give the current game info to the agent whose turn it is and get his action
        action = agent.digest(self.game_info)

        # according to protocoll, the agent returns None as action when it is game over for him.
        if not action:
            # feed the None action to the game_env, which needs to deal with that
            self.game_info = self.game_env.play(action)
        # If the agent supplies a valid action-integer, feed this to the game_env
        elif self.game_info['legal_actions'][action]:
            self.game_info = self.game_env.play(action)
        # If an agent cheats, the programm crashes with a ValueError
        else:
            raise ValueError('agent {} attempted an illegal move'.format(self.game_info['turn']))

        # keep track of game overs
        if self.game_info['game_over']:
            if self.game_info['turn'] == 0:
                self.agent_0_done = True
            else:
                self.agent_1_done = True

        return action

    def get_game_info(self):
        """This Function is executed by the main loop of the training/ evaluation program.
        It can be used for controlling the main loop and for printing/ bookkeeping.
        dicts are passed by reference so we need to make sure the user does not accidentally
        alter the internal state of arena
        :return: deepcopy of self.game_info
        """
        return copy.deepcopy(self.game_info)


class Agent(object):
    """Agent that plays the game. May be human, random or a reinforcement learner.
    The Agents made by each team should overwrite the functions defined here and follow the protocoll specified in the
    comments.
    """
    def digest(self, game_info):
        """This function is executed exclusively by the arena.
        It passes the game info and expects an action.
        :param game_info: dictionary with the following entries:

                        'turn':             0 or 1, whose turn it is
                        'p_state':          float array encoding the state of the game
                                            as viewed by the player who's 'turn' it is
                        'legal_actions':    boolean array encoding which actions are legal
                        'reward':           float reward that the player receives as a result from HIS last turn
                        'game_over':        boolean weather the game is over or not

                        NOTE: History should be encoded within the p_state array
        :return: integer encoding the action the player takes or None if the game is over.
                Integers must be in [0, ..., GameEnv.get_state_dim() - 1]
                This Interval is chosen such that it can be used as index of an array for later convenience
        """
        raise NotImplementedError


class GameEnv(object):
    """Game environment that simulates the card game
    The Game Envs made by each team should overwrite the functions defined here and follow the protocoll specified in
    the comments.
    """
    def reset(self):
        """This function is executed exclusively by the arena.
        reset the environment to begin a new game.
        :return: game_info for the first turn. (see Agent class)
        """
        raise NotImplementedError

    def play(self, action):
        """This function is executed exclusively by the arena.
        Execute one game step given the action of the current player.
        :param action: integer encoding the action, or None if the game is over and there is not action to take
        :return: game_info for the next turn. (see Agent class)
        """
        raise NotImplementedError

    def get_state_dim(self):
        """This Function is executed by the Agent developers.
        This function exists so that one does not need to know all details about the game environment in order
        to construct an agent
        :return: dimension of the array that encodes p_sate (see Agent class)
        """
        raise NotImplementedError

    def get_action_dim(self):
        """This Function is executed by the Agent developers.
        This function exists so that one does not need to know all details about the game environment in order
        to construct an agent
        :return: number of possible actions (see Agent class)
        """
        raise NotImplementedError


if __name__ == '__main__':
    """two random agents play one round against each other. The game is printed in textform
    """
    from unoengine import UnoEngine
    from agents import RandomAgent, ReinforcementAgent

    # setup the arena
    unoengine = UnoEngine()
    agent_0 = RandomAgent(unoengine.get_action_dim())
    # agent_0 = ReinforcementAgent(unoengine.get_action_dim())
    agent_1 = RandomAgent(unoengine.get_action_dim())

    arena = Arena(agent_0, agent_1, unoengine)

    # main loop
    # three rounds
    for _ in range(3):
        i = 0
        finished = False
        while not finished:
            # play until game over for both players
            finished = (arena.agent_0_done and arena.agent_1_done)

            # access game info for later printing. This is really only for printing.
            info = arena.get_game_info()
            player = info['turn']
            open_card = unoengine.card_names[int(info['p_state'][-2])]
            n_opponent_cards = info['p_state'][-1]
            hand_cards = unoengine.get_card_names(info['p_state'][:-2])
            reward = info['reward']

            # we keep track of game_over because we want to play exactly three rounds and exit the loop after that.
            # we call this before action,step() because the altered game_info after action.step() is for the opponent.
            game_over = info['game_over']

            # all game mechanics happens in the next line
            action = arena.step()

            # finally we print what was going on in the last step
            if game_over:
                output = 'Player {} finishes with a reward of {}'.format(player, reward)
            else:
                if action >= unoengine.n_different_cards:
                    a_name = 'draws '
                else:
                    a_name = 'plays ' + unoengine.card_names[action] + ' '

                output = 'Step {:3}: '.format(i)
                output += 'player {} '.format(player)
                output += a_name
                output += 'on ' + open_card + '. '
                output += 'Hand cards before that: {' + hand_cards + '}. '
                output += 'Number of opponents cards: {}'.format(n_opponent_cards) + '.'

            print(output)
            i += 1




