if __name__ == '__main__':
    """TODO: Document
    """
    from unoengine import UnoEngine
    from agents import RandomAgent, ReinforcementAgent

    agent_0 = ReinforcementAgent(unoengine.get_action_dim())
    agent_1 = RandomAgent(unoengine.get_action_dim())

    arena = Arena(agent_0, agent_1, unoengine)
