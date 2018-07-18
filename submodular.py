import numpy as np

import commons as coms
import environment as env


def plan(belief, n_questions = 3):
    optimal_actions = []
    while n_questions > 0:
        information_gain = np.zeros(env.n_actions)
        for _action in range(env.n_actions):
            information_gain[_action] = coms.conditional_entropy(_action, optimal_actions, belief)

        n_questions -= 1
        optimal_actions.append(information_gain.argmax())

    return optimal_actions[0]
