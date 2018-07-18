from commons import *


def plan(belief, n_questions = 3):
    optimal_actions = []
    while n_questions > 0:
        information_gain = np.zeros(n_actions)
        for _action in range(n_actions):
            information_gain[_action] = conditional_entropy(_action, optimal_actions, belief)

        n_questions -= 1
        optimal_actions.append(information_gain.argmax())

    return optimal_actions[0]
