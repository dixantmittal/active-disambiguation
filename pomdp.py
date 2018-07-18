from commons import *


def pick_expected_reward(updated_belief):
    MAX = updated_belief.max()
    return MAX * 2 - 1


def value(belief, gamma = 0.5, steps_remaining = 5):
    expected_terminal_reward = pick_expected_reward(belief)

    # Pruning sub-optimal actions
    if steps_remaining <= 0 or expected_terminal_reward > gamma * pick_expected_reward(np.ones(1)):
        return expected_terminal_reward

    Q_value = build_tree(belief, gamma = gamma, steps_remaining = steps_remaining)

    return Q_value.max() if Q_value.max() > expected_terminal_reward else expected_terminal_reward


def build_tree(belief, gamma = 0.5, steps_remaining = 5):
    if steps_remaining <= 0:
        return value(belief, steps_remaining = 0)

    Q_value = np.zeros(n_actions)

    for i, action in enumerate(ACTIONS):
        expected_value = 0
        for obs in OBSERVATIONS:
            updated_belief = belief_update(belief, action, obs)
            expected_value = expected_value + obs_likelihood(belief, action, obs).sum() * value(updated_belief, gamma = gamma,
                                                                                                steps_remaining = steps_remaining - 1)
        Q_value[i] = gamma * expected_value

    return Q_value


def plan(belief):
    Q_value = build_tree(belief, gamma = 0.9, steps_remaining = 3)
    return 'pick' if pick_expected_reward(belief) > Q_value.max() else Q_value.argmax()
