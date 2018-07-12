import numpy as np

np.set_printoptions(precision = 4, suppress = True)


# P(z|s,a)
def p_obs(z, s, a):
    return 0.01 if (a in s and z == 'no') or (a not in s and z == 'yes') else 0.99


# P(s|b,a,z)
def obs_likelihood(knowledge, belief, action, obs):
    likelihood = np.zeros(len(belief))
    for i, _object in enumerate(knowledge):
        likelihood[i] = p_obs(obs, _object[0] + ' ' + _object[1], action) * belief[i]

    return likelihood


def belief_update(knowledge, belief, action, obs):
    updated_belief = obs_likelihood(knowledge, belief, action, obs)
    return updated_belief / np.sum(updated_belief, keepdims = True)


def pick_expected_reward(updated_belief):
    MAX = updated_belief.max()
    return MAX * 2 - 1


def value(knowledge, belief, actions = None, observations = None, gamma = 0.5, steps_remaining = 5):
    expected_terminal_reward = pick_expected_reward(belief)

    # Pruning sub-optimal actions
    if steps_remaining <= 0 or expected_terminal_reward > gamma * pick_expected_reward(np.ones(1)):
        return expected_terminal_reward

    Q_value = build_tree(knowledge, belief, actions, observations, steps_remaining = steps_remaining)

    return Q_value.max() if Q_value.max() > expected_terminal_reward else expected_terminal_reward


def build_tree(knowledge, belief, actions, observations, gamma = 0.5, steps_remaining = 5):
    if steps_remaining <= 0:
        return value(knowledge, belief, steps_remaining = 0)

    Q_value = np.zeros(len(actions))
    i = 0
    for action in actions:
        expected_value = 0
        for obs in observations:
            updated_belief = belief_update(knowledge, belief, action, obs)
            updated_actions = list(actions)
            updated_actions.remove(action)

            expected_value = expected_value + obs_likelihood(knowledge, belief, action, obs).sum() * value(knowledge,
                                                                                                           updated_belief,
                                                                                                           updated_actions,
                                                                                                           observations,
                                                                                                           gamma = gamma,
                                                                                                           steps_remaining = steps_remaining - 1)
        Q_value[i] = gamma * expected_value

        i = i + 1

    return Q_value


def plan(knowledge, belief):
    semantic = set()
    spatial = set()

    for _object in knowledge:
        semantic.add(_object[0])
        spatial.add(_object[1])

    most_likely_obs = ['yes', 'no']
    actions = list(semantic.union(spatial))

    Q_value = build_tree(knowledge, belief, actions, most_likely_obs, gamma = 0.9, steps_remaining = 5)

    best_action = actions[Q_value.argmax()]

    return 'pick' if pick_expected_reward(belief) > Q_value.max() else best_action


def main():
    knowledge = np.array([["yellow cup", "left"],
                          # ["yellow cup", "middle"],
                          # ["yellow cup", "right"],
                          # ["red cup", "left"],
                          ["red cup", "middle"],
                          ["red cup", "right"],
                          # ["green cup", "left"],
                          # ["green cup", "middle"],
                          # ["green cup", "right"],
                          ["blue cup", "left"],
                          ["blue cup", "middle"],
                          ["blue cup", "right"]
                          ])

    size = len(knowledge)

    # belief = np.array([1,5])
    belief = np.ones(size)
    belief = belief / belief.sum(keepdims = True)

    try:
        while True:
            print(belief)
            action = plan(knowledge, belief)
            if action is 'pick':
                print("Picks up: ", knowledge[belief.argmax()])
                break

            print(action + "?")
            obs = input()
            belief = belief_update(knowledge, belief, action, obs)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
