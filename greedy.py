import numpy as np

np.set_printoptions(precision=4, suppress=True)


def p_obs(z, s, a):
    if (a in s and z == 'no') or (a not in s and z == 'yes'):
        return 0.01
    else:
        return 0.99


def obs_likelihood(knowledge, belief, action, obs):
    likelihood = np.zeros(len(belief))
    for i, _object in enumerate(knowledge):
        likelihood[i] = p_obs(obs, _object[0] + ' ' + _object[1], action) * belief[i]

    return likelihood


def belief_update(knowledge, belief, action, obs):
    updated_belief = obs_likelihood(knowledge, belief, action, obs)
    return updated_belief / np.sum(updated_belief, keepdims=True)


def cross_entropy_score(p):
    return -np.dot(p, np.log(p))


def l1_gain(p1, p2):
    return cross_entropy_score(p1) - cross_entropy_score(p2)
    # return np.sum(np.abs(arr1 - arr2))


def pick_expected_reward(updated_belief):
    MAX = updated_belief.max()
    return MAX * 2 - 1


def plan(knowledge, belief):
    semantic = set()
    spatial = set()

    for _object in knowledge:
        semantic.add(_object[0])
        spatial.add(_object[1])

    most_likely_obs = ['yes', 'no']

    best_action = ""
    best_gain = -np.inf

    for desc in semantic.union(spatial):
        action = desc
        gain = 0
        for obs in most_likely_obs:
            updated_belief = belief_update(knowledge, belief, action, obs)
            gain = gain + (l1_gain(belief, updated_belief) + 0.9 * pick_expected_reward(updated_belief)) * np.sum(
                obs_likelihood(knowledge, belief, action, obs))

        # asking questions incurs small price
        # gain = gain - 0.5

        if gain > best_gain:
            best_gain = gain
            best_action = action

    if pick_expected_reward(belief) > best_gain:
        return 'pick'

    return best_action


def main():
    knowledge = np.array([["yellow cup", "left"],
                          ["yellow cup", "middle"],
                          ["yellow cup", "right"],
                          # ["red cup", "left"],
                          # ["red cup", "middle"],
                          # ["red cup", "right"],
                          # ["green cup", "left"],
                          # ["green cup", "middle"],
                          # ["green cup", "right"],
                          ["blue cup", "left"],
                          ["blue cup", "middle"],
                          ["blue cup", "right"]])

    size = len(knowledge)

    # belief = np.array([1,5])
    belief = np.ones(size)
    belief = belief / belief.sum(keepdims=True)

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
