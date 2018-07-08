import numpy as np

np.set_printoptions(precision = 4, suppress = True)


def entropy(p):
    return -np.dot(p, np.log2(p))


# P(z|s,a)
def p_obs(z, s, a):
    return 0.05 if (a in s and z == 'no') or (a not in s and z == 'yes') else 0.95


# P(z|b,a)
def obs_likelihood(knowledge, belief, action, obs):
    likelihood = np.zeros(len(belief))
    for i, _object in enumerate(knowledge):
        likelihood[i] = p_obs(obs, _object[0] + ' ' + _object[1], action) * belief[i]

    return likelihood


# P(s|b,a,z)
def belief_update(knowledge, belief, action, obs):
    updated_belief = obs_likelihood(knowledge, belief, action, obs)
    return updated_belief / np.sum(updated_belief, keepdims = True)


# H(X|U) = \sum_u P(u) \sum_x P(x\u) log P(x|u)
def conditional_entropy(knowledge, belief, action, observations):
    entropies = np.zeros(len(belief))

    for o, _object in enumerate(knowledge):
        probabilities = np.zeros(len(observations))
        for j, observation in enumerate(observations):
            probabilities[j] = p_obs(observation, _object[0] + ' ' + _object[1], action)
        entropies[o] = entropy(probabilities)

    return belief.dot(entropies)


def plan(knowledge, belief, actions, observations):
    information_gain = np.zeros(len(actions))
    for i, action in enumerate(actions):
        obs_probability = np.zeros(len(observations))
        for j, observation in enumerate(observations):
            obs_probability[j] = obs_likelihood(knowledge, belief, action, observation).sum()

        information_gain[i] = entropy(obs_probability) - conditional_entropy(knowledge, belief, action, observations)

    return actions[information_gain.argmax()]


def main():
    knowledge = np.array([["yellow cup", "left"],
                          # ["yellow cup", "middle"],
                          # ["yellow cup", "right"],
                          ["red cup", "left"],
                          ["red cup", "middle"],
                          # ["red cup", "right"],
                          # ["green cup", "left"],
                          ["green cup", "middle"],
                          ["green cup", "right"],
                          # ["blue cup", "left"],
                          # ["blue cup", "middle"],
                          ["blue cup", "right"]
                          ])

    size = len(knowledge)

    belief = np.ones(size)
    belief = belief / belief.sum(keepdims = True)

    semantic = set()
    spatial = set()

    for _object in knowledge:
        semantic.add(_object[0])
        spatial.add(_object[1])

    observations = ['yes', 'no']

    actions = list(semantic.union(spatial))

    try:
        while True:
            print(belief)
            if belief.max() > 0.7:
                print("Picks up: ", knowledge[belief.argmax()])
                break
            action = plan(knowledge, belief, actions, observations)
            print(action + "?")

            actual_observation = input()
            belief = belief_update(knowledge, belief, action, actual_observation)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
