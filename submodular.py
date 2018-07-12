import numpy as np
from environment import *
from kafka import KafkaProducer, KafkaConsumer

np.set_printoptions(precision = 4, suppress = True)

consumer = KafkaConsumer('active-disambiguation-replies', group_id = 'alpha')
producer = KafkaProducer(bootstrap_servers = 'localhost:9092')


# P(z|s,a)
def p_obs(z, s, a):
    return 0.01 if (a in s and z == 'no') or (a not in s and z == 'yes') else 0.99


# P(s|b,a,z)
def belief_update(belief, action, obs):
    updated_belief = np.zeros(len(belief))
    for i, _object in enumerate(KNOWLEDGE):
        updated_belief[i] = p_obs(obs, _object[0] + ' ' + _object[1], action) * belief[i]
    return updated_belief / np.sum(updated_belief, keepdims = True)


# P(z1, z2, z3) = \sum_s P(z1|s) * P(z2|s) * P(z3|s)
# variable =                    observations with proper index
# conditional_distributions =   array containing distributions
def joint_probability(variables):
    probability = 0
    for _object in range(n_objects):
        conditional = 1
        for observation in variables:
            conditional = conditional * DISTRIBUTIONS[_object, observation]

        probability = probability + conditional * BELIEF[_object]

    return probability


# H(X|U) = - \sum_u P(u) \sum_x P(x|u) log P(x|u)
def conditional_entropy(conditional_variable, observed_variables = []):
    _entropy = 0

    for i in range(2 ** len(observed_variables)):
        observations = []
        for variable in observed_variables:
            observations.append(variable * 2 + i % 2)
            i = i // 2

        j_p = np.zeros(n_observations)

        j_p[0] = joint_probability(observations + [conditional_variable * 2])
        j_p[1] = joint_probability(observations + [conditional_variable * 2 + 1])

        _entropy = _entropy - np.dot(j_p, (np.log2(j_p) - np.log2(j_p.sum())))

    return _entropy


def plan(n_questions = 3):
    optimal_actions = []
    while n_questions > 0:
        information_gain = np.zeros(n_actions)
        for _action in range(n_actions):
            information_gain[_action] = conditional_entropy(_action, optimal_actions)

        n_questions -= 1
        optimal_actions.append(information_gain.argmax())

    return optimal_actions


def init_distributions():
    dist = np.zeros((len(KNOWLEDGE), len(ACTIONS), len(OBSERVATIONS)))
    for i, _object in enumerate(KNOWLEDGE):
        for j, action in enumerate(ACTIONS):
            for k, observation in enumerate(OBSERVATIONS):
                dist[i, j, k] = p_obs(observation, _object, action)

    return dist


if __name__ == '__main__':

    semantic = set()
    spatial = set()

    for desc in KNOWLEDGE:
        semantic.add(desc[0])
        spatial.add(desc[1])

    OBSERVATIONS = ['yes', 'no']

    ACTIONS = list(semantic.union(spatial))

    n_objects = len(KNOWLEDGE)
    n_actions = len(ACTIONS)
    n_observations = len(OBSERVATIONS)

    DISTRIBUTIONS = init_distributions().reshape((n_objects, -1))

    for _ in range(n_runs):
        BELIEF = np.ones(n_objects)
        BELIEF = BELIEF / BELIEF.sum(keepdims = True)
        try:
            while True:
                actions = plan(3)
                action = actions[0]

                for action in actions:
                    # print(BELIEF)
                    if BELIEF.max() > 0.7:
                        producer.send('active-disambiguation-questions',
                                      ('Picks up ' + KNOWLEDGE[BELIEF.argmax()][0] + ' on ' + KNOWLEDGE[BELIEF.argmax()][1]).encode('utf-8'))
                        producer.flush()
                        print('Picks up ' + KNOWLEDGE[BELIEF.argmax()][0] + ' on ' + KNOWLEDGE[BELIEF.argmax()][1])
                        raise KeyboardInterrupt

                    producer.send('active-disambiguation-questions', ('Did you mean the ' + ACTIONS[action] + '?').encode('utf-8'))
                    producer.flush()

                    for reply in consumer:
                        actual_observation = reply.value.decode('utf-8')
                        break

                    BELIEF = belief_update(BELIEF, ACTIONS[action], actual_observation)
                    if actual_observation is 'yes':
                        break

        except KeyboardInterrupt:
            pass
