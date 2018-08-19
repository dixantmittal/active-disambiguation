import numpy as np
import torch

import environment as env
import language_model as langmod

np.set_printoptions(precision = 2, suppress = True)

# Load language model
language_model = langmod.BLSTMEncoder({'word_emb_dim': 300,
                                       'enc_lstm_dim': 2048,
                                       'pool_type': 'max',
                                       'bsize': 64,
                                       'dpout_model': 0.0,
                                       'training': False,
                                       'glove_path': '/Users/dixantmittal/nltk_data/glove.840B.300d.txt'})
language_model.load_state_dict(torch.load('infersent.allnli.pickle'))
language_model.build_vocab_k_words(K = 10000)

# encode fixed descriptions
descriptions = [(obj[0] + ' ' + obj[1]) for obj in env.KNOWLEDGE]
encoded_descriptions = language_model.encode(descriptions)


# utils functions
def cosine(u, v):
    return np.dot(u, np.transpose(v)) / (np.linalg.norm(u, axis = 1, keepdims = True) * np.linalg.norm(v, axis = 1, keepdims = True))


def sentence_similarity_score(sentence):
    sentence = language_model.encode([sentence])
    cosine_distance = cosine(encoded_descriptions, sentence).reshape(-1)
    return (cosine_distance - np.min(cosine_distance) + 1e-20) ** 2


no_list = ['no', 'nope', 'false', 'stop', 'not']
yes_list = ['yes', 'yep', 'go ahead', 'go on']


# P(z|s,a)
def p_obs(z, s, a):
    s = s[0] + ' ' + s[1]
    lie = 1e-20
    return lie if (a in s and z in no_list) or (a not in s and z in yes_list) else 1 - lie


def split_sentence(observation):
    observation = observation.replace(',', '')

    response = ''
    description = observation
    for word in observation.split(' '):
        if word in (no_list + yes_list):
            response = word
            description = description.replace(word, '')
            break

    return response, description


# P(z|b,a)
def obs_likelihood(belief, action, observation):
    response, description = split_sentence(observation)

    likelihood = np.zeros(len(belief))
    for i, obj in enumerate(env.KNOWLEDGE):
        likelihood[i] = p_obs(response, obj, action) * belief[i]

    if len(description) > 0:
        likelihood = likelihood * sentence_similarity_score(description)
    return likelihood


# P(s|b,a,z)
def belief_update(belief, action, obs):
    updated_belief = obs_likelihood(belief, action, obs)
    return updated_belief / updated_belief.sum(keepdims = True)


def init_distributions():
    dist = np.zeros((env.n_objects, env.n_actions, env.n_observations))
    for i, _object in enumerate(env.KNOWLEDGE):
        for j, action in enumerate(env.ACTIONS):
            for k, observation in enumerate(env.OBSERVATIONS):
                dist[i, j, k] = p_obs(observation, _object, action)

    return dist


DISTRIBUTIONS = init_distributions().reshape((env.n_objects, -1))
