import numpy as np
import torch

import environment as env
import language_model as langmod
from string_utils import *

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
language_model.build_vocab_k_words(K = 20000)

# encode fixed descriptions
descriptions = [obj.description for obj in env.KNOWLEDGE]
encoded_descriptions = language_model.encode(descriptions)


# utils functions
def cosine(u, v):
    return np.dot(u, np.transpose(v)) / (np.linalg.norm(u, axis = 1, keepdims = True) * np.linalg.norm(v, axis = 1, keepdims = True))


def unigram_model(sentence):
    p_words = np.ones(env.n_objects)
    sentence = to_set(sentence, ' ')
    for i, obj in enumerate(env.KNOWLEDGE):
        common_words = len(obj.tokens.intersection(sentence))
        p_words[i] = len(obj) ** (-common_words) * (1e3 * len(obj)) ** (common_words - len(sentence))
    return p_words / p_words.sum(keepdims = True)


def sentence_similarity_score(sentence):
    return unigram_model(sentence) * cosine(encoded_descriptions, language_model.encode([sentence])).reshape(-1)


# P(z|s,a)
def p_obs(z, s, a):
    return 1e-5 if (a in s.tokens and z in no_list) or (a not in s.tokens and z in yes_list) else 1 - 1e-5


def split_sentence(observation):
    observation = clean_punctuation(observation)

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

    likelihood = np.ones(env.n_objects)
    if response is not '':
        for i, obj in enumerate(env.KNOWLEDGE):
            likelihood[i] = p_obs(response, obj, action)

    if len(description) > 0:
        likelihood = likelihood * sentence_similarity_score(description)
    return likelihood * belief


# P(s|b,a,z)
def belief_update(belief, action, obs):
    updated_belief = obs_likelihood(belief, action, obs)
    return updated_belief / updated_belief.sum(keepdims = True)


def init_distributions():
    dist = np.zeros((env.n_objects, env.n_actions, env.n_observations))
    for i, obj in enumerate(env.KNOWLEDGE):
        for j, action in enumerate(env.ACTIONS):
            for k, observation in enumerate(env.OBSERVATIONS):
                dist[i, j, k] = p_obs(observation, obj, action)

    return dist


DISTRIBUTIONS = init_distributions().reshape((env.n_objects, -1))
