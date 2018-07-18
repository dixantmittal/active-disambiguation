import numpy as np

n_runs = 100

KNOWLEDGE = WORLD = [
    ["yellow cup", "left"],
    ["yellow cup", "middle"],
    ["yellow cup", "right"],
    ["red cup", "left"],
    ["red cup", "middle"],
    ["red cup", "right"],
    ["green cup", "left"],
    ["green cup", "middle"],
    ["green cup", "right"],
    ["blue cup", "left"],
    ["blue cup", "middle"],
    ["blue cup", "right"]
]

semantic = set()
spatial = set()
for desc in KNOWLEDGE:
    semantic.add(desc[0])
    spatial.add(desc[1])

ACTIONS = list(semantic.union(spatial))

OBSERVATIONS = ['yes', 'no']

n_objects = len(KNOWLEDGE)
n_actions = len(ACTIONS)
n_observations = len(OBSERVATIONS)


def random_scenario():
    global KNOWLEDGE

    scenario = set()

    for i in range(np.maximum(np.random.randint(len(WORLD)), 3)):
        scenario.add(np.random.randint(len(WORLD)))

    KNOWLEDGE = []
    for i in scenario:
        KNOWLEDGE.append(WORLD[i])

    global n_objects
    n_objects = len(KNOWLEDGE)
