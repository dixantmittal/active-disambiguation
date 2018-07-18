import argparse

import numpy as np

import commons as coms
import environment as env
import greedy as gd
import pomdp as pd
import submodular as sm

MODE = {
    'pomdp': pd.plan,
    'greedy': gd.plan,
    'submodular': sm.plan
}


def get_reply(robot_says, intention = None, simulate = True):
    if simulate:
        reply = 'yes' if env.KNOWLEDGE[intention][0] in robot_says or env.KNOWLEDGE[intention][1] in robot_says else 'no'
        print('A:', reply)

        return reply

    else:
        return input('A: ')


def start_experiment(simulate = False, mode = 'user', n_runs = 5):
    env.random_scenario()
    preference = np.ones(env.n_objects)
    preference = preference / preference.sum(keepdims = True)
    
    for _ in range(n_runs):

        print("######### RUN:", _ + 1, "#########")
        belief = preference
        intention = np.random.randint(env.n_objects) if simulate else int(input('Enter your intention: '))
        print('\nIntention :', env.KNOWLEDGE[intention])

        planner = MODE[mode]
        try:
            while True:
                print('Belief: ', belief)
                if belief.max() > 0.8:
                    print('Picks up ' + env.KNOWLEDGE[belief.argmax()][0] + ' on ' + env.KNOWLEDGE[belief.argmax()][1])
                    print()
                    break
                else:
                    question = planner(belief)
                    robot_says = 'Did you mean the ' + env.ACTIONS[question] + '?'
                    print('Q:', robot_says)

                actual_observation = get_reply(robot_says = robot_says, intention = intention, simulate = simulate)

                belief = coms.belief_update(belief, env.ACTIONS[question], actual_observation)

        except KeyboardInterrupt:
            break

        preference = belief * preference + 0.01
        preference = preference / preference.sum(keepdims = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Mode to simulate: ')
    parser.add_argument('--simulate',
                        dest = 'simulate',
                        action = 'store_true',
                        help = 'Turn on Simulator')
    parser.add_argument('--planner',
                        dest = 'planner',
                        default = 'greedy',
                        help = 'Planner to use')
    parser.add_argument('--n_runs',
                        dest = 'n_runs',
                        default = '1',
                        help = 'number of runs')
    parser.set_defaults(sanity_check = False)
    args = parser.parse_args()

    start_experiment(args.simulate, args.planner, int(args.n_runs))
