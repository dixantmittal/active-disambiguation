from commons import *
import argparse


def plan(belief):
    information_gain = np.zeros(n_actions)
    for action in range(n_actions):
        for obs in OBSERVATIONS:
            updated_belief = belief_update(belief, ACTIONS[action], obs)
            information_gain[action] = information_gain[action] + entropy_diff(belief, updated_belief) * obs_likelihood(belief, ACTIONS[action],
                                                                                                                        obs).sum()
    return information_gain.argmax()


def get_observation(robot_says, intention = None, simulate = True):
    if simulate:
        if 'Picks up' in robot_says:
            if KNOWLEDGE[intention][0] not in robot_says or KNOWLEDGE[intention][1] not in robot_says:
                print('PICKED WRONG OBJECT')
            return

        reply = 'yes' if KNOWLEDGE[intention][0] in robot_says or KNOWLEDGE[intention][1] in robot_says else 'no'
        print(reply)

        return reply

    else:
        return input()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Mode to simulate: ')
    parser.add_argument('--simulate',
                        dest = 'simulate',
                        action = 'store_true',
                        help = 'Turn on Simulator')
    parser.set_defaults(simulate = False)
    args = parser.parse_args()

    preference = np.ones(n_objects)
    preference = preference / preference.sum(keepdims = True)

    n_runs = 4
    for _ in range(n_runs):
        belief = preference
        intention = np.random.randint(n_objects)
        intention = 0
        if args.simulate: print('\nIntention :', KNOWLEDGE[intention])
        print(belief)
        try:
            while True:
                if belief.max() > 0.8:
                    print('Picks up ' + KNOWLEDGE[belief.argmax()][0] + ' on ' + KNOWLEDGE[belief.argmax()][1])
                    print()
                    break
                else:
                    action = plan(belief)
                    robot = 'Did you mean the ' + ACTIONS[action] + '?'
                    print(robot)

                actual_observation = get_observation(robot_says = robot, intention = intention, simulate = True)

                belief = belief_update(belief, ACTIONS[action], actual_observation)

        except KeyboardInterrupt:
            break

        preference = belief * preference + 0.01
        preference = preference / preference.sum(keepdims = True)
