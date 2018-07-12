from commons import *


def plan(belief):
    information_gain = np.zeros(n_actions)
    for action in range(n_actions):
        for obs in OBSERVATIONS:
            updated_belief = belief_update(belief, ACTIONS[action], obs)
            information_gain[action] = information_gain[action] + entropy_diff(belief, updated_belief) * obs_likelihood(belief, ACTIONS[action],
                                                                                                                        obs).sum()
    return information_gain.argmax()


if __name__ == '__main__':
    for _ in range(n_runs):
        belief = np.ones(n_objects)
        belief = belief / belief.sum(keepdims = True)
        try:
            while True:

                if belief.max() > 0.7:
                    producer.send('active-disambiguation-questions',
                                  ('Picks up ' + KNOWLEDGE[belief.argmax()][0] + ' on ' + KNOWLEDGE[belief.argmax()][1]).encode('utf-8'))
                    producer.flush()
                    print('Picks up ' + KNOWLEDGE[belief.argmax()][0] + ' on ' + KNOWLEDGE[belief.argmax()][1])
                    raise KeyError

                action = plan(belief)
                producer.send('active-disambiguation-questions', ('Did you mean the ' + ACTIONS[action] + '?').encode('utf-8'))
                producer.flush()

                for reply in consumer:
                    actual_observation = reply.value.decode('utf-8')
                    break

                belief = belief_update(belief, ACTIONS[action], actual_observation)

        except KeyError:
            pass
        except KeyboardInterrupt:
            break

    consumer.commit()
