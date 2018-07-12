import argparse
from environment import *
from kafka import KafkaConsumer, KafkaProducer
import numpy as np

consumer = KafkaConsumer('active-disambiguation-questions', group_id = 'alpha')
producer = KafkaProducer(bootstrap_servers = 'localhost:9092')


def read():
    for question in consumer:
        return question.value.decode('utf-8')


def reply(message):
    producer.send('active-disambiguation-replies', message.encode('utf-8'))
    producer.flush()


def prepare_reply(question, intention):
    return 'yes' if KNOWLEDGE[intention][0] in question or KNOWLEDGE[intention][1] in question else 'no'


def simulate():
    intention = np.random.randint(len(KNOWLEDGE))
    n_questions = 0
    print('Intention :', KNOWLEDGE[intention])
    while True:
        question = read()
        print(question)

        if 'Picks up' in question:
            if KNOWLEDGE[intention][0] not in question or KNOWLEDGE[intention][1] not in question:
                print('PICKED WRONG OBJECT')
            break

        to_reply = prepare_reply(question, intention)
        print(to_reply)
        reply(to_reply)
        n_questions += 1

    print('Questions asked :', n_questions, '\n\n')
    return n_questions


def user():
    while True:
        question = read()
        print(question)

        if 'Picks up' in question:
            break

        reply(input())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Mode to simulate: ')
    parser.add_argument('--simulate',
                        dest = 'simulate',
                        action = 'store_true',
                        help = 'Turn on Simulator')
    parser.set_defaults(sanity_check = False)
    args = parser.parse_args()

    if args.simulate:
        n_q = 0
        for i in range(n_runs):
            n_q += simulate()
        print('Average questions: ', n_q / n_runs)
    else:
        user()

    consumer.commit()
