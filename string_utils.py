no_list = ['no', 'nope', 'false', 'stop', 'not']
yes_list = ['yes', 'yep', 'go ahead', 'go on']


def clean_punctuation(str):
    return str.replace(',', '').replace('.', '')


def to_set(str, delim):
    return set(str.split(delim))
