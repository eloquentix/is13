__author__ = 'gabriel'

import re
import pickle
from random import shuffle

DIGIT_RE = re.compile('\d')


def decode_s(triple):
    first, second, third = triple
    first = first.encode('utf8', errors='replace').decode('utf8')
    second = second.encode('utf8', errors='replace').decode('utf8')
    third = third.encode('utf8', errors='replace').decode('utf8')
    return first, second, third


def load_data():
    train_corpus = pickle.load(open('/tmp/train.pck', 'rb'))
    test_corpus = pickle.load(open('/tmp/test.pck', 'rb'))

    train_corpus = [[decode_s(triple) for triple in sent] for sent in train_corpus]
    test_corpus = [[decode_s(triple) for triple in sent] for sent in test_corpus]

    # corpus = [[(word.decode('utf-8', errors='replace'), tag.decode('utf-8', errors='replace'),
    #             cls.decode('utf-8', errors='replace')) for word, tag, cls in sent] for sent in corpus]
    # remove for end of days
    shuffle(train_corpus)
    # corpus = corpus[:100]
    # remove for end of days
    train_sents = [[s[0] for s in x] for x in train_corpus]
    train_target = [[t[2] for t in sent] for sent in train_corpus]
    test_sents = [[s[0] for s in x] for x in test_corpus]
    test_target = [[t[2] for t in sent] for sent in test_corpus]
    target_lex = list(set([t for sent in test_target + train_target for t in sent]))
    target_lex = dict([(w, i) for i, w in enumerate(target_lex)])
    freq = get_freq(train_sents)
    train_sents = [replace_digits(sent) for sent in train_sents]
    train_sents = [replace_uniq(sent, freq) for sent in train_sents]
    test_sents = [replace_digits(sent) for sent in test_sents]
    test_sents = [replace_uniq(sent, freq) for sent in test_sents]
    words = list(set([w for sent in train_sents for w in sent]))
    lex = dict([(w, i) for i, w in enumerate(words)])
    train_sents = [[lex[w] for w in sent] for sent in train_sents]
    train_target = [[target_lex[t] for t in sent] for sent in train_target]
    test_sents = [[lex[w] for w in sent] for sent in test_sents]
    test_target = [[target_lex[t] for t in sent] for sent in test_target]
    dic = {'labels2idx': target_lex, 'words2idx': lex}
    # to make sure
    data = list(zip(test_sents, test_target))
    shuffle(data)
    test, valid = test_validate_split(data, test_size=0.5)
    train_lex, train_y = train_sents, train_target
    valid_lex, valid_y = unzip(valid)
    test_lex, test_y = unzip(test)
    # No clue what these are
    train_ne, test_ne, valid_ne = [], [], []
    return (train_lex, train_ne, train_y), (valid_lex, valid_ne, valid_y), (test_lex, test_ne, test_y), dic


def test_validate_split(data, test_size=0.5):
    """
    Split the data list into 3 different lists, the train one will have train_size * original_len
    The other 2 will have each half of the remaining length
    :param data: The list to be split (zip if multiple)
    :param train_size: float from 0 to 1 signifying a percentage
    :return: 3 separate lists
    """
    # dirty filthy hack
    shuffle(data)
    staying = int(test_size * len(data))
    test = data[:staying]
    valid = data[staying:]
    return test, valid


def unzip(data):
    """
    'Unzip' a list of tuples into 2 separate lists, further code should be added to
    extend this to fit for triples or more
    :param data: list of tuples
    :return: 2 separate lists
    """
    data = list(zip(*data))
    return list(data[0]), list(data[1])


def replace_digits(sent):
    return [DIGIT_RE.sub('DIGIT', w) for w in sent]


def replace_uniq(sent, freq):
    return ['<UNK>' if w not in freq or freq[w] <= 1 else w for w in sent]


def get_freq(sents):
    freq = {}
    for sent in sents:
        for w in sent:
            if w not in freq:
                freq[w] = 1
            else:
                freq[w] += 1
    return freq

if __name__ == '__main__':
    load_data()

