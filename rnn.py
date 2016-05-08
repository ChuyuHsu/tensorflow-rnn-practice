#!/usr/bin/env python
# -*- coding: utf-8 -*-
import data.load as load
import numpy as np
from functools import reduce


train_set, valid_set, test_set, dic = load.atisfold(3)
idx2label = dict((v,k) for k,v in dic['labels2idx'].items())
idx2word = dict((v,k) for k,v in dic['words2idx'].items())

train_lex, train_ne, train_y = train_set
valid_lex, valid_ne, valid_y = valid_set
test_lex,  test_ne,  test_y  = test_set

sent = train_lex[0]
print(list(map(lambda x: idx2word[x], sent)))

vocsize = len(set(reduce(lambda x, y: list(x)+list(y),
                         train_lex+valid_lex+test_lex)))

nclasses = len(set(reduce(lambda x, y: list(x)+list(y),
                          train_y+valid_y+test_y)))

nsentences = len(train_lex)

print("vocsize: %d, # of classes: %d, # of sentences: %d" % (vocsize, nclasses, nsentences))


def context_window(sentence, width=3):
    """
    window: int corresponding to the size of the window
        given a list of indexes composing a sentence

    :sentence: array containing the word indexes
    :width: window width
    :returns: return a list of list of indexes corresponding
        to context windows surrounding each word in the sentence
    """
    assert((width % 2) == 1)
    assert(width >= 1)
    l = list(sentence)

    lpadded = (width // 2) * [-1] + l + (width // 2) * [-1]
    out = [lpadded[i:(i + width)] for i in range(len(l))]
    return out


import tensorflow as tf


# nv: number of vocaburary
# de: dimension of word embedding
# cs: context of window size
nv, de, cs = 1000, 50, 5

# inputs and outputs
input_x = tf.placeholder(tf.int32, [None, sequence_len])

# word embedding
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(tf.random_uniform([nv, de], -1.0, 1.0), name='W')
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
