import numpy as np
import tensorflow as tf
import random

latent_dim = 16
label_smoothing = 0.1

MIN_WORD_LENGTH = 6
MIN_LENGTH = 6 + 1  # +1 for EOS token
MAX_WORD_LENGTH = 10
MAX_LENGTH = MAX_WORD_LENGTH + 1  # +1 for EOS token

PAD_TOKEN = 26
EOS_TOKEN = 27
VOCAB_SIZE = 28  # 26 letters + PAD + EOS

def vectorize(word):
    # Convert word to list of indices (a=0, b=1, ..., z=25)
    return [ord(c) - ord('a') for c in word]

def pad(word):
    if len(word) < MAX_LENGTH:
        word += [PAD_TOKEN] * (MAX_LENGTH - len(word))
    return word

def fileToData(fname):
    with open(fname, 'r') as f:
        return np.array([pad(vectorize(line.strip()) + [EOS_TOKEN]) for line in f.readlines()], dtype=np.int32)

train_words = 'train_words.txt'
train_unwords = 'train_unwords.txt'
test_words = 'test_words.txt'
test_unwords = 'test_unwords.txt'

english_frequencies = tf.constant([0.078, 0.02, 0.04, 0.038, 0.1137, 0.014, 0.03, 0.023, 0.086, 0.0025, 0.0097, 0.053, 0.027, 0.072, 0.061, 0.028, 0.0019, 0.073, 0.087, 0.067, 0.033, 0.01, 0.0091, 0.0027, 0.016, 0.0044], dtype=tf.float32)
