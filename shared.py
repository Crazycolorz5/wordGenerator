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

english_frequencies = tf.constant([
            0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202, 0.0609,
            0.0697, 0.0015, 0.0077, 0.0403, 0.0241, 0.0675, 0.0751, 0.0193,
            0.0010, 0.0599, 0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015,
            0.0197, 0.0007
        ], dtype=tf.float32)
