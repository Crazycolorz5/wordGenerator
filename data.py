from shared import *

true_words = fileToData(train_words)
false_words = fileToData(train_unwords)

# scale factor controls how strongly frequencies are enforced
# higher scale = distributions closer to the frequency profile
scale = 10.0
concentration = english_frequencies * scale

def random_dirichlet_sequence(batch_size):
    sequences = np.zeros((batch_size, MAX_LENGTH, VOCAB_SIZE), dtype=np.float32)
    for i in range(batch_size):
        # pick a random length between 6 and 10
        length = np.random.randint(MIN_WORD_LENGTH, MAX_WORD_LENGTH)
        # fill letter positions with dirichlet samples
        sequences[i, :length, :26] = np.random.dirichlet(concentration, size=(length,))
        # place EOS token
        sequences[i, length, EOS_TOKEN] = 1.0
        # pad the rest — leave as zeros (all-zero vector for PAD positions)
    return tf.cast(sequences, tf.float32)

dirichlet_negatives = random_dirichlet_sequence(len(true_words))
# One-hot encode only the integer arrays
# Concatenate with dirichlet_negatives (already one-hot-like)
x_train = np.concatenate((tf.one_hot(np.concatenate((true_words, true_words, false_words),), depth=VOCAB_SIZE), dirichlet_negatives), axis=0)
y_train = np.concatenate((np.repeat(1, 2*len(true_words)), np.repeat(0, len(false_words)), np.repeat(0, len(dirichlet_negatives))),)

true_words_test = fileToData(test_words)
false_words_test = fileToData(test_unwords)
x_test = tf.one_hot(np.concatenate((true_words_test, false_words_test),), depth=VOCAB_SIZE)
y_test = np.concatenate((np.repeat(1, len(true_words_test)), np.repeat(0, len(false_words_test))),)