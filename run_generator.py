import tensorflow as tf
import numpy as np
from shared import *

decoder = tf.keras.models.load_model('final_vae_decoder.keras')

# Generate new words
with open(train_words, 'r') as f:
    real_words = set(line.strip() for line in f)
with open(test_words, 'r') as f:
    real_words |= set(line.strip() for line in f)

def decode_word(indices):
    letters = []
    for i in indices:
        if i == EOS_TOKEN or i == PAD_TOKEN:
            break
        letters.append(chr(i + ord('a')))
    return ''.join(letters)

def decode_words(model_output):
    return list(map(decode_word, model_output))

def sample_with_temperature(probs, temperature=0.7):
    # probs shape: (N, 6, 26)
    log_probs = tf.math.log(probs + 1e-7) / temperature
    # tf.random.categorical expects shape (batch, num_classes) so reshape
    flat = tf.reshape(log_probs, (-1, VOCAB_SIZE))  # (N*6, 26)
    sampled = tf.random.categorical(flat, num_samples=1)  # (N*6, 1)
    return tf.reshape(sampled, (-1, MAX_LENGTH))  # (N, 6)

target = 100
fake_words = []
while len(fake_words) < target:
    needed = target - len(fake_words)
    random_latent_points = np.random.normal(size=(needed, latent_dim))
    generated = decoder.predict(random_latent_points)
    words = sample_with_temperature(generated, temperature=0.7)
    candidates = decode_words(words)
    fake_words += [w for w in candidates if w not in real_words]

print(fake_words)