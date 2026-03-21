import tensorflow as tf
import numpy as np
from data import *

classifier = tf.keras.models.load_model('word_model.keras')
classifier.trainable = False

real_sample = tf.cast(x_test[y_test == 1][:20], tf.int32)
fake_sample = tf.cast(x_test[y_test == 0][:20], tf.int32)
print("Real words:", classifier(real_sample, training=False).numpy().flatten())
print("Non-words:", classifier(fake_sample, training=False).numpy().flatten())

def runModel(s):
    # format input str
    pass

while True:
    i = input("Input 6 letter word:")
    i = i.strip()
    if len(i) != 6:
        print("Not a 6 letter word.")
        continue
    print(classifier.predict(i))
