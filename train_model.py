import tensorflow as tf
from data import *

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(MAX_LENGTH, VOCAB_SIZE)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing), # No longer a yes/no, now want meaningful information about how "real word" it is.
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=3,
    min_delta=0.01,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=30, callbacks=[callback])

model.save('word_model.keras')
model.save('word_model_backup.keras')

# model = tf.keras.models.load_model('word_model.keras')

# model.predict(x)


# Utility stuff output by Claude
def predict_from_indices(indices):
    """Convenience wrapper for inference only. 
    Use saved model directly for differentiable contexts."""
    one_hot = tf.one_hot(indices, depth=VOCAB_SIZE)
    return model(one_hot, training=False).numpy()
