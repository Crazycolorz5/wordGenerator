import tensorflow as tf
import numpy as np
import keras
from shared import *
from data import *

pretrain = True

classifier = tf.keras.models.load_model('word_model.keras')
classifier.trainable = True

# Encoder
encoder_inputs = tf.keras.Input(shape=(MAX_LENGTH, VOCAB_SIZE))
x = tf.keras.layers.LSTM(128)(encoder_inputs)
z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)

@keras.saving.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z])

# Decoder
decoder_inputs = tf.keras.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(128, activation='relu')(decoder_inputs)
x = tf.keras.layers.RepeatVector(MAX_LENGTH)(x)
x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
decoder_outputs = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax')
)(x)
decoder = tf.keras.Model(decoder_inputs, decoder_outputs)


# VAE
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, classifier, batch_size=32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.classifier_optimizer = tf.keras.optimizers.Adam()
        self._batch_size = batch_size
        self.trainClassifier = False
        self.letter_frequencies = tf.constant([
            0.0817, 0.0149, 0.0278, 0.0425, 0.1270, 0.0223, 0.0202, 0.0609,
            0.0697, 0.0015, 0.0077, 0.0403, 0.0241, 0.0675, 0.0751, 0.0193,
            0.0010, 0.0599, 0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015,
            0.0197, 0.0007
        ], dtype=tf.float32)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data

        # We are using a GAN style interleaved training, with the VAE as the generator and the classifier as the discriminator.
        # We begin training with a forward pass of the VAE:

        with tf.GradientTape() as vae_tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            targets = tf.argmax(x, axis=-1)  # shape (N, MAX_LENGTH)

            # mask is 1 where target is not PAD, 0 where it is
            mask = tf.cast(tf.not_equal(targets, PAD_TOKEN), tf.float32)

            per_position_loss = tf.keras.losses.sparse_categorical_crossentropy(
                targets, reconstruction
            )  # shape (N, MAX_LENGTH)

            reconstruction_loss = tf.reduce_sum(per_position_loss * mask) / tf.reduce_sum(mask)
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            classifier_score = self.classifier(reconstruction, training=False)
            classifier_loss = -tf.reduce_mean(tf.math.log(classifier_score + 1e-7))
            
            # Penalty for repeated letters. This was becoming a big issue in testing.
            consecutive_similarity = tf.constant(0.0)
            # Consider runs of 2, 3, ... 6 letters.
            # Mask out PAD tokens, assuming they only occur after EOS token.
            mask = tf.cast(tf.not_equal(targets[:, 1:], PAD_TOKEN), tf.float32)
            adjacent = tf.reduce_sum(reconstruction[:, 1:] * reconstruction[:, :-1], axis=-1)
            adjacent = adjacent * mask
            for i in range(1, MAX_LENGTH):
                # product of i consecutive adjacent pairs — only high if all i pairs are similar
                run_product = tf.reduce_sum(
                    tf.stack([tf.reduce_prod(adjacent[:, j:j+i], axis=-1) 
                            for j in range(MAX_LENGTH-i)], axis=-1)
                )
                consecutive_similarity += (i ** 2.5) * run_product


            # Penalty for rare letters
            # mean probability assigned to each letter across all positions
            mask_expanded = tf.cast( 
                tf.logical_and( # Remove PAD and EOS tokens from frequency calculation
                    tf.not_equal(targets, PAD_TOKEN),
                    tf.not_equal(targets, EOS_TOKEN)
                ), tf.float32
            )[:, :, tf.newaxis]

            masked_reconstruction = reconstruction * mask_expanded
            mean_letter_probs = tf.reduce_sum(masked_reconstruction, axis=[0, 1]) / tf.reduce_sum(mask_expanded)
            
            frequency_penalty = tf.reduce_sum(
                tf.maximum(0.0, mean_letter_probs - tf.pad(self.letter_frequencies, [[0, VOCAB_SIZE - 26]]))
            )

            total_loss = reconstruction_loss + 0.1 * kl_loss + 2 * classifier_loss + 0.2 * consecutive_similarity + 0.5 * frequency_penalty

        vae_grads = vae_tape.gradient(total_loss, 
                                    self.encoder.trainable_variables + 
                                    self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(vae_grads,
                                        self.encoder.trainable_variables + 
                                        self.decoder.trainable_variables))

        # And now do an adversarial pass on the classifier:
        with tf.GradientTape() as cls_tape:
            real_scores = self.classifier(x, training=True)
            # Consider VAE output as positive as we want to train the in-between spaces as word-like.
            vae_scores = self.classifier(reconstruction, training=True)
            
            # two random strings to match two positive examples
            random_scores_1 = self.classifier(random_dirichlet_sequence(self._batch_size), training=True)
            random_scores_2 = self.classifier(random_dirichlet_sequence(self._batch_size), training=True)
            
            cls_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones_like(real_scores) * (1-label_smoothing), real_scores)
            ) + tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.ones_like(vae_scores) * (1-label_smoothing), vae_scores)
            ) + tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros_like(random_scores_1) + label_smoothing, random_scores_1)
            ) + tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(tf.zeros_like(random_scores_2) + label_smoothing, random_scores_2)
            )
        if self.trainClassifier:
            cls_grads = cls_tape.gradient(cls_loss, self.classifier.trainable_variables)
            self.classifier_optimizer.apply_gradients(
                zip(cls_grads, self.classifier.trainable_variables)
            )

        return {"loss": total_loss, 
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss, 
            "classifier_loss": classifier_loss,
            "cls_gan_loss": cls_loss,
            "consecutive_penalty": consecutive_similarity,
            "frequency_penalty": frequency_penalty}
    
    def save(self, prefix='vae'):
        self.encoder.save(prefix + '_encoder.keras')
        self.decoder.save(prefix + '_decoder.keras')

    def load(self, prefix='vae'):
        self.encoder = tf.keras.models.load_model(prefix + '_encoder.keras')
        self.decoder = tf.keras.models.load_model(prefix + '_decoder.keras')
    
vae = VAE(encoder, decoder, classifier)
vae.compile(optimizer='adam')

class AnyImprovingEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitors, patience=5, min_delta=0.01):
        super().__init__()
        self.monitors = monitors
        self.patience = patience
        self.min_delta = min_delta
        self.best = {}
        self.stagnant_counts = {}
        self.best_weights = None
        self.best_total_loss = float('inf')

    def on_train_begin(self, logs=None):
        self.best = {m: float('inf') for m in self.monitors}
        self.stagnant_counts = {m: 0 for m in self.monitors}
        self.best_weights = None
        self.best_total_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        # track best weights by total loss
        total_loss = logs.get('loss')
        if total_loss is not None and total_loss < self.best_total_loss:
            self.best_total_loss = total_loss
            self.best_weights = self.model.get_weights()

        for monitor in self.monitors:
            current = logs.get(monitor)
            if current is None:
                continue
            if current < self.best[monitor] - self.min_delta:
                self.best[monitor] = current
                self.stagnant_counts[monitor] = 0
            else:
                self.stagnant_counts[monitor] += 1

        if all(self.stagnant_counts[m] >= self.patience for m in self.monitors):
            print(f"\nAll monitored losses stagnant for {self.patience} epochs, stopping.")
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                print("Restored best weights.")
            self.model.stop_training = True

callback = AnyImprovingEarlyStopping(
    monitors=['loss', 'classifier_loss', 'kl_loss', 'reconstruction_loss'],
    patience=5,
    min_delta=0.01
)

true_words_soft = tf.one_hot(true_words, depth=VOCAB_SIZE).numpy()

if pretrain:
    # Train just the autoencoder for a while.
    vae.fit(true_words_soft, true_words_soft, epochs=20, batch_size=32,
            callbacks=[callback])
    vae.save('pre_adversarial_vae')
else:
    vae.load('pre_adversarial_vae')

vae.trainClassifier = True
# Now let them work adversarially.
vae.fit(true_words_soft, true_words_soft, epochs=50, batch_size=32,
        callbacks=[callback])

vae.save('final_vae')
