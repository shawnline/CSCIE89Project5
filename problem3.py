import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from keras.layers import LSTM

# Load and preprocess the data
max_features = 10000  # Top 10,000 words
maxlen = 200  # First 200 words of each review

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post', truncating='post')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')

# Remove padding characters by trimming sequences after they are encoded and decoded
def trim_padding(sequence):
    return [word for word in sequence if word != 0]

# Define an autoencoder
embedding_dim = 128  # Embedding dimension
latent_dim = 64  # Number of latent dimensions (experiment with different values)

# Encoder
input_review = Input(shape=(maxlen,))
embedded = Embedding(input_dim=max_features, output_dim=embedding_dim)(input_review)
encoded = LSTM(latent_dim)(embedded)

# Decoder
decoder_lstm = LSTM(embedding_dim, return_sequences=True)(encoded)
decoded_output = TimeDistributed(Dense(max_features, activation='softmax'))(decoder_lstm)

# Autoencoder model
autoencoder = Model(input_review, decoded_output)
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train,
                          epochs=10,
                          batch_size=64,
                          validation_split=0.2)

# Plotting training and validation loss trends
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the word index for conversion from numerical indices to text
word_index = imdb.get_word_index()
index_word = {index + 3: word for word, index in word_index.items()}  # Shift by 3 due to special tokens
index_word[0] = '<PAD>'
index_word[1] = '<START>'
index_word[2] = '<UNK>'
index_word[3] = '<UNUSED>'

# Function to convert numerical sequences to text
def sequence_to_text(sequence):
    return ' '.join([index_word.get(i, '?') for i in sequence])

# Select five random samples from the test set and visualize their reconstruction
num_samples = 5
samples = np.random.choice(len(X_test), num_samples, replace=False)

for i in samples:
    original = X_test[i]
    reconstructed = autoencoder.predict(np.expand_dims(original, axis=0))
    reconstructed_sequence = np.argmax(reconstructed, axis=-1).flatten()

    # Trim padding from original and reconstructed sequences
    original_trimmed = trim_padding(original)
    reconstructed_trimmed = trim_padding(reconstructed_sequence)

    print("Original Review:")
    print(sequence_to_text(original_trimmed))
    print("\nReconstructed Review:")
    print(sequence_to_text(reconstructed_trimmed))
    print("\n" + "="*50 + "\n")

# The smallest number of codings (latent_dim) that retained semantic integrity can be varied by experimenting.
# You can start with 64 (as done here) and reduce it, e.g., to 32 or 16, to see how the reconstructions degrade.
