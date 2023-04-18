# Import Tensorflow 2.0
import tensorflow as tf

# Download and import the MIT Introduction to Deep Learning package
import mitdeeplearning as mdl

# Import all remaining packages
import os
import time
import functools
import numpy as np
from tqdm import tqdm
import soundfile as sf
from IPython import display as ipythondisplay
#####################################################################
# Download the dataset
songs = mdl.lab1.load_training_data()

# Print one of the songs to inspect it in greater detail!
example_song = songs[0]
print("\nExample song: ")
print(example_song)

# Convert the ABC notation to audio file and listen to it
mdl.lab1.play_song(example_song)

# Join our list of song strings into a single string containing all songs
songs_joined = "\n\n".join(songs)

# Find all unique characters in the joined string
vocab = sorted(set(songs_joined))
print("There are", len(vocab), "unique characters in the dataset")
#####################################################################
# vectorize text & Define numerical representation of text
# Create a mapping from character to unique index.
# For example, to get the index of the character "d",
#   we can evaluate `char2idx["d"]`.
char2idx = {u: i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. This is
#   the inverse of char2idx and allows us to convert back
#   from unique index to the character in our vocabulary.
idx2char = np.array(vocab)

# This gives us an integer representation for each character.
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


def vectorize_string(string):
    '''
    function to convert the all songs string to a vectorized
    (i.e., numeric) representation. Use the appropriate mapping
    above to convert from vocab characters to the corresponding indices.

    the output of the `vectorize_string` function 
    should be a np.array with `N` elements, where `N` is
    the number of characters in the input string
    '''
    vectorized_output = np.array([char2idx[char] for char in string])

    return vectorized_output


# Test the vectorize_string function
vectorized_songs = vectorize_string(songs_joined)
print('{} ---- characters mapped to int ----> {}'.format(
    repr(songs_joined[:10]), vectorized_songs[:10]))

# check that vectorized_songs is a numpy array
assert isinstance(vectorized_songs,
                  np.ndarray), "returned result should be a numpy array"

# Each input sequence that we feed into our RNN
# will contain seq_length characters from the text.
# To do this, we'll break the text into chunks of seq_length+1.
# Suppose seq_length is 4 and our text is "Hello".
# Then, our input sequence is "Hell" and the target sequence is "ello".

### Batch definition to create training examples ###


def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)

    '''construct a list of input sequences for the training batch'''

    input_batch = [vectorized_songs[i: i+seq_length] for i in idx]

    '''construct a list of output sequences for the training batch'''
    output_batch = [vectorized_songs[i+1: i+seq_length+1] for i in idx]

    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])

    return x_batch, y_batch


# Perform some simple tests to make sure your batch function is working properly!
test_args = (vectorized_songs, 10, 2)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
   not mdl.lab1.test_batch_func_next_step(get_batch, test_args):
    print("======\n[FAIL] could not pass tests")
else:
    print("======\n[PASS] passed all tests!")

# visualize the batch function
x_batch, y_batch = get_batch(vectorized_songs, seq_length=5, batch_size=1)

for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
    print("Step {:3d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(
        target_idx, repr(idx2char[target_idx])))

# Define the RNN model


def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )

### Defining the RNN Model ###
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    '''Add LSTM and Dense layers to define the RNN model using the Sequential API.'''
    model = tf.keras.Sequential([
        # Layer 1: Embedding layer to transform indices into dense vectors
        #   of a fixed embedding size
        tf.keras.layers.Embedding(
            vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

        # Layer 2: LSTM with `rnn_units` number of units.
        # Call the LSTM function defined above to add this layer.
        LSTM(rnn_units),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size.
        # Add the Dense layer.
        tf.keras.layers.Dense(vocab_size)
    ])

    return model


# Define model parameters
model = build_model(len(vocab), embedding_dim=256,
                    rnn_units=1024, batch_size=32)

# Display the model architecture
model.summary()

# test the model
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:     ",
      x.shape,
      "(batch_size, sequence_length)"
      )

print("Prediction shape:",
      pred.shape,
      "(batch_size, sequence_length, vocab_size)"
      )

# tsting prediction from untrained model
sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
sampled_indices

# Decode the text
print("Input: \n", repr("".join(idx2char[x[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

#########################################################################################
# Training the model
# Defining the loss function


def compute_loss(labels, logits):
    '''define the loss function to compute and return the loss between
    the true labels and predictions (logits). Set the argument from_logits=True.'''
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)

    return loss


'''compute the loss using the true next characters from the example batch 
    and the predictions from the untrained model several cells above'''
example_batch_loss = compute_loss(y, pred)

print("Prediction shape: ", pred.shape,
      " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

# instantiate a new model for training using the `build_model`
# function and the hyperparameters created above.'''
model = build_model(len(vocab), embedding_dim=256,
                    rnn_units=1024, batch_size=4)


# TODO: instantiate an optimizer with its learning rate.
# Checkout the tensorflow website for a list of supported optimizers.
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/
# Try using the Adam optimizer to start.'''

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


@tf.function
def train_step(x, y):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:

        # feed the current input into the model and generate predictions'''
        y_hat = model(x)

        # compute the loss!'''
        loss = compute_loss(y, y_hat)

    # Now, compute the gradients
    # complete the function call for gradient computation.
    # Remember that we want the gradient of the loss with respect all
    # of the model parameters.
    # HINT: use `model.trainable_variables` to get a list of all model
    # parameters.'''
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


#########################################################################################
### Hyperparameter setting and optimization ###
# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters:
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

# Checkpoint location:
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

#########################################################################################
# Begin training!#
#########################################################################################
history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'):
    tqdm._instances.clear()  # clear if it exists

for iter in tqdm(range(num_training_iterations)):

    # Grab a batch and propagate it through the network
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
    loss = train_step(x_batch, y_batch)

    # Update the progress bar
    history.append(loss.numpy().mean())
    plotter.plot(history)

    # Update the model with the changed weights!
    if iter % 100 == 0:
        model.save_weights(checkpoint_prefix)

# Save the trained model and the weights
model.save_weights(checkpoint_prefix)
#########################################################################################
# reload the model if needed
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
#########################################################################################
### Prediction of a generated song ###
def generate_text(model, start_string, generation_length=1000):
    # Evaluation step (generating ABC text using the learned RNN model)

    # Convert the start string to numbers (vectorize)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        # Evaluate the inputs and generate the next character predictions
        predictions = model(input_eval)

        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Use a multinomial distribution to sample
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the prediction along with the previous hidden state
        # as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        # Add the predicted character to the generated text
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# ABC files start with "X" - this may be a good start string
generated_text = generate_text(model, start_string="X", generation_length=1000)
print(generated_text)

#########################################################################################
### Play back generated songs ###
generated_songs = mdl.lab1.extract_song_snippet(generated_text)
print(f"Extracted {len(generated_songs)} song snippets from the generated text.")

# Save the generated songs to a file
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the generated songs to a file
def song_to_waveform(song, bpm=120):
    note_duration = 60 / bpm / 4
    song_waveform = np.array([])
    print(f'Song: {song}')
    for j, note in enumerate(song):  # Add enumerate to get the index of the note
        print(f'Note (iteration {j}): {note}')  # Add this line to inspect the note variable
        print(f'Note type: {type(note)}')  # Print the type of the note variable
        if isinstance(note, tuple) and len(note) >= 2:  # Check if note is a tuple with at least 2 elements
            freq = note_to_freq(note[0])  # Use the custom note_to_freq function
            duration = note[1] * note_duration
            if freq != 0:
                sine_wave = mdl.lab1.generate_sine_wave(freq, duration, amplitude=0.3)
            else:
                sine_wave = np.zeros(duration * sample_rate)
            song_waveform = np.concatenate((song_waveform, sine_wave), axis=None)
        else:
            print(f"Skipping invalid note (iteration {j}): {note}")
    return song_waveform

# Define a dictionary that maps from note names to frequencies   
_NOTE_FREQUENCIES = {
    'c': 261.63,
    'c#': 277.18,
    'd': 293.66,
    'd#': 311.13,
    'e': 329.63,
    'f': 349.23,
    'f#': 369.99,
    'g': 392.00,
    'g#': 415.30,
    'a': 440.00,
    'a#': 466.16,
    'b': 493.88,
}

# Define a function that converts a note name to a frequency
def note_to_freq(note):
    if note.lower() in _NOTE_FREQUENCIES:
        return _NOTE_FREQUENCIES[note.lower()]
    else:
        return 0

for i, song in enumerate(generated_songs):
    # Synthesize the waveform from a song
    waveform = song_to_waveform(song)

    # If it's a valid song (correct syntax), save it as a .wav file
    if waveform is not None:
        filename = os.path.join(output_dir, f"generated_song_{i}.wav")
        sf.write(filename, waveform, samplerate=22050)
        print(f"Generated song {i} saved as {filename}")