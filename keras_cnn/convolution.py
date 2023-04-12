import tensorflow as tf

def generate_model(input_image_shape):
    model = tf.keras.Sequential([
        # first convolutional layer
        # result is to learn 32 features/results
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_image_shape),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # second convolutional layer
        # result will learn 64 features/patterns
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # fully connected layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

n = 100  # Number of images in the batch
input_shape = (32, 32, 3)  # Shape of the tensor (height, width, channels)
model = generate_model(input_shape)

model.summary()

# Generate a random tensor and pass it through the network
random_tensor = tf.random.normal((n, *input_shape))
output = model(random_tensor)
print(output.shape)
print(output)
