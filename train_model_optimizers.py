import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Create a simple neural network model
def create_model(optimizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Train the models with Adam and SGD optimizers
adam_model = create_model(optimizer='adam')
sgd_model = create_model(optimizer='sgd')

# Train with Adam optimizer
history_adam = adam_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Train with SGD optimizer
history_sgd = sgd_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 4. Plot the accuracy comparison
plt.figure(figsize=(12, 6))

# Plot for Adam
plt.subplot(1, 2, 1)
plt.plot(history_adam.history['accuracy'], label='Training Accuracy (Adam)')
plt.plot(history_adam.history['val_accuracy'], label='Validation Accuracy (Adam)')
plt.title('Adam Optimizer Performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot for SGD
plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['accuracy'], label='Training Accuracy (SGD)')
plt.plot(history_sgd.history['val_accuracy'], label='Validation Accuracy (SGD)')
plt.title('SGD Optimizer Performance')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
