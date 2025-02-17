import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple neural network model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Function to train a model with a given optimizer
def train_model(optimizer_name):
    model = create_model()
    optimizer = tf.keras.optimizers.Adam() if optimizer_name == "Adam" else tf.keras.optimizers.SGD()
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=2)
    return history

# Train with Adam
adam_history = train_model("Adam")

# Train with SGD
sgd_history = train_model("SGD")

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
plt.plot(adam_history.history['val_accuracy'], label='Adam Validation Accuracy')
plt.plot(sgd_history.history['val_accuracy'], label='SGD Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Adam vs SGD Optimizer Performance on MNIST')
plt.show()
