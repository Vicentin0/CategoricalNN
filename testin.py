from Categorical_model import model, np

import tensorflow as tf
import os

TRAINLIMIT = 5000
NUM_ITERATIONS = 10000
LEARNING_RATE = 0.1

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Flatten images

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# One-hot encode labels
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

y_train_one_hot = one_hot_encode(y_train[:TRAINLIMIT], 10)
y_test_one_hot = one_hot_encode(y_test, 10)

input_neurons_num = 28 * 28
outputs_neurons_num = 10
hiddens_neurons_num = 16

nn = model(input_neurons_num,hiddens_neurons_num,outputs_neurons_num)
parameters = nn.train(x_train[:TRAINLIMIT], y_train_one_hot[:TRAINLIMIT], num_iterations=NUM_ITERATIONS, learning_rate=LEARNING_RATE)
predictions = nn.predict(x_test[10:20])

os.system("cls")
print(f"Model loss: {nn.loss}")
print(f"Trained with: {TRAINLIMIT}")
print(f"Prediction: {predictions}")
print(f"Results   : {np.argmax(y_test_one_hot[10:20], axis=1)}")  # Convert one-hot back to original labels
