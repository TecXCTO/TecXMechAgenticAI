import autokeras as ak
import tensorflow as tf

# Load a dataset (e.g., MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Initialize an ImageClassifier which automatically searches for the best NN code
clf = ak.ImageClassifier(max_trials=3) # max_trials: number of models to test

# Start the search process
clf.fit(x_train, y_train, epochs=2)

# Export the automatically generated model code/architecture
model = clf.export_model()
model.summary()

#####$$#


import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def generate_random_architecture():
    """Randomly generates a list representing layers and neurons."""
    num_layers = random.randint(1, 5)
    return [random.randint(8, 128) for _ in range(num_layers)]

def build_model_from_code(architecture):
    """Builds a Keras model based on the generated architecture list."""
    model = Sequential()
    model.add(Dense(architecture[0], activation='relu', input_shape=(10,))) # Example input
    for neurons in architecture[1:]:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Automatically generate 5 different network structures
for i in range(5):
    arch_code = generate_random_architecture()
    print(f"Trial {i+1}: Generating NN with layers: {arch_code}")
    model = build_model_from_code(arch_code)
