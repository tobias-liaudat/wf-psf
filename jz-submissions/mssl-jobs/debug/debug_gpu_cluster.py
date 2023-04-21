
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# PRint tf info
print('tf.__version__: ', tf.__version__)
print('tf.test.is_built_with_cuda(): ', tf.test.is_built_with_cuda())
print('tf.config.list_physical_devices(GPU): ', tf.config.list_physical_devices('GPU'))

# Load data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


# scaling image values between 0-1
X_train_scaled = X_train/255
X_test_scaled = X_test/255
# one hot encoding labels
y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')


def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')    
    ])
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

model = get_model()
model.summary()


time_1 = time.time()
model = get_model()
model.fit(X_train_scaled, y_train_encoded, epochs = 3)
time_2 = time.time()
print('Training time (s): ', time_2 - time_1)


time_cpu_1 = time.time()
with tf.device('/CPU:0'):
    model_cpu = get_model()
    model_cpu.fit(X_train_scaled, y_train_encoded, epochs = 3)
time_cpu_2 = time.time()

print('CPU training time (s): ', time_cpu_2 - time_cpu_1)


time_gpu_1 = time.time()
with tf.device('/GPU:0'):
    model_gpu = get_model()
    model_gpu.fit(X_train_scaled, y_train_encoded, epochs = 3)
time_gpu_2 = time.time()

print('CPU training time (s): ', time_gpu_2 - time_gpu_1)


# # loading dataset
# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# # checking shape

# print(train_images.shape)

# print(train_labels[0])