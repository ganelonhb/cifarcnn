# +---------------------------------------------+
# - Project:    CPSC 483 Final Exam             -
# - File:       cifarcnn.py                     -
# - Author:     Zachary Worcester               -
# - Email:      zworcester0@csu.fullerton.edu   -
# +---------------------------------------------+
# |             Project Description             |
# | An implementation of a CNN that classifies  |
# | images based on the CIFAR-10 Dataset.       |
# |                                             |
# |||||||||||||||||||||||||||||||||||||||||||||||

#                 PYLINT SECTION
# Disabled no-member because at least in the ver-
# sion of tensorflow I'm using, there is indeed
# a keras member.
#
# Disabled invalid-name because in this partic-
# ular case, capital X is more descriptive.
#
# Disabled too-many-instance-attributes because,
# Let's face it, there should not be this limit
# in the first place.
# ===============================================
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes

"""Implements the CIFARCNN class"""

import os
from io import BytesIO

import tensorflow as tf

from keras.losses import SparseCategoricalCrossentropy as SCC
from keras.metrics import SparseCategoricalAccuracy as SCA

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

class CIFARCNN():
    """A Class that trains and saves a CNN trained on CIFAR-10."""
    def __init__(self, epochs=20, model='training/'):
        """Initializes a CIFARCNN"""

        model_path, _ = os.path.split(model)
        os.makedirs(model_path, exist_ok=True)

        self.labels = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
            ]

        # 2) Sequential Model
        self.cnn = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=2,
                    padding='valid'
                    ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=2,
                    padding='valid'
                    ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=2,
                    padding='valid'
                    ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu', padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    padding='same'
                    ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(
                    pool_size=2,
                    strides=2,
                    padding='valid'
                    ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(
                    units=128,
                    activation='relu'
                    ),
                tf.keras.layers.Dense(
                    units=128,
                    activation='relu'
                    ),
                tf.keras.layers.Dense(
                    units=10,
                    activation='softmax'
                    ),
            ])
        self.cnn.compile(
            optimizer='adam',
            loss=SCC(from_logits=True),
            metrics=[SCA()]
            )

        if not os.path.isfile(os.path.join(f'{model_path}', 'checkpoint')):
            # 1) Data Preprocess
            (self._X_train, self._y_train), (self._X_test, self._y_test) \
                = tf.keras.datasets.cifar10.load_data()

            self._X_train, self._X_test \
                = self._X_train / 255., self._X_test / 255.
            self._y_train, self._y_test \
                = self._y_train.flatten(), self._y_test.flatten()

            # 3) Training
            self.train_history = self.cnn.fit(
                self._X_train,
                self._y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2
                )
            # 4) Evaluating
            self.evaluate_history = self.cnn.evaluate(self._X_test, self._y_test)

            self.cnn.save_weights(model)
        else:
            _, (self._X_test, self._y_test) = tf.keras.datasets.cifar10.load_data()

            self._X_test = self._X_test / 255.
            self._y_test = self._y_test.flatten()

            _,_ = self.cnn.evaluate(self._X_test, self._y_test)

            latest = tf.train.latest_checkpoint(model_path)
            self.cnn.load_weights(latest)

            self.train_history = None
            self.evaluate_history = self.cnn.evaluate(self._X_test, self._y_test)

        self._y_pred = np.argmax(self.cnn.predict(self._X_test), axis=1)
        self._confusion_matrix = tf.math.confusion_matrix(self._y_test, self._y_pred)

        if self.train_history is not None:
            self.plot_sca().save("SCA.png")
            self.plot_loss().save("Loss.png")

            print(f'Test Accuracy: {self.evaluate_history[1]}')
            print(f'y_pred: {self._y_pred}')
            print(f'Confusion Matrix:\n{self._confusion_matrix}')

    def make_prediction(self, image):
        """Make a prediction for a single image."""
        img = tf.keras.preprocessing.image.load_img(
            image,
            target_size=(32,32),
            interpolation='bicubic'
            )
        input_arr = tf.keras.utils.img_to_array(img)
        input_arr = np.array([input_arr])

        input_arr /= 255.

        pred = self.cnn.predict(input_arr)

        print(pred)

        return self.labels[pred.argmax()]



    def plot_loss(self, dot_color="red", line_color="blue"):
        """plot the loss"""
        plt.scatter(
            range(len(self.train_history.history['loss'])),
            self.train_history.history['loss'],
            color=dot_color
            )
        plt.plot(
            range(len(self.train_history.history['loss'])),
            self.train_history.history['loss'],
            color=line_color
            )
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')

        plt.cla()

        return Image.open(img_buf)

    def plot_sca(self, dot_color="red", line_color="blue"):
        """plot the sparse categorical accuracy"""
        plt.scatter(
            range(len(self.train_history.history['sparse_categorical_accuracy'])),
            self.train_history.history['sparse_categorical_accuracy'],
            color=dot_color
            )
        plt.plot(
            range(len(self.train_history.history['sparse_categorical_accuracy'])),
            self.train_history.history['sparse_categorical_accuracy'],
            color=line_color
            )
        plt.title('Sparse Categorical Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('SCA')

        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')

        plt.cla()

        return Image.open(img_buf)
