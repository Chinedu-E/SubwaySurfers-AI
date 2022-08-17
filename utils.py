import numpy as np
import mss
from keras.layers import Layer
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import easyocr

READER = easyocr.Reader(['en'])


class NoisyDense(Layer):
    def __init__(self, units=32):
        super(NoisyDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        mu_init = tf.random_uniform_initializer(minval=-1 * 1 / np.power(input_shape[1], 0.5),
                                                maxval=1 * 1 / np.power(input_shape[1], 0.5))
        sigma_init = tf.constant_initializer(0.4 / np.power(input_shape[1], 0.5))

        self.w_mu = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=mu_init,
            trainable=True,
        )
        self.w_sigma = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=sigma_init,
            trainable=True,
        )

        self.b_mu = self.add_weight(
            shape=(self.units,),
            initializer=mu_init,
            trainable=True,
        )

        self.b_sigma = self.add_weight(
            shape=(self.units,),
            initializer=mu_init,
            trainable=True,
        )

    def call(self, inputs):
        p = self.sample_noise([inputs.shape[-1], 1])
        q = self.sample_noise([1, self.units])
        f_p = self.f(p)
        f_q = self.f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)

        w = self.w_mu + tf.multiply(self.w_sigma, w_epsilon)
        ret = tf.matmul(inputs, w)
        # bias
        b = self.b_mu + tf.multiply(self.b_sigma, b_epsilon)

        return ret + b

    @tf.function
    def sample_noise(self, shape):
        noise = tf.random.normal(shape)
        return noise

    @tf.function
    def f(self, x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))


def screenshot():
    mon = {'top': 150, 'left': 0, 'width': 400, 'height': 600}
    with mss.mss() as sct:
        state = np.array(sct.grab(mon))
    state = cv2.resize(state, (150, 150))

    return grayscale(state)


def grayscale(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGRA2GRAY)
    state = np.reshape(state, (150, 150, 1))
    state = state/255.
    return state


def read_coins() -> int:
    coin_portion = {'top': 75, 'left': 380, 'width': 80, 'height': 60}
    with mss.mss() as sct:
        img = np.array(sct.grab(coin_portion))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # rgb to hsv color space

    s_ch = hsv_img[:, :, 1]  # Get the saturation channel
    # Apply threshold - pixels above 5 are going to be 255, other are zeros.
    thesh = cv2.threshold(s_ch, 5, 255, cv2.THRESH_BINARY)[1]
    # Apply opening morphological operation for removing artifacts.
    thesh = cv2.morphologyEx(thesh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Fill the background in thesh with the value 128 (pixel in the foreground stays 0.
    cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=128, loDiff=1,
                  upDiff=1)
    # Set all the pixels where thesh=128 to black.
    img[thesh == 128] = (0, 0, 0)
    coins = READER.readtext(img)
    if len(coins) > 0:
        coins = int(coins[0][1])
    else:
        coins = 0
    return coins

