import pyautogui
import keyboard
import time
import cv2
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from utils import screenshot, read_coins
from keras.models import load_model


class SubwayEnv(Env):

    def __init__(self):
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(150, 150, 1), dtype=np.uint64)
        self.game_over_model = load_model('endgame1.h5')
        self.coins = 0

    def step(self, action):
        action_map = {0: 'up',
                      1: 'down',
                      2: 'left',
                      3: 'right',
                      4: 'nothing'}
        action = action_map[action]

        if action != 'nothing':
            keyboard.press_and_release(action)

        time.sleep(4 / 15)
        state = screenshot()
        pyautogui.click(426, 198)
        reward, done = self.reward_function(state)
        info = {}

        return state, reward, done, info

    def reset(self):
        # TO-DO
        ...

    def render(self):
        ...

    def start(self):
        ...

    def reward_function(self, state):
        coins = read_coins()
        done = self.is_game_over(state)
        if done:
            reward = -15
            return reward, done

        ep_diff = coins - self.coins
        if ep_diff > 1:
            reward = 2
        else:
            reward = -1

        self.coins = coins

        return reward, done

    def is_game_over(self, state) -> bool:
        state = cv2.resize(state, (96, 96))
        state = np.expand_dims(state, -1)
        classes = ['end', 'in-game']
        pred = self.game_over_model.predict(state, verbose=0)[0]
        pred = classes[np.argmax(pred, axis=-1)]
        if pred == 'end':
            return True
        else:
            return False
