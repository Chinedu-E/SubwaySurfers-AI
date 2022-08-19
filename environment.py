import pyautogui
import keyboard
import time
import cv2
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from utils import screenshot, read_coins
import train_endgame as eg


class SubwayEnv(Env):

    def __init__(self):
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(80, 120, 3), dtype=np.uint64)
        self.game_over_model = eg.build()
        self.game_over_model.load_weights('endgame1.h5')
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
        time.sleep(0.125)
        state = screenshot()
        # pyautogui.click(426, 198)
        reward, done = self.reward_function(state)
        info = {}

        return state, reward, done, info

    def reset(self):
        start_time = time.time()
        while True:
            pyautogui.click(358, 800, clicks=3, interval=1.0)
            if (time.time() - start_time) > 3:  # failsafe triggered after 4 seconds of resetting
                pyautogui.click(426, 198)
                time.sleep(0.5)
                pyautogui.click(358, 800, clicks=3, interval=1.0)

            state = screenshot()
            if self.is_game_over(state):  # clicking till we are back to game screen
                pass
            else:
                break
        return state

    def render(self):
        ...

    def reward_function(self, state):
        #coins = read_coins()
        done = self.is_game_over(state)
        if done:
            reward = -1
            return reward, done
        reward = 0.5

        #ep_diff = coins - self.coins
        #if ep_diff > 1:
        #    reward = -2
        #else:
        #    reward = 1
        #self.coins = coins

        return reward, done

    def is_game_over(self, state) -> bool:
        state = cv2.resize(state, (96, 96))
        state = np.float32(state)
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = np.expand_dims(state, -1)
        state = np.expand_dims(state, 0)
        pred = self.game_over_model.predict(state, verbose=0)[0][0]
        if pred < 0.001:
            return True
        else:
            return False
