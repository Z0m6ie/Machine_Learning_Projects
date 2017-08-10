import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten
from keras.optimizers import Adam

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # exploration will not decay futher
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=8, subsample=(4, 4), activation='relu', padding='same', input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        model.add(Conv2D(64, kernel_size=4, subsample=(2, 2), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=3, subsample=(1, 1), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.summary()
