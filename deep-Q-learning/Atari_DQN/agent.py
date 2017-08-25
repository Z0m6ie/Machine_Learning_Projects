import numpy as np
import random
from PIL import Image
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
sizes = (80, 80, 4)


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=400000)
        self.gamma = 0.99   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.005  # exploration will not decay futher
        self.epsilon_decay = 0.0000398
        self.learning_rate = 0.0001
        self.loss = 0
        self.model = self._build_model()
        self.weight_backup = 'model_weights.h5'
        self.old_I_2 = None
        self.old_I_3 = None
        self.old_I_4 = None
        self.old_I_1 = None

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=8, subsample=(
            4, 4), activation='relu', padding='same', input_shape=sizes))
        model.add(Conv2D(64, kernel_size=4, subsample=(
            2, 2), activation='relu', padding='same'))
        model.add(Conv2D(64, kernel_size=3, subsample=(
            1, 1), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self):
        self.model.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def RGBprocess(self, raw_img):
        processed_observation = Image.fromarray(raw_img, 'RGB')
        processed_observation = processed_observation.convert('L')
        processed_observation = processed_observation.resize((80, 80))
        processed_observation = np.array(processed_observation)
        processed_observation = processed_observation.reshape(
            1, processed_observation.shape[0], processed_observation.shape[1])
        return processed_observation

    def stack(self, processed_observation):
        I_4 = self.old_I_3 if self.old_I_3 is not None else np.zeros(
            (1, 80, 80))
        I_3 = self.old_I_2 if self.old_I_2 is not None else np.zeros(
            (1, 80, 80))
        I_2 = self.old_I_1 if self.old_I_1 is not None else np.zeros(
            (1, 80, 80))
        I_1 = processed_observation
        processed_stack = np.stack((I_4, I_3, I_2, I_1), axis=3)
        self.old_I_4 = I_4
        self.old_I_3 = I_3
        self.old_I_2 = I_2
        self.old_I_1 = I_1
        return processed_stack

    def remember(self, state, action, reward, new_state, done):
        if len(self.memory) >= 400000:
            self.memory.popleft()
            self.memory.append([state, action, reward, new_state, done])
        else:
            self.memory.append([state, action, reward, new_state, done])

    def memory_replay(self, batch_size):
        if len(self.memory) < batch_size:
            pass
        else:
            Sample = random.sample(self.memory, batch_size)

            inputs = np.zeros(
                (len(Sample), Sample[0][0].shape[1], Sample[0][0].shape[2], Sample[0][0].shape[3]))
            targets = np.zeros((inputs.shape[0], self.action_size))

            for i in range(0, len(Sample)):
                sample_state = Sample[i][0]
                sample_action = Sample[i][1]
                sample_reward = Sample[i][2]
                sample_new_state = Sample[i][3]
                sample_done = Sample[i][4]

                inputs[i:i + 1] = sample_state

                targets[i] = self.model.predict(sample_state)
                future_reward = self.model.predict(sample_new_state)

                if sample_done:
                    targets[i, sample_action] = sample_reward
                else:
                    targets[i, sample_action] = sample_reward + \
                        self.gamma * np.max(future_reward)
            self.loss += self.model.train_on_batch(inputs, targets)
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
