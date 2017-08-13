import sys
import gym
from gym import wrappers
from agent import Agent
import numpy as np

batch_size = 32
episodes = sys.argv[1] if len(sys.argv) > 1 else 2000
episodes = int(episodes)
#env_name = sys.argv[2] if len(sys.argv) > 2 else "Pong-v0"

env_name = "CartPole-v0"

env = gym.make(env_name)

env = wrappers.Monitor(env, env_name, force=True)

agent = Agent(env.observation_space.shape[0], env.action_space.n)

for i_episodes in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    index = 0
    done = False
    while not done:
        env.render()
        action = agent.act(state)
        new_state, reward, done, info = env.step(action)
        new_state = np.reshape(new_state, [1, env.observation_space.shape[0]])
        agent.remember(state, action, reward, new_state, done)
        state = new_state
        index += 1
    agent.memory_replay(batch_size)
    if done:
        print("{} episode, score = {} ".format(i_episodes + 1, index + 1))
        agent.save_model()

env.close()
