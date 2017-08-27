import sys
import gym
from gym import wrappers
from agent import Agent
import numpy as np


batch_size = 32
episodes = sys.argv[1] if len(sys.argv) > 1 else 5000
env_name = sys.argv[2] if len(sys.argv) > 2 else "Breakout-v0"

episodes = int(episodes)
env_name = env_name
D = 84 * 84
score = []
text_file = open("Output.txt", "w")

env = gym.make(env_name)

env = wrappers.Monitor(env, env_name, force=True)

agent = Agent(env.observation_space.shape, env.action_space.n)

for i_episodes in range(episodes):
    State = env.reset()
    state = agent.RGBprocess(State)
    state = agent.stack(state)
    totalreward = 0
    done = False
    while not done:
        #if i_episodes % 50 == 0:
            #env.render()
        action = agent.act(state)
        new_state, reward, done, info = env.step(action)
        new_state = agent.RGBprocess(new_state)
        new_state_dif = agent.stack(new_state)
        agent.remember(state, action, reward, new_state_dif, done)
        state = new_state_dif
        totalreward += reward
    agent.memory_replay(batch_size)
    if done:
        text_file.write("{} episode, score = {}\n".format(
            i_episodes + 1, totalreward))
        agent.save_model()
        score.append(totalreward)
        if i_episodes % 100 == 0:
            print("{} episode, score = {}, rolling mean 100 episodes = {}".format(
                i_episodes + 1, totalreward, np.mean(score[-100:])))
        if i_episodes % 40 == 0:
            agent.update_target_model()

env.close()
text_file.close()
gym.upload(env_name, api_key='sk_WRCITkqmTJKYB9hvBk5tPA')
