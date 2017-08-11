import sys
import gym
from gym import wrappers
from agent import agent

batch_size = 32
episodes = sys.argv[1] if len(sys.argv) > 1 else 5000
env_name = sys.argv[2] if len(sys.argv) > 2 else "Pong-v0"

env = gym.make(env_name)
env.reset()
env = wrappers.Monitor(env, env_name, force=True)
for i_episodes in range(episodes):
    state = env.reset()
    while (True):
        env.render()
        action = agent.act(state)
        new_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, new_state, done)
        state = new_state
    agent.memory_replay(batch_size)
    if done:
        print("Game finished after {} episodes".format(i_episodes + 1))
        agent.save_model()
        break

env.close()
