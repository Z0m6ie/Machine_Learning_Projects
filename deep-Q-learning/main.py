import gym
from gym import wrappers

env = gym.make('Pong-v0')
env.reset()
env = wrappers.Monitor(env, 'Pong-v0', force=True)
for i_episodes in range():# TODO: )
    state = env.reset()
    while (True):
        env.render()
        action = agent choose action
        new_state, reward, done, info = env.step(action)
        store in list(state, action, reward, new_state, done)
        state = new_state
        if done:
            print("Game finished after {} episodes".format(i_episode+1))
            break
    #Carry out mini-batch update on neural network

env.close()
