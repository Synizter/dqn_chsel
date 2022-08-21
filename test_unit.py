import numpy as np
import silence_tensorflow
silence_tensorflow.silence_tensorflow()
import gym
import tensorflow as tf
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
class TEST_ENV(gym.Env):
    def __init__(self):
        self.reset()
        self.weigth_reward = np.random.random((19))
        self.max_ch = 9
        self.action_space = gym.spaces.Discrete(19)
        self.observation_space = gym.spaces.MultiBinary(19)
        
    def step(self, action):
        done = False
        info = None
        self.state[action] = 1
        reward = np.sum(self.state * self.weigth_reward)
        if len(np.where(self.state == 1)[0]) > self.max_ch or self.rounds == 0:
            done = True
        self.rounds -= 1
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros((19), dtype=np.int32)
        self.state[1] = 1
        self.state[5] = 1

        self.rounds = 9
        self.reward_threshold = 0.5
        return self.state

    def get_weitght(self):
        return self.weigth_reward


if __name__ =="__main__":
    env = TEST_ENV()
    
    highest_reward = np.sum(env.get_weitght())
    print(highest_reward, env.get_weitght(), env.state * env.get_weitght(), highest_reward * 90 / 100)