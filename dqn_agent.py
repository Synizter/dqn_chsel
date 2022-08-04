import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import mean_squared_error
from matplotlib import pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 1e-3
        self.gamma = 0.95
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.004
        self.batch_size = 16
        
        self.memory_buffer= list()
        self.max_memory_buffer = 1500
        
        self.model = Sequential([
            Flatten(),             
            Dense(units=64,activation = 'relu'),
            Dense(units=32,activation = 'relu'),
            Dense(self.n_actions, activation = 'softmax')
        ])
        self.model.compile(loss="mse", optimizer = SGD(learning_rate = self.lr))
        # self.model.build(state_size)
        # self.model.summary()
        
    def compute_action(self, current_state):
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        q_values = tf.reduce_sum(self.model.predict(current_state, verbose = False), axis = 0)
        return np.argmax(q_values)

    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)

    def store_episode(self,current_state, action, reward, next_state, done):
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
        #write memory buffer to file as backup
        __w = '{}, {}, {}, {}, {}\n'.format(current_state, action, reward, next_state, done)
        with open('log_memory_buffer.txt', 'a+') as f:
            f.write(__w)

    def resume_action_approx_training(self, model_path = 'approximator'):
        self.model = tf.keras.models.load_model(model_path)

    def train(self, batch_size = 32):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]
        for experience in batch_sample:
            q_current_state = self.model.predict(experience["current_state"], verbose = False)
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_predict = self.model.predict(experience["next_state"], verbose = False)
                q_predict = tf.reduce_sum(q_predict, axis = 0)
                q_target = q_target + self.gamma*np.max(q_predict)
            
            q_current_state[0][experience["action"]] = q_target
            # train the model
            hist = self.model.fit(experience["current_state"], q_current_state, verbose=False)
        self.model.save('approximator')
        return hist.history['loss']