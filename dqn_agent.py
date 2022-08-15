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
        self.n_actions = action_size #nbr of action
        self.lr = 1e-3 #Learning rate for model
        self.gamma = 0.95 #discount factor
        self.exploration_proba = 1.0 #Coolant factor
        self.exploration_proba_decay = 0.005 #Decay Rate
        self.batch_size = 32 #barch size for memory sampling
        
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
            #predict Q (reward) of current state eg 
            #input = [[1,1,0,0,0,1,1,0,1, 0, 0, 1, 0 ,1, 1, 0, 0, 0, 1]] | output = [[0.0652849  0.02962924 0.04938378 0.04408209 0.07093532 
            #                                                                        0.047186980.04342251 0.05206998 0.03284323 0.06921446 0.0743172  0.06933404
            #                                                                        0.06899458 0.03939189 0.06939222 0.03298171 0.03278639 0.0538005 0.05494894]]
            q_target = experience["reward"]
            if not experience["done"]:
                q_predict = self.model.predict(experience["next_state"], verbose = False) #predict q of next state
                q_predict = tf.reduce_sum(q_predict, axis = 0) #flatten output
                q_target = q_target + self.gamma*np.max(q_predict) #apply discount factor
            
            q_current_state[0][experience["action"]] = q_target #set a q value of an action (predicted  by network) to q target
            #[0] because output of network is array of array
        
            # train the model
            hist = self.model.fit(experience["current_state"], q_current_state, verbose=False)
        self.model.save('approximator')
        return hist.history['loss']

if __name__ == "__main__":

    t = {"current_state":[[1,1,0,0,0,1,1,0,1, 0, 0, 1, 0 ,1, 1, 0, 0, 0, 1]],
         "action":11,
         "reward":-0.07506811324897217,
         "next_state":[[1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1]],
         "done":False}
    agent = DQNAgent(19, 19)
    q_current_state = agent.model.predict(t['current_state'])
    q_target = t['reward']
    print(q_target)
    # print(q_current_state, q_current_state[0])
    if not t['done']:
        q_predict = agent.model.predict(t["next_state"], verbose = False)
        print(q_predict)
        q_predict = tf.reduce_sum(q_predict, axis = 0)
        print(q_predict)
        q_target = q_target + agent.gamma*np.max(q_predict)
        print(q_target)
        
        
        

