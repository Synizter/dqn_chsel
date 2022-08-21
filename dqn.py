import test_unit
import silence_tensorflow
silence_tensorflow.silence_tensorflow()
import tensorflow as tf
from dqn_agent import DQNAgent
from memory import ExperienceReplay
import numpy as np

seed = 420
gamma = 0.99
epsilon = 1.0
decay_rate = 0.005
batch_size=  32
max_step_per_eps = 19
#TEST

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
loss = tf.keras.losses.Huber()

running_reward = 0
max_episode = 100000

env =  test_unit.TEST_ENV()
score_weigth = env.get_weitght()

num_action = env.action

q = DQNAgent(num_action, 128) # q predictor
q_target = DQNAgent(num_action, 128) # future reward predictor
buffer = ExperienceReplay(10000)


def predict_action(state):
    if np.random.rand(1)[0] < epsilon:
        return np.random.randint(num_action)
    else:
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = q(state_tensor, training = False)
        action = tf.argmax(action_prob, axis=1).numpy()
        return action

def expolration_decay():
    epsilon = epsilon * np.exp(-decay_rate)

for i in range(max_episode):
    state = env.state.copy()
    episode_reward = 0
    done = False
    while not done:
        action = predict_action(state)
        expolration_decay()
        
        _state, reward, done = env.step(action)
        episode_reward += reward
        buffer.store(state, action, reward, _state, done)
        
        if buffer.current > batch_size:
            state_sample, action_sample, reward_sample, next_state_sample, done_sample = buffer.sample(32)
            future_reward = q_target.predict(next_state_sample)
            updated_q_values = reward_sample + gamma



