
# Create environment
import numpy as np
from eeg_env import EEGChannelOptimze
import capilab_dataset2
import model_set
from sklearn.model_selection import train_test_split
from dqn_agent import DQNAgent
from tqdm import tqdm
import tensorflow as tf
#STFU
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

BASELINE_F1 = 0.40

# ENV--------------------------------------------------------------------------------------------------------------------------------------
# fname = ['Datasets\Lai_JulyData.mat', 'Datasets\Lai_JulyData.mat', 'Datasets\Suguro_JulyData.mat', 'Datasets\Takahashi_JulyData.mat']
fname = ['Datasets/Lai_JulyData.mat', 'Datasets/Takahashi_JulyData.mat']
dataset_channel_map = {'F4': 0, 'C4': 1, 'Pa': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}



if __name__ == "__main__":
    fname = ['Datasets/Takahashi_JulyData.mat']
    dataset_channel_map = {'F4': 0, 'C4': 1, 'P4': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}
    
    Xx, yy = capilab_dataset2.get(fname)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise ("NO GPU Available")
    try:
            # Restrict TensorFlow to only use the first GPU
           
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

        X, x_test,  y, y_test = train_test_split(Xx, yy, test_size = .1, stratify = yy, random_state = 420)
        dataset_info = {
            "X":X,
            "y":y,
            "x_test":x_test,
            "y_test":y_test,
            "nbr_class":y.shape[1],
            "data_shape":(X.shape[1], X.shape[2]),
            "nbr_data":X.shape[0],
            "ch_map":dataset_channel_map
        }
        env = EEGChannelOptimze(dataset_info, model_set.Custom1DCNN, 0.25)
    except Exception as e:
        print(e)

    # AGNET-------------------------------------------------------------------------------------------------------
    n_episodes = 1000 
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # TRAIN-------------------------------------------------------------------------------------------------------
    batch_size = agent.batch_size
    total_steps = 0
    with tqdm(total = n_episodes, position = 0, leave = True) as pbar:
        for e in tqdm(range(n_episodes), ncols = 100, position = 0, leave = True, desc ="DQN Training>"):
            
            
            current_state = np.array([env.reset()])
            episode_step = 0
            done = False
            r = 0
            episode_reward = []
            
            while not done:
                total_steps = total_steps + 1
                action = agent.compute_action(current_state)
                
                next_state, reward, done, _ = env.step(action)
                r+= reward #calculate reward increase/decrease from baseline
                episode_reward.append(reward)
                next_state = np.array([next_state])
                agent.store_episode(current_state, action, reward, next_state, done)
                if done:
                    agent.update_exploration_probability()
                
                    # episode_chs.append(env.observation_space)
                    d = '{},{},{},{},{},{}\n'.format(e, 
                                                        env.state, 
                                                        agent.exploration_proba, 
                                                        env.reward_threshold, 
                                                        episode_reward,
                                                        np.mean(np.array(episode_reward)))
                    
                    with open('log_agent_reward.txt', 'a+') as f:
                        f.write(d)
                    break
                current_state = next_state
                episode_step += 1

            if total_steps >= batch_size:
                loss = agent.train(batch_size=batch_size)
                # episode_loss.append(loss)
                with open('log_action_approximator.txt', 'a+') as f:
                    f.write(str(loss[-1]) + '\n')
            pbar.update()