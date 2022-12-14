
# Create environment
import numpy as np
from eeg_env import EEGChannelOptimze
import capilab_dataset2
import model_set
from sklearn.model_selection import train_test_split
from dqn_agent import DQNAgent
from tqdm import tqdm

#STFU
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

BASELINE_F1 = 0.40

# ENV--------------------------------------------------------------------------------------------------------------------------------------
# fname = ['Datasets\Lai_JulyData.mat', 'Datasets\Lai_JulyData.mat', 'Datasets\Suguro_JulyData.mat', 'Datasets\Takahashi_JulyData.mat']
fname = ['Datasets/Suguro_JulyData.mat']
dataset_channel_map = {'F4': 0, 'C4': 1, 'Pa': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}



if __name__ == "__main__":
    X, y = capilab_dataset2.get(fname)
    try:
        _x, x_test,  _y, y_test = train_test_split(X, y, test_size = .05, stratify = y, random_state = 420)
        x_train, x_val, y_train, y_val = train_test_split(_x, _y, test_size = .2, stratify = _y, random_state = 420)
        dataset_info = {
            "RawX":X,
            "RawY":y,
            "TrainX":x_train,
            "TrainY":y_train,
            "ValX":x_val,
            "ValY":y_val,
            "TestX":x_test,
            "TestY":y_test,
            "OutputClass":y.shape[1],
            "Shape":(X.shape[1], X.shape[2]),
            "NbrData":X.shape[0],
            "ChMap":dataset_channel_map
        }
        env = EEGChannelOptimze(dataset_info, model_set.Custom1DCNN, BASELINE_F1)
    except Exception as e:
        print(e)

    # AGNET-------------------------------------------------------------------------------------------------------
    n_episodes = 1500 
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
                    episode_reward = np.array(episode_reward)

                    # episode_chs.append(env.observation_space)
                    d = '{},{},{},{},{},{},{}\n'.format(e, 
                                                        BASELINE_F1 + r, 
                                                        env.state, 
                                                        agent.exploration_proba, 
                                                        env.reward_threshold, 
                                                        episode_reward,
                                                        np.mean(episode_reward))
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