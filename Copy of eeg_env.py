import gym
import numpy as np
import capilab_dataset2
import model_set
import tensorflow as tf

MAX_CHANNELS_SELECT =  6

class EEGChannelOptimze(gym.Env):
    
    def __init__(self, dataset_info, classifier, initial_acc_thresh):
        super(EEGChannelOptimze, self).__init__()
        self.classifier = classifier
        self.dataset_info = dataset_info

        self.action_space = gym.spaces.Discrete(len(self.dataset_info['ChMap']))
        # self.observation_space = np.zeros((len(self.dataset_info['ChMap'])))
        self.observation_space = gym.spaces.MultiBinary(len(self.dataset_info['ChMap']))
        self.state = np.zeros((len(self.dataset_info['ChMap'])), dtype=int)
        #add reliable chanenl
        self.state[1] = 1 #c4
        self.state[3] = 1 #cz
        self.state[5] = 1 #c3
        self.rounds = MAX_CHANNELS_SELECT

        self.initial_acc_thresh = initial_acc_thresh
        self.reward_threshold = self.initial_acc_thresh
        #MODEL PARAMETERS
        ckpt = tf.keras.callbacks.ModelCheckpoint('tkhDatasetTest/',monitor='val_loss', verbose=False,
                                    save_best_only=True, save_weights_only=False, mode='auto',save_freq="epoch")
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True, verbose = False)
        
        self.cb_list = [ckpt, es]
        self.classifier_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
        self.classifier_loss = tf.keras.losses.CategoricalCrossentropy()


    def step(self, action, verbose = False):
        self.rounds -= 1

        done = False
        info = {}
        reward = 0
        
        self.state[action] = 1 #select the channel
        
        # print("Take action {} and the current state is {}".format(action, self.state))
        #extracted target channel
        target_ch = np.where(self.state == 1)[0] 

        x_train = self.dataset_info['TrainX'][:,:,target_ch]
        x_val = self.dataset_info['ValX'][:,:,target_ch]

        clf = self.classifier(inp_shape=self.dataset_info['Shape'], output_classes=self.dataset_info['OutputClass'])
        clf.compile(optimizer = self.classifier_optimizer, loss=self.classifier_loss , metrics=['accuracy'])

        clf.fit(x_train, self.dataset_info['TrainY'], batch_size=16, epochs = 80, verbose = verbose, validation_data = (x_val,self.dataset_info['ValY']), callbacks = self.cb_list)
        _, val_acc =  clf.evaluate(x_val,self.dataset_info['ValY'], verbose = verbose)
        reward = val_acc - self.reward_threshold
        
        #if newly added channel yield higher accuracy ,change reward threshhold and have reward +1
        if reward > 0:
            self.reward_threshold = val_acc
        
        if len(np.where(self.observation_space == 1)[0]) == 8 or self.rounds == 0 or reward < 0:
                done = True
        
        # print("Done taking an action, val_acc is {}, reward is {} and episode end is {}".format(val_acc,reward, done))

        return self.state, reward, done, info
    
    def reset(self):
        # print("Reset env")
        # self.observation_space = np.zeros((19,))
        self.state = np.zeros((len(self.dataset_info['ChMap'])), dtype=int)\
        #add reliable chanenl
        self.state[1] = 1 #c4
        self.state[3] = 1 #cz
        self.state[5] = 1 #c3
        self.rounds = MAX_CHANNELS_SELECT
        self.reward_threshold = self.initial_acc_thresh
        return self.state



if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    # fname = ['Datasets\Lai_JulyData.mat', 'Datasets\Lai_JulyData.mat', 'Datasets\Suguro_JulyData.mat', 'Datasets\Takahashi_JulyData.mat']
    fname = ['Datasets/Suguro_JulyData.mat']
    dataset_channel_map = {0: 'F4', 1: 'C4', 2: 'Pa', 3: 'Cz', 4: 'F3', 5: 'C3', 6: 'P3', 7: 'F7', 8: 'T3', 
                                    9: 'T5', 10: 'Fp1', 11: 'Fp2', 12: 'T4', 13: 'F8', 14: 'Fz', 15: 'Pz', 16: 'T6', 17: 'O2', 18: 'O1'}
    
    X, y = capilab_dataset2.get(fname)
    try:
        _x, x_test,  _y, y_test = train_test_split(X, y, test_size = .1, stratify = y, random_state = 420)
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
        env = EEGChannelOptimze(dataset_info, model_set.Custom1DCNN, 0.35)
        # print(env.action_space.n, env.observation_space.n)
        print(env.state)
        #check compatible
        # from gym.utils.env_checker import check_env
        # # check_env(env)

        # x, r, d, _ = env.step(env.action_space.sample())
        # print(env.reward_threshold, r, d)

    except Exception as e:
        print(e)





    