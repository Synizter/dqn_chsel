import gym
import numpy as np
import capilab_dataset2
import model_set
import tensorflow as tf

#matrix eval
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

MAX_CHANNELS_SELECT =  8

class EEGChannelOptimze(gym.Env):
    
    def __init__(self, dataset_info, classifier, initial_acc_thresh, checkpoint_path = 'classifier_ckpt'):
        super(EEGChannelOptimze, self).__init__()
        self.classifier = classifier
        self.dataset_info = dataset_info
        self.checkpoint_path = checkpoint_path
        self.action_space = gym.spaces.Discrete(len(self.dataset_info['ChMap']))

        self.observation_space = gym.spaces.MultiBinary(len(self.dataset_info['ChMap']))
        self.initial_acc_thresh = initial_acc_thresh
        
        self.reset()
        # #MODEL PARAMETERS
        # ckpt = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path ,monitor='val_loss', verbose=False,
        #                             save_best_only=True, save_weights_only=False, mode='auto',save_freq="epoch")
        # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True, verbose = False)    
        # self.cb_list = [ckpt, es]
    


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
        y_train = self.dataset_info['TrainY']
        x_val = self.dataset_info['ValX'][:,:,target_ch]
        y_val = self.dataset_info['ValY']
        x_test = self.dataset_info['TestX'][:,:,target_ch]
        y_test = self.dataset_info['TestY']

        classifier_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
        classifier_loss = tf.keras.losses.CategoricalCrossentropy()
        clf = self.classifier(inp_shape=self.dataset_info['Shape'], output_classes=self.dataset_info['OutputClass'])
        clf.compile(optimizer = classifier_optimizer, loss= classifier_loss , metrics=['accuracy'])
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=9, restore_best_weights=True, verbose = False) 
        
        clf.fit(x_train, y_train,
                batch_size=32, 
                epochs = 50, 
                verbose = verbose, 
                validation_data = (x_val,y_val),
                callbacks = [es])

        # loss, val_acc =  clf.evaluate(x_val,y_val, verbose = verbose)

        y_preds = clf.predict(x_test, verbose = verbose)
        predicted = np.argmax(y_preds, axis=1)
        ground_truth = np.argmax(y_test, axis=1)
        
        acc = accuracy_score(ground_truth, predicted)
        reward = acc - self.reward_threshold
        #if newly added channel yield higher accuracy ,change reward threshhold and have reward +1
        if reward > 0:
            self.reward_threshold = acc
        
        if len(np.where(self.state == 1)[0]) == MAX_CHANNELS_SELECT + 2 or self.rounds == 0:
                done = True

        # tf.keras.models.save_model(clf, 'test/')
        
        #cleanup resource and clear tf session
        # del clf, classifier_loss, classifier_optimizer, x_train, x_val, val_acc, reward
        tf.keras.backend.clear_session()
        

        return self.state, reward, done, info
    
    def reset(self):
        # print("Reset env")
        # self.observation_space = np.zeros((19,))
        self.state = np.zeros((len(self.dataset_info['ChMap'])), dtype=int)
        #add reliable chanenl
        # self.state[self.dataset_info['ChMap']['Cz']] = 1
        self.state[self.dataset_info['ChMap']['C3']] = 1
        self.state[self.dataset_info['ChMap']['C4']] = 1
        self.rounds = 19
        self.reward_threshold = self.initial_acc_thresh
        return self.state



if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    # fname = ['Datasets\Lai_JulyData.mat', 'Datasets\Lai_JulyData.mat', 'Datasets\Suguro_JulyData.mat', 'Datasets\Takahashi_JulyData.mat']
    fname = ['Datasets/Lai_JulyData.mat', 'Datasets/Takahashi_JulyData.mat', 'Datasets/Lai_JulyData.mat']
    dataset_channel_map = {'F4': 0, 'C4': 1, 'Pa': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}
    
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
        #check compatible
        # from gym.utils.env_checker import check_env
        # # check_env(env)
        
        x, r, d, _ = env.step(dataset_info['ChMap']['Cz'], verbose = True)
        print(env.reward_threshold, r, d, env.state)


    except Exception as e:
        print(e)





    