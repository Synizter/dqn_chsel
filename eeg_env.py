import gym
import numpy as np
import capilab_dataset2
import model_set
import tensorflow as tf

#shut up
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#matrix eval
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

MAX_CHANNELS_SELECT =  6
DEBUG = False

class EEGChannelOptimze(gym.Env):
    
    def __init__(self, dataset_info, classifier, initial_acc_thresh, checkpoint_path = 'classifier_ckpt'):
        super(EEGChannelOptimze, self).__init__()
        self.classifier = classifier
        self.dataset_info = dataset_info
        self.checkpoint_path = checkpoint_path
        self.action_space = gym.spaces.Discrete(len(self.dataset_info['ch_map']))
        self.observation_space = gym.spaces.MultiBinary(len(self.dataset_info['ch_map']))
        
        self.initial_acc_thresh = initial_acc_thresh
        self.reset()    

    def fold_cv_step(self, X, y, x_test, y_test, k = 10, verbose = False):
        f1 = []
        recall = []
        prec = []
        acc = []
        # print("\nSTATE {}".format(self.state))
        kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = 420)
        if not DEBUG:
            for i , (train, val) in enumerate(kfold.split(X, np.argmax(y, axis = 1))):
                # print("Training on fold {}/{}".format(i+1, k))
                classifier_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
                classifier_loss = tf.keras.losses.CategoricalCrossentropy()
                clf = self.classifier(inp_shape=self.dataset_info['data_shape'], output_classes=self.dataset_info['nbr_class'])
                clf.compile(optimizer = classifier_optimizer, loss= classifier_loss , metrics=['accuracy'])
                es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=9, restore_best_weights=True, verbose = False) 
                
                clf.fit(X[train], y[train],
                        batch_size=32, 
                        epochs = 50, 
                        verbose = verbose, 
                        validation_data = (X[val],y[val]),
                        callbacks = [es])
                y_preds = clf.predict(x_test, verbose = verbose)
                predicted = np.argmax(y_preds, axis=1)
                ground_truth = np.argmax(y_test, axis=1)
                
                acc.append(accuracy_score(ground_truth, predicted))
                f1.append(f1_score(ground_truth, predicted, average = 'macro'))
                prec.append(precision_score(ground_truth, predicted, average = 'macro'))
                recall.append(recall_score(ground_truth, predicted, average = 'macro'))
                # print("F1 {:.5f} | PRECISION {:.5f} | RECALL {:.5f} | ACC {:.5f}".format(f1[-1], prec[-1], recall[-1], acc[-1]))

                tf.keras.backend.clear_session()
                
            # print(np.array(acc).mean(), np.array(recall).mean(), np.array(prec).mean(),np.array(f1).mean())
        else:
            return 1.0, 1.0, 1.0, 1.0
        return np.array(acc).mean(), np.array(recall).mean(), np.array(prec).mean(),np.array(f1).mean()
        
            
    def step(self, action, verbose = False, k = 10):
        self.rounds -= 1

        done = False
        info = {}
        reward = 0
        
        self.state[action] = 1 #select the channel
        target_ch = np.where(self.state == 1)[0] 

        X = self.dataset_info['X'][:,:,target_ch]
        y = self.dataset_info['y']
        x_test = self.dataset_info['x_test'][:,:,target_ch]
        y_test = self.dataset_info['y_test']
        
        acc_m, rc_m, psc_m, f1_m = self.fold_cv_step(X,y, x_test, y_test, verbose=verbose)
        eval_crit = acc_m
        reward = eval_crit - self.reward_threshold
        #if newly added channel yield higher accuracy ,change reward threshhold and have reward +1
        if reward > 0:
            self.reward_threshold = eval_crit
        
        if len(np.where(self.state == 1)[0]) == MAX_CHANNELS_SELECT + 2 or self.rounds == 0:
                done = True

        return self.state, reward, done, info
    
    def reset(self):
        # print("Reset env")
        # self.observation_space = np.zeros((19,))
        self.state = np.zeros((len(self.dataset_info['ch_map'])), dtype=int)
        #add reliable chanenl
        # self.state[self.dataset_info['ChMap']['Cz']] = 1
        self.state[self.dataset_info['ch_map']['C3']] = 1
        self.state[self.dataset_info['ch_map']['C4']] = 1

        self.rounds = 6
        self.reward_threshold = self.initial_acc_thresh
        return self.state


if __name__ == "__main__":
    import sys, os
    import traceback

    from sklearn.model_selection import train_test_split

    # fname = ['Datasets\Lai_JulyData.mat', 'Datasets\Lai_JulyData.mat', 'Datasets\Suguro_JulyData.mat', 'Datasets\Takahashi_JulyData.mat']
    fname = ['Datasets/Lai_JulyData.mat', 'Datasets/Takahashi_JulyData.mat', 'Datasets/Suguro_JulyData.mat']
    dataset_channel_map = {'F4': 0, 'C4': 1, 'P4': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 'F7': 7, 'T3': 8, 'T5': 9, 
                           'Fp1': 10, 'Fp2': 11, 'T4': 12, 'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18}
    
    Xx, yy = capilab_dataset2.get(fname)
    try:
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
        env = EEGChannelOptimze(dataset_info, model_set.Custom1DCNN, 0.35)
        # print(env.action_space.n, env.observation_space.n)
        #check compatible
        # from gym.utils.env_checker import check_env
        # # check_env(env)
        
        print(env.state)
                
        x, r, d, _ = env.step(dataset_info['ch_map']['Cz'])
        print(env.reward_threshold, r, d, env.state)

        x, r, d, _ = env.step(dataset_info['ch_map']['P3'])
        print(env.reward_threshold, r, d, env.state)
        
        x, r, d, _ = env.step(dataset_info['ch_map']['P4'])
        print(env.reward_threshold, r, d, env.state)


    except Exception as e:
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())





    