import collections
import numpy as np

from test_unit import TEST_ENV

class ExperienceReplay:
    def __init__(self, max_recall):
        self.current = 0
        self.max_recall = max_recall
        self.state_hist =  collections.deque()
        self.action_hist = collections.deque()
        self.reward_hist = collections.deque()     
        self.next_state_hist = collections.deque()
        self.done_hist = collections.deque()
        
    
    def store(self, state, action, reward, next_state, done):
        if self.__current >= self.max_recall:
            self.state_hist.popleft()
            self.action_hist.popleft()
            self.reward_hist.popleft()
            self.next_state_hist.popleft()
            self.done_hist.popleft()  

        self.state_hist.append(state.copy())
        self.action_hist.append(action)
        self.reward_hist.append(reward)
        self.next_state_hist.append(next_state.copy())
        self.done_hist.append(done)
            
        self.current += 1
    
    def sample(self, batch_size):
        indices  = np.random.choice(range(len(self.done_hist)), size = batch_size)
        
        state_sample = np.array([self.state_hist[i] for i in indices])
        action_sample = np.array([self.action_hist[i] for i in indices])
        reward_sample = np.array([self.reward_hist[i] for i in indices])
        next_state_sample = np.array([self.next_state_hist[i] for i in indices])
        done_sample = np.array([self.done_hist[i] for i in indices])
        
        return state_sample, action_sample, reward_sample, next_state_sample, done_sample

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            indices = list(range(indices.stop)[indices])
        else:
            indices = [indices]
            
        state_sample = [self.state_hist[i] for i in indices]
        action_sample = [self.action_hist[i] for i in indices]
        reward_sample = [self.reward_hist[i] for i in indices]
        next_state_sample = [self.next_state_hist[i] for i in indices]
        done_sample = [self.done_hist[i] for i in indices]
        
        return state_sample, action_sample, reward_sample, next_state_sample, done_sample
    


if __name__ =="__main__":
    env = TEST_ENV()
    memory = ExperienceReplay(10)
    done = False

    while not done:
        action = np.random.randint(0,19)
        state = env.state.copy()
        _state, reward, done = env.step(action)
        memory.store(state, action, reward, _state, done)
        print(memory[0])
        state = _state
        


            