import numpy as np
import pickle

## Part 1 - Linear Model Training using SGD
# This part can be skipped unless you want to understand the details of how the linear model is being trained using Stochastic Gradient Descent. 
# A starting point can be found here : https://medium.com/deeplearningschool/2-1-linear-regression-f782ada81a53
# However there are many online ressources on the topic. 

class NLinearModels(object):
    def __init__(self,x_example,number_of_regressors=4,learning_rate = 0.1):
        shape_input = x_example.reshape(-1).shape[0]
        limit = np.sqrt(6 / (shape_input + number_of_regressors)) 
        self.W = np.random.uniform(-limit,limit, size=(shape_input,number_of_regressors)) #HE INITIALIZATION
        #self.W = np.ones((shape_input,number_of_regressors))/10 #HE INITIALIZATION
        self.bias = np.zeros(number_of_regressors)
        self.learning_rate = learning_rate
    
    def forward(self,x):
        return x.dot(self.W) + self.bias

    def predict(self,x):
        x = np.array(x)
        x = x.reshape(x.shape[0],-1)
        return self.forward(x)

    def cost(self,y_hat,y):
        return ((y_hat-y)**2).mean(axis=0)
        
    def backward(self,x,y_hat,y):
        m = y_hat.shape[0]
        dl = 2*(y_hat-y)/m
        self.bias_gradient = np.sum(dl,axis=0) 
        self.W_gradient = x.T.dot(dl)/m 

    def train_on_batch(self,_input,target):
        _input = np.array(_input)
        y = np.array(target)
        x = _input.reshape(_input.shape[0],-1)
        y_hat = self.forward(x)
        cost = self.cost(y_hat,y).sum()
        self.backward(x,y_hat,y)
        self.update_weights()
        return cost

    def update_weights(self):

        self.W -= self.learning_rate * self.W_gradient 
        self.bias -= self.learning_rate * self.bias_gradient

    def load(self):
        W = np.load(open('save_rl/W.npy',"rb"))
        bias = np.load(open('save_rl/bias.npy',"rb"))
        self.W = W
        self.bias = bias

    def save(self):
        np.save(open("save_rl/W.npy","wb"),self.W)
        np.save(open("save_rl/bias.npy","wb"),self.bias)

## Part 2 - Experience Replay
## This part has to be read and understood in order to code the main.py file. 

class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory. 
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """
    def __init__(self, max_memory=100, discount=.9):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience
        
        In the memory the information whether the game ended at the experience is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, experience, game_over):
        #Save an experience to memory
        self.memory.append([experience, game_over])
        #We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        
        #How many experiences do we have?
        len_memory = len(self.memory)
        
        #Calculate the number of actions that can possibly be taken in the game
        num_actions = 4
        
        #Dimensions of the game field
        env_dim = list(self.memory[0][0][0].shape)
        env_dim[0] = min(len_memory, batch_size)
        
        
        #We want to return an input and target vector with inputs from an observed state...
        inputs = np.zeros(env_dim)
        #...and the target r + gamma * max Q(s’,a’)
        #Note that our target is a matrix, with possible fields not only for the action taken but also
        #for the other possible actions. The actions not take the same value as the prediction to not affect them
        Q = np.zeros((inputs.shape[0], num_actions))
        
        #We draw experiences to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
#            idx = -1
            state, action_t, reward_t, state_tp1 = self.memory[idx][0]
            #We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            #add the state s to the input
            inputs[i:i+1] = state
            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            Q[i] = model.predict([state])[0]

            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                Q[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                next_round = model.predict([state_tp1])[0]
                Q[i, action_t] = reward_t + self.discount*np.max(next_round)
        return inputs, Q

    def load(self):
        self.memory = pickle.load(open("save_rl/memory.pkl","rb"))
    def save(self):
        pickle.dump(self.memory,open("save_rl/memory.pkl","wb"))
