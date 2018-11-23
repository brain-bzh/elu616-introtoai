# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat

###############################
# Team name to be displayed in the game 
TEAM_NAME = "KerasAI"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here

import numpy as np
import random as rd
import pickle
import time

###############################
# Please put your global variables here

# Global variables
global model,exp_replay,input_tm1, action, score

# Function to create a numpy array representation of the maze

def input_of_parameters(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2*mazeHeight-1,2*mazeWidth-1,1)
    canvas = np.zeros(im_size)
    (x,y) = player
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[y_cheese+center_y-y,x_cheese+center_x-x,0] = 1
#    (x_enemy, y_enemy) = opponent
#    canvas[y_enemy+center_y-y,x_enemy+center_x-x,1] = 1
#    canvas[center_y,center_x,2] = 1
    canvas = np.expand_dims(canvas,axis=0)
    return canvas


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

    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global model,exp_replay,input_tm1, action, score
    input_tm1 = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    action = -1
    score = 0
    model = NLinearModels(input_tm1[0])
    W = np.load(open('save_rl/W.npy',"rb"))
    bias = np.load(open('save_rl/bias.npy',"rb"))
    model.W = W
    model.bias = bias
    
    


###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):    
    global model,input_tm1, action, score
    input_t = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    input_tm1 = input_t    
    output = model.predict(input_tm1)
    action = np.argmax(output[0])
    score = playerScore
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]

def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass    
