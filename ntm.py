import tensorflow as tf 
import numpy as np 
import tensorflow.keras as tk

from tensorflow.keras import backend as K
from tensorflow.keras.layers import RNN,Dense
from tensorflow.keras.activations import softmax,tanh
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

'''
def _roll_out(leng,n_shifts): #not clear

    eye = np.eye(leng)
    shifts = range(n_shifts//2,-n_shifts//2,-1)
    C = np.asarray([np.roll(eye,s,axis = 
'''

def renorm(s):
    return s/K.sum(s,axis = 1,keepdims = True)

def cosine_similarity(K,M):
    return cosine_similarity(K,M)

def controller_read_ouput_dim(shift_range,input_dim):

    beta,k,g,sharpness = 1,1,1,1
    shift = shift_range #not clear
    return (input_dim+k+beta+g+shift+sharpness)

def controller_write_output_dim(input_):


