import tensorflow as tf 
import numpy as np 
import tensorflow.keras as tk

from tensorflow.keras import backend as K
from tensorflow.keras.layers import RNN,Dense
from tensorflow.keras.activations import softmax,tanh
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

'''
def _roll_out(leng,n_shifts):

    eye = np.eye(leng)
    shifts = range(n_shifts//2,-n_shifts//2,-1)
    C = np.asarray([np.roll(eye,s,axis = 
'''

def renorm(s):
    return s/K.sum(s,axis = 1,keepdims = True)

def cosine_similarity(K,M):

