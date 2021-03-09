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
#m_depth is the number of rows/elements in a memory block
#n_slots is the number of memory blocks 

def renorm(s):
    return s/K.sum(s,axis = 1,keepdims = True)

def cosine_similarity(K,M):
    return cosine_similarity(K,M)

def controller_read_ouput_dim(m_depth,shift_range):
    beta,k,g,sharpness = 1,1,1,1
    shift = shift_range #not clear
    return (m_depth+k+beta+g+shift+sharpness) #m_depth consist of the weights of the weight that will be multiplied with memory

def controller_write_output_dim(m_depth):
    controller_read_dim = controller_read_output_dim(m_depth,shift_range)
    return controller_read_dim + 2*m_depth #erase and add vector to be included

def controller_input_output_dim(input_dim,output_dim,read_head,write_head,m_depth):

    read_controller_output_dim = controller_read_output_dim
    write_controller_outptut_dim = controller_write_output_dim
    output_controller_dim = (output_dim + write_heads*write_controller_output_dim \
            + read_head*read_controller_output_dim)
    input_controller_dim = input_dim + read_head*m_depth

    return output_dim_controller, input_dim_controller

def NeuralTuringMachine(RNN):

    def __init__(self,units,
                      n_slots = 50, #number of columns
                      m_depth = 20, #number of rows
                      shift_range = 3,
                      read_head = 1,
                      write_head = 1,
                      controller_model = None,
                      batch_size = 777,
                      stateful = False,
                      **kwargs):
        self.output_dim = units
        self.units = units
        self.read_head = read_head
        self.write_head = write_head 
        self.controller = controller_model
        self.n_slots = n_slots
        self.m_depth = m_depth
        self.shift_range = shift_range
        self.batch_size = batch_size

        try:
            
