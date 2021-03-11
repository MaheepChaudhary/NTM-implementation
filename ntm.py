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
'''
'''

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

class NeuralTuringMachine(RNN):

    def __init__(self,units,
                      n_slots = 50, #number of columns, containing a feature 
                      m_depth = 20, #number of rows, containing a elements of the feature
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

        '''
        try:
           


        except:
        '''

    self.controller_read_output_dim = controller_read_output_dim(m_depth,shift_range)
    self.controller_write_output_dim = controller_write_ouptut_dim(m_depth)
    
    super().__init__()
'''
    def build(self,input_shape):
        
   



'''

    def get_intial_state():
        
       # init_old_ntm_output = K.ones((self.batch_size, self.ouptut_dim),name = "init_old_ntm_output")*0.42
       # init_M = K.ones((self.batch_size,self.n_slots,self.m_depth),name = "main_menmory")*0.042
        init_wr = np.zeros((self.batch_size,self.read_heads,self.n_slots))
       # init_wr[:,:,0] = 1
        init_wr = K.variable(init_wr,bane = "init_weights_read")
        init_ww = np.zeros((self.batch_Size,self.write_heads,self.n_slots))
       # init_ww[:,:,0] = 1
        init_ww = K.variable(init_ww,name = "init_weights_write")
        
        return [init_old_ntm_output,init_M,init_wr,init_ww]

    def read_from_memory(self,weights):
        return K.sum((weights[:,:,None]*M),axis = 1)

    def write_to_memory_erase(self,weights,erase_wt,M):
        M_tilda =  M*(1 - weights[:,:,None]*erase_wt[:,None,:])
        return M_tilda
    #What is the dimension of the erase vector?

    def write_to_memeory_add(self,weights,add_wt,M):
        return M_tilda + (weights[:,:None]*add_wt[:,None,:])
   
   def get_weight_vector(self,M,w_prev,k,beta,g,shift,gamma):
       num = beta*cosine_similarity(k,M)
       w_c = softmax(num)
       w_g = g*w_c + (1-g)*w_prev
       '''
       C_s =
       w_tilda = 
       '''
       w_out = renorm(w_tilda**gamma)
       
       return w_out

   '''
   def run_controller(self,):








   '''
   def split_and_apply_activations(self,controller_output):

      ntm_output,controller_instructions_read,controller_instructions_write = tf.split(\
              controller_output,np.asarray([self.output_dim,
                                            self.read_head*self.controller_read_output_dim,
                                            self.write_head*self.controller_write_output_dim]),axis = 1)

       controller_instructions_read = tf.split(controller_instructions_read,self.read_head,axis = 1)
       controller_instructions_read = [tf.split(x,np.asarray([self.m_depth,1,1,1,3,1]),axis = 1) for x in controller_instructions_read]

       controller_instructions_write = tf.split(controller_instructions_write,self.write_heads,axis = 1)

       controller_instructions_write = [tf.split(y,np.asarray([self.m_depth,1,1,1,3,1,self.m_depth.self.m_depth]),axis = 1) for x in controller_instructions_write]
