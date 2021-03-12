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
      '''  
      ntm_output = get_activation(ntm_output)
      controller_instructions_read = [(tanh(k), hard_sigmoid(beta)+0.5, sigmoid(g), softmax(shift), 1 + 9*sigmoid(gamma)) for
                (k, beta, g, shift, gamma) in controller_instructions_read]
      controller_instructions_write = [
                (tanh(k), hard_sigmoid(beta)+0.5, sigmoid(g), softmax(shift), 1 + 9*sigmoid(gamma), hard_sigmoid(erase_vector), tanh(add_vector))  for 
                (k, beta, g, shift, gamma, erase_vector, add_vector) in controller_instructions_write]
       
      return (ntm_output, controller_instructions_read, controller_instructions_write)
      '''    

    '''
    @property
    def output_shape(self):
    '''


    def step(self,layer_output,states):
       _,M,weights_read_tm1,weights_write_tm1 =  states[:4]

       weights_read_tm1 = K.reshape(weights_read_tm1,(self.batch_size,self.read_head,self.n_slots))
       weights_write_tm1 = K.reshape(weights_write_tm1,(self.batch_size,self.write_head,self.n_slots))

       memory_read_input = K.concatenate([self.read_from_memory(M,weights_read_tm1[:,i]) for i in range(self.read_head)])
       
       controller_output = self.run_controller(layer_input,memory_read_input)

       ntm_output, controller_instructions_read, controller_instructions_write = self.split_and_apply_activations(controller_output)

       weights_write = []

       for i in range(slef.write_head):
           write_head = controller_instructions_write[i]
           old_weight_vector = weights_write_tm1[:,i]
           weight_vector = self.get_weight_vector(M, old_weight_vector, *tuple(write_head[:5]))
       weights_write.append(weight_vector)

       for i in range(self.write_heads):
           M = self.write_to_memory_erase(M,weights_write[i],controller_instructions_write[i][5])
    
       for i in range(self.write_heads):
           M = self.write_to_memory_add(M,weights_write[i],controller_instructions_write[i][6])

       weights_read = []
       for i in range(self.read_heads):
           read_head = controller_instructions_read[i]
           old_weight_vector = weights_read_tm1[:,i]
           weight_vector = self.get_weight_vector(M,old_weight_vector,*read_heads)
           weights_read.append(weight_vector)

       return ntm_output, [ntm_output,M,K.stack(weigths_read,axis = 1),K.stack(weights_write,axis = 1)]

        
