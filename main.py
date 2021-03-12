import tensorflow as tf  
import numpy as np 
import keras 
from keras.layers.recurrent import LSTM
from keras.layers.models import Sequential
from keras.initializers import RandomNormal
from keras.optimizers import Adam,SGD
from keras.layers.core import Dense
import model
from ntm import controller_input_output_shape as controller_shape

#building this model for having the controller as the model_ntm and also for controller architecture as Dense
output_dim = 8
input_dim = ouput_dim + 2
batch_size = 100
read_head = 1
write_head = 1

lr = 5e-4
clipnorm = 10 
sgd = Adam(lr = lr, clipnorm = clipnorm)
sameInit = RandomNormal(seed = 0)

controller = Sequential()
controller_name = "dense"

controller_shape(input_dim,ouput_dim,20,128,3,read_head,write_head) 
controller.add(Dense(units = controller_output_dim,
                     kernel_initializer = sameInit,
                     bias_intializer = sameInit,
                     activation = 'linear',
                     input_dim = controller_input_dim))

controller.compile(loss = "binary_crossentropy",optimizer = sgd, metrics = ["binary_accuracy"], sample_weight_mode = "temporal")

model = model.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,\
        controller_model=controller, read_head=read_head, write_head=write_head,activation="sigmoid")


