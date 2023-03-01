"""
Created on Tue Feb 28 15:36:24 2023

@author: daniel
"""

import caffe
import numpy as np

#define function that loads data
# ---- ENTER CODE ----


#define main()



# Define the network architecture
# this should match the paper's figure 6


# the following is an example of a similar network and needs to be modified
net = caffe.NetSpec()
net.data = caffe.layers.Input(name='data', shape=[2, 3, 224, 224])
net.conv1 = caffe.layers.Convolution(name='conv1', kernel_size=11, num_output=96, stride=2)
net.relu1 = caffe.layers.ReLU(name='relu1', in_place=True)
net.pool1 = caffe.layers.Pooling(name='pool1', pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
net.norm1 = caffe.layers.LRN(name='norm1', local_size=5, alpha=1e-4, beta=0.75)
net.conv2 = caffe.layers.Convolution(name='conv2', kernel_size=5, num_output=256, stride=1, pad=2, weight_filler=dict(type='xavier'))
net.relu2 = caffe.layers.ReLU(name='relu2', in_place=True)
net.pool2 = caffe.layers.Pooling(name='pool2', pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
net.norm2 = caffe.layers.LRN(name='norm2', local_size=5, alpha=1e-4, beta=0.75)
net.conv3 = caffe.layers.Convolution(name='conv3', kernel_size=3, num_output=384, stride=1, pad=1, weight_filler=dict(type='xavier'))
net.relu3 = caffe.layers.ReLU(name='relu3', in_place=True)
net.conv4 = caffe.layers.Convolution(name='conv4', kernel_size=3, num_output=384, stride=1, pad=1, weight_filler=dict(type='xavier'))
net.relu4 = caffe.layers.ReLU(name='relu4', in_place=True)
net.conv5 = caffe.layers.Convolution(name='conv5', kernel_size=3, num_output=256, stride=1, pad=1, weight_filler=dict(type='xavier'))
net.relu5 = caffe.layers.ReLU(name='relu5', in_place=True)
net.pool5 = caffe.layers.Pooling(name='pool5', pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
net.fc6 = caffe.layers.InnerProduct(name='fc6', num_output=4096, weight_filler=dict(type='xavier'))
net.relu6 = caffe.layers.ReLU(name='relu6', in_place=True)

# save training data in file 

# figure out what the output is and how to best return it

