from __future__ import division


#---------------- Loading the MNIST dataset ----------------------------------

import gzip
import cPickle
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set


#---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap = cm.Greys_r)
plt.show() # Let's see a sample
print train_y[57]


#---------------- Creating the training set of only two digits, 1 and 8 ------

samples = [] # Our training set, initially empty
labels = [] # Our labels set, initially empty
j = 0
for label in train_y:
    if label == 1:
        samples.append(train_x[j])
        labels.append([1]) # Label '1' for the set of ones
    if label == 8:
        samples.append(train_x[j])
        labels.append([-1]) # Label '-1' for the set of eights
    j += 1

   
#---------------- Training the neural network --------------------------------
    
import neurolab as nl
import numpy as np

intervals = [[0.0,1.0] for i in range(784)] #intervals of input values
net = nl.net.newff(intervals,[4, 1],[nl.trans.TanSig(), nl.trans.TanSig()])
error = net.train( samples , labels , epochs=10, show=1, goal=0.01)

test_x, test_y = test_set

n = 0
j = 0
misclassified = 0
for i in test_y:
    if i == 1:
        n += 1
        if net.sim([test_x[j]]) < 0.0:
            misclassified += 1
    if i == 8:
        n += 1
        if net.sim([test_x[j]]) > 0.0:
            misclassified += 1
    j += 1
        
print "----------- Results -------------"
print "# test samples (1 and 8): " + str(n)
print "# misclassified samples: " + str(misclassified)
print "Error rate: " + str((misclassified/n)*100.0) + "%"