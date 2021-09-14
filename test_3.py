import numpy as np
import scipy.io
import scipy
from nn import *
import matplotlib.pyplot as plt
from matplotlib import transforms
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 20
learning_rate = 5e-3 
hidden_size = 64

#img = train_x[0]
#img = np.reshape(img, (32,32))
#plt.imshow(img)
#plt.show()

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here

initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    avg_acc = 0
    for xb,yb in batches:
         # forward

        post_act = forward(xb, params, 'layer1', sigmoid)
        post_act = forward(post_act, params, 'output', softmax)

        # loss

        loss, acc = compute_loss_and_acc(yb, post_act)

        # be sure to add loss and accuracy to epoch totals 

        total_loss += loss / batch_num
        total_acc += acc
        avg_acc = total_acc / batch_num

        # backward

        delta1 = post_act
        delta1[np.arange(post_act.shape[0]),np.argmax(yb, axis=1)] -= 1
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient

        params['boutput'] -= learning_rate * params['grad_boutput']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']

        # training loop can be exactly the same as q2!

    #forward pass for validation data
    valid_post_act = forward(valid_x, params, 'layer1', sigmoid)
    valid_post_act = forward(valid_post_act, params, 'output', softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_post_act)    

    if itr % 10 == 0 or itr == max_iters - 1:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))
        print("Validation loss: ", valid_loss, " Validation accuracy: ", valid_acc)
# run on validation set and report accuracy! should be above 75%

print('Validation accuracy: ',valid_acc)
#if True: # view the data
    #for crop in xb:
        #plt.imshow(crop.reshape(32,32).T)
        #plt.show()
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
test_post_act = forward(test_x, params, 'layer1', sigmoid)
test_post_act = forward(test_post_act, params, 'output', softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, test_post_act)

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

sum = 0

for i in range(test_y.shape[0]):
    post_act_max = np.argmax(test_post_act[i])
    test_max = np.argmax(test_y[i])
    confusion_matrix[post_act_max][test_max] += 1

sum = np.trace(confusion_matrix) 
test_post_act_acc = sum / 1800
print ("Test accuracy: ", test_post_act_acc)



plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
