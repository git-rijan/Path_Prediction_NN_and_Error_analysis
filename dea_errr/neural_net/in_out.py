import numpy as np
import random

def Feedfrwd(a,biases,weights):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(biases, weights):
            a = sgmod(np.dot(w, a)+b)
        return a

#other function
def sgmod(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def in_out(my_dict):
    biases=[]
    with open('neural_net/biases.npy', 'rb') as f:
        biases.append(np.load(f))
        biases.append(np.load(f))

    weights=[]
    with open('neural_net/weights.npy', 'rb') as f:
        weights.append(np.load(f))
        weights.append(np.load(f))

    ret_dict={}
    for loc in my_dict.keys():

        input_arr=np.array(loc).reshape(2,1)

        output_arr=Feedfrwd(input_arr,biases,weights)

        new_array = output_arr.reshape(-1, 2).tolist()

        ret_dict[loc]=new_array

    return ret_dict
