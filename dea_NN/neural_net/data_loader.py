import numpy as np
import random


def load_data(my_dict):

    list_keys=list(my_dict.keys())
    rndm_list_keys=random.sample(list_keys, int(0.8*len(list_keys))) # 80 % represents amount of data objects you want in training phase
    other_list_keys=list(set(list_keys)-set(rndm_list_keys))

    training_dict={k:my_dict[k] for k in rndm_list_keys}
    test_dict={k:my_dict[k] for k in other_list_keys}

    training_inputs=[np.array(x).reshape(2,1) for x in training_dict.keys()]
    training_results=[np.array(y).reshape(2*len(y),1) for y in training_dict.values()]
    training_data=list(zip(training_inputs,training_results))

    test_inputs=[np.array(x).reshape(2,1) for x in test_dict.keys()]
    test_results=[np.array(y).reshape(2*len(y),1) for y in test_dict.values()]
    test_data=list(zip(test_inputs,test_results))

    return (training_data,test_data,test_dict)

if __name__=='__main__':
    load_data(my_dict,l,b)
