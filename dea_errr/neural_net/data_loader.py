import numpy as np
import random

def load_data(my_dict):

    list_keys=list(my_dict.keys())
    rndm_list_keys=random.sample(list_keys, int(0.8*len(list_keys))) # 80 % represents amount of data objects you want in training phase
    other_list_keys=list(set(list_keys)-set(rndm_list_keys))

    trn_dict={k:my_dict[k] for k in rndm_list_keys}
    tst_dict={k:my_dict[k] for k in other_list_keys}

    trn_inputs=[np.array(x).reshape(2,1) for x in trn_dict.keys()]
    trn_results=[np.array(y).reshape(2*len(y),1) for y in trn_dict.values()]
    trn_data=list(zip(trn_inputs,trn_results))

    tst_inputs=[np.array(x).reshape(2,1) for x in tst_dict.keys()]
    tst_results=[np.array(y).reshape(2*len(y),1) for y in tst_dict.values()]
    tst_data=list(zip(tst_inputs,tst_results))

    return (trn_data,tst_data,tst_dict)

if __name__=='__main__':
    load_data(my_dict,l,b)
