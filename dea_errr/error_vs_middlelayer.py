# Always delete weights.npy and biases.npy, before running every program
import pickle
from neural_net  import data_loader
from neural_net import network
from neural_net import in_out
from errors import mse
from plot import plot_dict

def main():
    with open('dict_for_NN_4_one_output_pts.pickle', 'rb') as f:
        dict_for_NN = pickle.load(f)
        print('No of dataset:',len(dict_for_NN))

    from constraint_and_unconstrain_dict import constrain_dict,unconstrain_dict
    dict_for_NN_constrained,minimum,maximum=constrain_dict(dict_for_NN)

    trn_data,tst_data,tst_dict = data_loader.load_data(dict_for_NN_constrained)
    print('No of training data',len(trn_data))

    # looping
    mid=0
    my_dict={}
    while mid <=60:
        print(mid)
        net=network.Neural_Net([2,mid,2]) # input: a tuple =2; , middle layer=30; output: 5 tuples=2*5 points =10
        net.S_G_D(trn_data,30,10,3)# epochs, mini_batch_size, learning rate

        dict_by_NN_On_test_data_constrained=in_out.in_out(tst_dict)

        dict_by_NN_On_test_data_unconstrained=unconstrain_dict(dict_by_NN_On_test_data_constrained,minimum,maximum)
        test_dict_unconstrained=unconstrain_dict(tst_dict,minimum,maximum)

        # error analysis to be done now by checking test_dict(original) and dict_by_NN_On_test_data(by NN)

        error=mse(dict_by_NN_On_test_data_unconstrained,test_dict_unconstrained,factor_raise=10,factor_high=0)
        my_dict[mid]=error
        mid += 5
    return my_dict

if __name__=='__main__':
    new_dict=main()
    plot_dict(new_dict, xlabel='No of Neurons in Mid Layer', ylabel='MSE (Mean Square Error)', xlim=(0,45), ylim=(0,30), title='MSE vs No of Mid Layer')
