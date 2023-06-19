# Always delete weights.npy and biases.npy, before running every program
import pickle

with open('dict_for_NN_4_one_output_pts.pickle', 'rb') as f:
    dict_for_NN = pickle.load(f)
    print('No of dataset:',len(dict_for_NN))


from constraint_and_unconstrain_dict import constrain_dict,unconstrain_dict
dict_for_NN_constrained,minimum,maximum=constrain_dict(dict_for_NN)

from neural_net  import data_loader
from neural_net import network
from neural_net import in_out

training_data,test_data,test_dict = data_loader.load_data(dict_for_NN_constrained)
print('No of training data',len(training_data))


net=network.Network([2,30,2]) # input: a tuple =2; , middle layer=30; output: 5 tuples=2*5 points =10
net.SGD(training_data,30,10,1.1)# epochs, mini_batch_size, learning rate

dict_by_NN_On_test_data_constrained=in_out.in_out(test_dict)

dict_by_NN_On_test_data_unconstrained=unconstrain_dict(dict_by_NN_On_test_data_constrained,minimum,maximum)
test_dict_unconstrained=unconstrain_dict(test_dict,minimum,maximum)
print(test_dict_unconstrained)
print()
print()
print()
print(dict_by_NN_On_test_data_unconstrained)

# from plot import create_plots,create_animations
# create_plots(test_dict_unconstrained,3,3,3) # 3 for 3 by 3 picture, l=2,b=2 for x lim an y lim
# create_animations(test_dict_unconstrained,3) # 3 for 3 by 3 animations

