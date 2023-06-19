# how does error varies agnst different learning rate for different shuffling
from plot import plot_dicts
import error_vs_learningrate

my_lst=[]
for i in range(3):
    print(i)
    exec(f"new_dict_{i} = {error_vs_learningrate.main()}")
    my_lst.append(eval(f"new_dict_{i}"))


print(my_lst)
plot_dicts(my_lst, xlabel='Learning rate', ylabel='MSE (Mean Square Error)', xlim=(0.1,3.8), ylim=(0,30), title='MSE vs Learning rate')
