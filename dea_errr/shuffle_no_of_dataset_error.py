# how does error varies agnst difeerent no of dataset for different shuffling
from plot import plot_dicts
import error_vs_no_of_dataset

my_lst=[]
for i in range(5):
    print(i)
    exec(f"new_dict_{i} = {error_vs_no_of_dataset.main()}")
    my_lst.append(eval(f"new_dict_{i}"))


print(my_lst)
plot_dicts(my_lst, xlabel='Number of Dataset', ylabel='MSE (Mean Square Error)', xlim=(10,700), ylim=(0,80), title='MSE vs No of Dataset')
