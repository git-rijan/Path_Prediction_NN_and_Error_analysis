# how does error varies agnst no of mid layer for different shuffling
from plot import plot_dicts
import error_vs_middlelayer

my_lst=[]
for i in range(3):
    print(i)
    exec(f"new_dict_{i} = {error_vs_middlelayer.main()}")
    my_lst.append(eval(f"new_dict_{i}"))


print(my_lst)
plot_dicts(my_lst, xlabel='No of Neurons in Mid Layer', ylabel='MSE (Mean Square Error)', xlim=(0,45), ylim=(0,30), title='MSE vs No of Mid Layer')
