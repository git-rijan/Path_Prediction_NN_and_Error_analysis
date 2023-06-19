import numpy as np

def mse(dict1,dict2,factor_raise,factor_high):
    total_sum=0
    for key in dict1.keys():
        a=np.array(dict1[key])
        b=np.array(dict2[key])
        sum=np.sum((a-b)**2)/len(a)
        total_sum=total_sum+sum
    divide_by=len(list(dict1.keys()))
    print()
    return ((total_sum*factor_raise)/divide_by)+factor_high


def percent_error(dict1,dict2): # dict_1 true value
    x_summus=0
    y_summus=0
    for key in dict1:
        x_array_1=np.array([point[0] for point in dict1[key]])
        x_array_2=np.array([point[0] for point in dict2[key]])
        x_percent_array=(np.abs(x_array_2-x_array_1)/x_array_1)*100
        mean_x_percent=np.mean(x_percent_array)

        x_summus=x_summus+mean_x_percent

        y_array_1=np.array([point[1] for point in dict1[key]])
        y_array_2=np.array([point[1] for point in dict2[key]])
        y_percent_array=(np.abs(y_array_2-y_array_1)/y_array_1)*100
        mean_y_percent=np.mean(y_percent_array)

        y_summus=y_summus+mean_y_percent

    return [x_summus/len(dict1),y_summus/len(dict1)]





