import numpy as np
def BoxFeature(input_list):
    """
    get the feature of box figure.
    
    > @param[in] input_list:    the series
    return: 
    < @param[out] out_list:     the feature value
    < @param[out_note]:         [ave,min,Q1,Q2,Q3,max,error_number]
    """
    percentile = np.percentile(input_list, (25, 50, 75), interpolation='linear')
    Q1 = percentile[0]  # upper quartile
    Q2 = percentile[1]
    Q3 = percentile[2]  # lower quartile
    IQR = Q3 - Q1       # Interquartile range
    ulim = Q3 + 1.5*IQR # upper limit
    llim = Q1 - 1.5*IQR # lower limit
    # llim = 0 if llim < 0 else llim
    # out_list = [llim,Q1,Q2,Q3,ulim]
    # ------- count the number of anomalies ----------
    right_list = []     # normal data
    Error_Point_num = 0
    value_total = 0
    average_num = 0
    for item in input_list:
        if item < llim or item > ulim:
            Error_Point_num += 1
        else:
            right_list.append(item)
            value_total += item
            average_num += 1
    average_value =  value_total/average_num
    out_list = [average_value,min(right_list), Q1, Q2, Q3, max(right_list), Error_Point_num]
    return out_list
