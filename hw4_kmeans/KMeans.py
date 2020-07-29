import sys
import csv
import numpy as np


def k_means(data, k):
    kcenter = []

    # Pick k cluster centers
    for i in range(k):
        kcenter.append(np.array(data[i]))
    
    while True:
        distance = []
        stop = True
        
        # Assign each point to closest center
        for center in kcenter:
            distance.append(((data - center) ** 2).sum(axis=1))        
        distance = np.array(distance)
        distance = distance.argmin(axis=0)
        
        # Find the new center
        for i in range(k):
            temp = data[distance == i].mean(axis=0)
            if (temp - kcenter[i]).sum() != 0:
                kcenter[i] = temp
                stop = False
        
        # Repeat until none of the cluster assignments change
        if stop:
            break

    return distance, kcenter


if __name__ == '__main__':
    if sys.argv[-1] == 'blackbox41':
        from Blackbox41 import blackbox41 as bb
    elif sys.argv[-1] == 'blackbox42':
        from Blackbox42 import blackbox42 as bb
    else:
        print('invalid blackbox')
        sys.exit()

    # Get data from blackbox
    data_set = bb.ask()

    # Number of clusters
    label_num = 4
    labels, k_center = k_means(data_set, label_num)

    with open('results_'+sys.argv[-1]+'.csv', 'w') as fp:
        writer = csv.writer(fp)
        for i in labels:
            writer.writerow(str(i))
