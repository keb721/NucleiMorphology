import numpy as np

f = '{}_distributions_all_{}_{}.txt'
new_f = '{}_{}_distributions_all_{}_{}.dat'

cutoffs = {'solid': ['7', '11'], 'dotpro': ['04', '0625']}
types   = ['constructed', 'bigconstr']
connect = ['oriented', 'proximal']


for tp in types:
    for cutoff in cutoffs.keys():
        for value in cutoffs[cutoff]:
            data = open(f.format(tp, cutoff, value), 'r').read().split('\n')
            for i in range(2):
                output = open(new_f.format(connect[i], tp, cutoff, value), 'w')
                output.write(data[0]+'\n')
                for j in range(int(0.5*len(data))-1):
                    print(1+i+2*j, len(data))
                    # File name doesn't actually matter
                    output.write(data[1+i+2*j] + '\n')

            output.close()

                    
        




