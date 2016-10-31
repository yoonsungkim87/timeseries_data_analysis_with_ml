def make_artificial_datasets(t_dur = 100, delay = [10,20,30,40], prt= False):
    import random
    import numpy as np
    
    buffers = [None for _ in range(len(delay))]
    for i in range(len(delay)):
        buffers[i] = [0] * delay[i]
    result = []
    for i in range(t_dur):
        temp_in = [None for _ in range(len(delay))]
        temp_out = 0
        for j in range(len(delay)):
            temp_in[j] = random.randrange(1,10)
            buffers[j].append(temp_in[j])
            temp_out += buffers[j].pop(0)
        result.append(temp_in + [temp_out])
        if prt:
            print('%2d'%i+' >>  '),
            for j in range(len(delay)):
                print('X%d'%j+': %2d'%temp_in[j]),
            print('Y: %2d'%temp_out)
    return np.array(result, dtype=np.int32)

def plot_np_array_2d(np_array, t_direction_down=True, hidden_option=False):
    get_ipython().magic(u'matplotlib inline')
    import seaborn as sns
    import pandas as pd
    
    data_list = []
    if t_direction_down:
        time = np_array.shape[0]
        cat = np_array.shape[1]
    else:
        time = np_array.shape[1]
        cat = np_array.shape[0]
    for i in range(cat):
        for j in range(time):
            if t_direction_down:
                data_list.append([np_array[j][i],i,j,0])
            else:
                data_list.append([np_array[i][j],i,j,0])
    result = pd.DataFrame(data_list,columns = ['data','cat','time','dim'])
    if hidden_option:
        sns.plt.ylim(-2,12)
        sns.set(style='darkgrid')
    sns.tsplot(data = result, time='time', condition='cat', value = 'data', unit='dim')
    if hidden_option:
        return result
    
def main():
    #Chainer Module Import
    import train_ptb
    plot_np_array_2d(make_artificial_datasets())

if __name__ == '__main__':
    main()