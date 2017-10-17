import numpy as np
import random
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import pandas as pd
from sklearn import preprocessing


def make_artificial_datasets(duration = 100, delay = [3,5], hidden = 0):
    x1_buffer, x2_buffer, X_dataset, y_dataset = [0] * delay[0] , [0] * delay[1], [], []
    for i in range(duration):
        a = random.randrange(1,10)
        b = random.randrange(1,10)
        c = x1_buffer.pop(0)
        d = x2_buffer.pop(0)
        z = c+d
        X_dataset.append([a,b])
        y_dataset.append([z])
        x1_buffer.append(a)
        x2_buffer.append(b)
        if hidden:
            print i,"th execution --------- X1:", a,", X2:", b
        else:
            print i,"th execution --------- X1:", a,", X2:", b,", Y:", z
        X_array = np.array(X_dataset)
        y_array = np.array(y_dataset)
    return X_array, y_array


def np_array_to_plot(array, axis = 0,dim =1, hidden_key = 0):
    data_list = []
    if axis:
        time = array.shape[0]
        cat = array.shape[1]
    else:
        time = array.shape[1]
        cat = array.shape[0]
    #print cat, time
    for i in range(cat):
        for j in range(time):
            for k in range(dim):
                if axis:
                    data_list.append([array[j][i],i,j,k])
                else:
                    data_list.append([array[i][j],i,j,k])
    
    result = pd.DataFrame(data_list,columns = ['data','cat','time','dim'])
    if hidden_key:
        sns.plt.ylim(-2,12)
        sns.set(style='darkgrid')
    sns.tsplot(data = result, time='time', condition='cat', value = 'data', unit='dim')
    if hidden_key:
        return result
    
    
def readFile(path, row_start, column_start, output_border_start):
    with open(path) as f:
        content = f.readlines()
    content = [a[0].split(",")for a in [x.strip().split("\n") for x in content]]
    np_content = np.array(content)
    np_content = np_content[row_start-1:].transpose()[column_start-1:].transpose()
    np_content = np.asarray(np_content, dtype=np.float32)
    return np_content[:,:output_border_start-1], np_content[:,output_border_start-1:]

    
def main():
    X_raw, Y_raw = readFile(path="./COP_Data.csv", row_start=2, column_start=2, output_border_start=55)
    x_dim = X_raw.shape[1]
    y_dim = Y_raw.shape[1]
    X = preprocessing.scale(X_raw)
    Y = preprocessing.scale(Y_raw)
    time = 100
    cutoff = X.shape[0] / (time * 2)
    X = X[:cutoff*time,:].reshape(-1,time,x_dim)
    Y = Y[:cutoff*time,:].reshape(-1,time,y_dim)
    print X.shape, Y.shape
    samples = X.shape[0]
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    from keras.layers.recurrent import LSTM
    #from keras.optimizers import SGD
    hidden = 100
    model = Sequential()
    model.add(LSTM(output_dim=hidden, return_sequences=True, stateful=True, init='uniform', 
                   batch_input_shape=(samples, time, x_dim)))
    model.add(LSTM(output_dim=hidden, return_sequences=True, stateful=True, init='uniform'))
    model.add(TimeDistributed(Dense(y_dim)))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    #sgd = SGD(lr=0.1, decay=1e-4, momentum=0.2, nesterov=True)
    #model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(X, Y, batch_size=samples, nb_epoch=1000)
    X_test, Y_test_origin = X_raw[-cutoff*time:,:], Y_raw[-cutoff*time:,:]
    X_test = preprocessing.scale(X_test)
    Y_test = preprocessing.scale(Y_test_origin)
    X_test = X_test.reshape(-1,time,x_dim)
    Y_test = Y_test.reshape(-1,time,y_dim)
    pred = model.predict(X_test, batch_size=samples)
    y_compare = np.array([pred[0].transpose()[0], Y_test[0].transpose()[0]])
    np_array_to_plot(y_compare)
    Y_norm_delta = Y_test[0][5][0] - Y_test[0][6][0]
    Y_origin_delta = Y_test_origin[5][0] - Y_test_origin[6][0]
    std = Y_origin_delta /  Y_norm_delta
    mean = Y_test_origin[5][0]- std * Y_test[0][5][0]
    print std, mean
    pred_norm = std * pred + mean

    for i in range(5,15):
        print(i,"th execution --------- predict:", pred_norm[0].transpose()[0][i],
              ", actual:", Y_test_origin.transpose()[0][i])
        
if __name__=='__main__':
    main()
