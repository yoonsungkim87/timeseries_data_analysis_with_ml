import matplotlib.pyplot as plt
import lstm

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

x_raw, y_raw, info = lstm.load_data(path="./COP_Data.csv", row_start=2, column_start=2, output_border_start=55, do_normalize=True)

length = 100
cutoff = int(x_raw.shape[0] / (length * 2))
x_dim = x_raw.shape[1]
y_dim = y_raw.shape[1]
x_train, y_train = x_raw[:cutoff*length,:].reshape(-1,length,x_dim), y_raw[:cutoff*length,:].reshape(-1,length,y_dim)
x_test, y_test = x_raw[-cutoff*length:,:].reshape(-1,length,x_dim), y_raw[-cutoff*length:,:].reshape(-1,length,y_dim)
samples = x_train.shape[0]


m_ = lstm.build_model(samples, length, x_dim, 100, 1, y_dim)
m_.fit(x_train, y_train, batch_size=samples, nb_epoch=100)
y_pred = lstm.predict_sequence(m_, x_test, batch_size=samples)

plot_results(y_pred.reshape(-1, y_dim).transpose()[0], y_test.reshape(-1, y_dim).transpose()[0])