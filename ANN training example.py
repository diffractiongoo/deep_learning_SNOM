import numpy as np
import h5py
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv1D
from keras.callbacks import CSVLogger

## Number of epochs used for training ##
n_epoch = 5

## Define the model ##
model = Sequential()
model.add(Conv1D(64, activation='relu', input_shape=(351, 2,), kernel_size=(25)))
model.add(Conv1D(128, activation='relu', kernel_size=(15)))
model.add(Conv1D(256, activation='relu', kernel_size=(5)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='relu'))
model.summary()
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

## Store the training history ##
csv_logger = CSVLogger("noisy_validation_model_history_log.csv", append=True)

## Train the neural network ##
for i in range(1, 31):
    print('step: ' + str(i))
    ## Load data ##
    # with h5py.File('F:/PycharmProjects/test/paper_radiative/spheroid_paper_radiative_new_method_25a_' + str(i) + '.mat', 'r') as f:
    with h5py.File(
            'F:/PycharmProjects/test/noisy_test/spheroid_paper_radiative_noisy_new_method_25a_' + str(i) + '.mat',
            'r') as f:
    # with h5py.File('/data/yuz289/40_60_Zmin_increment=0.01/spheroid_paper_radiative_change_L_change_Zmin_' + str(i) + '.mat', 'r') as f:
        #print("Keys: %s" % f.keys())
        S_matrix = np.array(f.get('S_matrix'))
        Parameters = np.array(f.get('Parameters'))
        parameters = np.transpose(Parameters)
        parameters = parameters[:, 0:4]
        # print(parameters)

        Data = np.zeros(shape=(120000, 351, 2))
        for l in range(1, 120001):
            temp = [j[l-1] for j in S_matrix]
            Data[l-1, :, 0] = [k[0] for k in temp]
            Data[l-1, :, 1] = [k[1] for k in temp]
    del S_matrix
    ## create the training/validation sets ##
    X_train, X_test, y_train, y_test = train_test_split(Data, parameters, random_state=32, test_size=0.20)
    ## Train the network ##
    model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=2, epochs=n_epoch, callbacks=[csv_logger])
    del Data
    del X_train
    del X_test
    del y_train
    del y_test
    del parameters

##Save weights##
# model.save_weights('ML_spheroid_paper_radiative_new_method_25a_relu_Adam_0.0001.h5')
model.save_weights('ML_spheroid_paper_radiative_noisy_new_method_25a_relu_Adam_0.0001_validation.h5')



