import pandas as pd
import numpy as np
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv1D, Dropout, MaxPooling1D, BatchNormalization, PReLU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate


class Dataset_Loader:
    def __init__(self, training_file='CSV.csv', test_file='CSV.csv', window_length=8, number_of_sensors=2):
        self.training_X, self.training_Y, self.training_windowID, self.training_segmnetID = self.csvReaderToArrays(csv=training_file, window_length=window_length, number_of_sensors=number_of_sensors)
        self.test_X, self.test_Y, self.test_windowID, self.test_segmnetID = self.csvReaderToArrays(csv=test_file, window_length=window_length, number_of_sensors=number_of_sensors)
        self.X, self.Y, self.windowID, self.segmnetID = self.concatenate_train_and_test(self.training_X, self.training_Y, self.training_windowID, self.training_segmnetID, self.test_X, self.test_Y, self.test_windowID, self.test_segmnetID)
        #self.X_transposed = np.transpose(self.training_X, axes=(0, 2, 1))  # Transpose inner matrix ()axes 1 and 2 are switched

        self.X_crossVal_train = []
        self.X_crossVal_train_trasposed = []
        self.Y_crossVal_train = []


        self.X_crossVal_test = []
        self.X_crossVal_test_trasposed = []
        self.Y_crossVal_test = []

        self.num_sensors = number_of_sensors

    def normalizeCrossVal(self):
        # TODO: aqui cuando se lee y se escribe al arreglo numpy X y Y, restar mean y divifir std
        # aqui uso el training y aplico al test la modificacion
        self.X_crossVal_train = np.array(self.X_crossVal_train)
        self.Y_crossVal_train = np.array(self.Y_crossVal_train)
        self.X_crossVal_test = np.array(self.X_crossVal_test)
        self.Y_crossVal_test = np.array(self.Y_crossVal_test)

        # this is the same as np.sum(np.sum(np_array_2d, axis = 1), axis=0), which gets the sum of each in vertical
        # (resulting in a matrix) and then sum again in the vertical (easier way is done as with std)
        x_sum = np.sum(self.X_crossVal_train, axis=(1, 0)) #this is the same as np.sum(np.sum(np_array_2d, axis = 1), axis=0), which gets the sum of each in vertical and then of all
        num_elements = self.X_crossVal_train.shape[0] * self.X_crossVal_train.shape[1]
        x_mean = x_sum/num_elements


        if self.num_sensors == 0 or self.num_sensors == 1:
            x_std = np.std(self.X_crossVal_train.reshape(-1, 3), axis=0) #Reshaping so all instances are colapsed one after the other
        elif self.num_sensors == 2:
            x_std = np.std(self.X_crossVal_train.reshape(-1, 6), axis=0)
        else:
            return -1

        self.X_crossVal_train = (self.X_crossVal_train-x_mean)/x_std
        self.X_crossVal_test = (self.X_crossVal_test - x_mean) / x_std



    def trasposeX(self, num_instances_training, num_instances_test):
        # Needs transposing so all points (in vertical line columns) are near each other sequentially when reshaped
        # and not next to the horizontal points for knn
        self.X_crossVal_train_trasposed = np.transpose(self.X_crossVal_train,
                                                       axes=(
                                                       0, 2, 1))  # Transpose inner matrix ()axes 1 and 2 are switched
        self.X_crossVal_train_trasposed = self.X_crossVal_train_trasposed.reshape(num_instances_training, -1)

        self.X_crossVal_test_trasposed = np.transpose(self.X_crossVal_test,
                                                       axes=(
                                                           0, 2,
                                                           1))  # Transpose inner matrix ()axes 1 and 2 are switched
        self.X_crossVal_test_trasposed = self.X_crossVal_test_trasposed.reshape(num_instances_test, -1)


    @staticmethod
    def csvReaderToArrays(csv='CSV.csv', window_length=8, number_of_sensors=2):
        X = []
        Y = []
        dataframe_segmnetID = None
        windowID = []
        if number_of_sensors == 2 or number_of_sensors == 1 or number_of_sensors == 0:
            dataframe = pd.read_csv(csv)
            dataframe_X = dataframe.drop(columns=['segmnetID', 'windowID', 'class_label'])
            dataframe_Y = dataframe[['class_label']]
            dataframe_segmnetID = dataframe[['segmnetID']]
            dataframe_windowID = dataframe[['windowID']]
            start = 0
            stop = len(dataframe)
            step = window_length
            #print("Dataset BEFORE before: " + str(len(dataframe)))
            for i in range(start, stop, step):
                instance = []
                for j in range(i, i+step, 1):
                    instance.append(dataframe_X.iloc[j].to_numpy())
                    if j == i:
                        Y.append(dataframe_Y.iloc[i].to_numpy())
                        windowID.append(dataframe_windowID.iloc[i])
                aux = np.asarray(instance)
                X.append(aux)
            #print("Dataset AFTER before: " + str(len(X)))
            X = np.asarray(X)
            Y = to_categorical(Y) #one-hot-encoding
            Y = np.asarray(Y)
            #print(X.shape, Y.shape, len(windowID), len(dataframe_segmnetID))
        return X, Y, windowID, dataframe_segmnetID

    @staticmethod
    def concatenate_train_and_test(X_train, Y_train, windowID_train, segmentID_train, X_test, Y_test, windowID_test, segmentID_test):
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        windowID = np.concatenate((windowID_train, windowID_test), axis=0)
        segmentID = np.concatenate((segmentID_train, segmentID_test), axis=0)
        return X, Y, windowID, segmentID


class Model_Library:
    @staticmethod
    def ModelSelector(num_classes, timeSteps=8, features=6, weights_path=None, model_string='default_1D_CNN', seed=10):
        if model_string == 'default_1D_CNN':
            model = Sequential()
            model.add(Conv1D(filters=16, kernel_size=3, activation='linear', input_shape=(timeSteps, features), kernel_initializer=he_normal(seed=seed), padding='SAME'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Conv1D(filters=32, kernel_size=3, activation='linear', kernel_initializer=he_normal(seed=seed), padding='SAME'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Conv1D(filters=64, kernel_size=3, activation='linear', kernel_initializer=he_normal(seed=seed), padding='SAME'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Conv1D(filters=128, kernel_size=3, activation='linear', kernel_initializer=he_normal(seed=seed), padding='valid'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Conv1D(filters=128, kernel_size=3, activation='linear', kernel_initializer=he_normal(seed=seed), padding='valid'))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Conv1D(filters=64, kernel_size=3, activation='linear', kernel_initializer=he_normal(seed=seed), padding='valid'))
            model.add(BatchNormalization())
            model.add(PReLU())
            #model.add(Dropout(0.7))
            model.add(MaxPooling1D(pool_size=2, name='feature_layer'))
            model.add(Flatten())
            model.add(Dense(100, activation='linear', kernel_initializer=he_normal(seed=seed)))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Dense(num_classes, activation='softmax', kernel_initializer=he_normal(seed=seed)))
            if weights_path:
                model.load_weights(weights_path)
            return model


        elif model_string == 'multi_head_CNN_6':
            # head 1
            # Here it uses only one input because it is only one column input_shape=(timeSteps)
            inputs1_1 = Input(shape=(timeSteps, 1))
            conv1_1 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_1)
            batch1_1 = BatchNormalization()(conv1_1)
            prelu1_1 = PReLU()(batch1_1)
            drop1_1 = Dropout(0.45)(prelu1_1)
            pool1_1 = MaxPooling1D(pool_size=2)(drop1_1)
            flat1_1 = Flatten()(pool1_1)


            # head 2
            inputs1_2 = Input(shape=(timeSteps, 1))
            conv1_2 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_2)
            batch1_2 = BatchNormalization()(conv1_2)
            prelu1_2 = PReLU()(batch1_2)
            drop1_2 = Dropout(0.45)(prelu1_2)
            pool1_2 = MaxPooling1D(pool_size=2)(drop1_2)
            flat1_2 = Flatten()(pool1_2)


            # head 3
            inputs1_3 = Input(shape=(timeSteps, 1))
            conv1_3 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_3)
            batch1_3 = BatchNormalization()(conv1_3)
            prelu1_3 = PReLU()(batch1_3)
            drop1_3 = Dropout(0.45)(prelu1_3)
            pool1_3 = MaxPooling1D(pool_size=2)(drop1_3)
            flat1_3 = Flatten()(pool1_3)


            # head 4
            inputs1_4 = Input(shape=(timeSteps, 1))
            conv1_4 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_4)
            batch1_4 = BatchNormalization()(conv1_4)
            prelu1_4 = PReLU()(batch1_4)
            drop1_4 = Dropout(0.45)(prelu1_4)
            pool1_4 = MaxPooling1D(pool_size=2)(drop1_4)
            flat1_4 = Flatten()(pool1_4)


            # head 5
            inputs1_5 = Input(shape=(timeSteps, 1))
            conv1_5 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_5)
            batch1_5 = BatchNormalization()(conv1_5)
            prelu1_5 = PReLU()(batch1_5)
            drop1_5 = Dropout(0.45)(prelu1_5)
            pool1_5 = MaxPooling1D(pool_size=2)(drop1_5)
            flat1_5 = Flatten()(pool1_5)


            # head 6
            inputs1_6 = Input(shape=(timeSteps, 1))
            conv1_6 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_6)
            batch1_6 = BatchNormalization()(conv1_6)
            prelu1_6 = PReLU()(batch1_6)
            drop1_6 = Dropout(0.45)(prelu1_6)
            pool1_6 = MaxPooling1D(pool_size=2)(drop1_6)
            flat1_6 = Flatten()(pool1_6)


            # merge
            # merged = Concatenate()([flat1_1, flat1_2, flat1_3, flat1_4, flat1_5, flat1_6])
            merged = Concatenate()([pool1_1, pool1_2, pool1_3, pool1_4, pool1_5, pool1_6])
            # interpretation
            preluFinal = PReLU()(merged)
            convFinal = Conv1D(filters=64, kernel_size=3, activation='linear',
                               kernel_initializer=he_normal(seed=seed), padding='SAME')(preluFinal)
            batchFinal = BatchNormalization()(convFinal)
            preluFinal = PReLU()(batchFinal)
            preluFinal2 = PReLU()(preluFinal)
            convFinal2 = Conv1D(filters=32, kernel_size=3, activation='linear',
                                kernel_initializer=he_normal(seed=seed), padding='SAME')(preluFinal2)

            batchFinal2 = BatchNormalization()(convFinal2)
            preluFinal2 = PReLU()(batchFinal2)
            flat1Final2 = Flatten()(preluFinal2)
            # dense1 = Dense(100, activation='relu')(merged)
            # outputs = Dense(num_classes, activation='softmax')(dense1)
            outputs = Dense(num_classes, activation='softmax')(flat1Final2)
            model = Model(inputs=[inputs1_1, inputs1_2, inputs1_3, inputs1_4, inputs1_5, inputs1_6], outputs=outputs)

            if weights_path:
                model.load_weights(weights_path)

            return model


        elif model_string == 'multi_head_CNN_2':
            # head 1
            # Here it uses only one input because it is only one column input_shape=(timeSteps)
            inputs1_1 = Input(shape=(timeSteps, 3))
            conv1_1 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_1)
            batch1_1 = BatchNormalization()(conv1_1)
            prelu1_1 = PReLU()(batch1_1)
            drop1_1 = Dropout(0.45)(prelu1_1)
            pool1_1 = MaxPooling1D(pool_size=2)(drop1_1)
            flat1_1 = Flatten()(pool1_1)

            # head 2
            inputs1_2 = Input(shape=(timeSteps, 3))
            conv1_2 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_2)
            batch1_2 = BatchNormalization()(conv1_2)
            prelu1_2 = PReLU()(batch1_2)
            drop1_2 = Dropout(0.45)(prelu1_2)
            pool1_2 = MaxPooling1D(pool_size=2)(drop1_2)
            flat1_2 = Flatten()(pool1_2)

            # merge
            # merged = Concatenate()([flat1_1, flat1_2])
            merged = Concatenate()([pool1_1, pool1_2])
            # interpretation
            preluFinal = PReLU()(merged)
            convFinal = Conv1D(filters=64, kernel_size=3, activation='linear',
                               kernel_initializer=he_normal(seed=seed), padding='SAME')(preluFinal)
            batchFinal = BatchNormalization()(convFinal)
            preluFinal = PReLU()(batchFinal)
            preluFinal2 = PReLU()(preluFinal)
            convFinal2 = Conv1D(filters=32, kernel_size=3, activation='linear',
                                kernel_initializer=he_normal(seed=seed), padding='SAME')(preluFinal2)

            batchFinal2 = BatchNormalization()(convFinal2)
            preluFinal2 = PReLU()(batchFinal2)
            flat1Final2 = Flatten()(preluFinal2)
            # dense1 = Dense(100, activation='relu')(merged)
            # outputs = Dense(num_classes, activation='softmax')(dense1)
            outputs = Dense(num_classes, activation='softmax')(flat1Final2)
            model = Model(inputs=[inputs1_1, inputs1_2], outputs=outputs)

            if weights_path:
                model.load_weights(weights_path)

            return model

        elif model_string == 'multi_head_CNN_3':
            # head 1
            # Here it uses only one input because it is only one column input_shape=(timeSteps)
            inputs1_1 = Input(shape=(timeSteps, 1))
            conv1_1 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_1)
            batch1_1 = BatchNormalization()(conv1_1)
            prelu1_1 = PReLU()(batch1_1)
            drop1_1 = Dropout(0.45)(prelu1_1)
            pool1_1 = MaxPooling1D(pool_size=2)(drop1_1)
            flat1_1 = Flatten()(pool1_1)


            # head 2
            inputs1_2 = Input(shape=(timeSteps, 1))
            conv1_2 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_2)
            batch1_2 = BatchNormalization()(conv1_2)
            prelu1_2 = PReLU()(batch1_2)
            drop1_2 = Dropout(0.45)(prelu1_2)
            pool1_2 = MaxPooling1D(pool_size=2)(drop1_2)
            flat1_2 = Flatten()(pool1_2)


            # head 3
            inputs1_3 = Input(shape=(timeSteps, 1))
            conv1_3 = Conv1D(filters=16, kernel_size=3, activation='linear',
                             kernel_initializer=he_normal(seed=seed), padding='SAME')(inputs1_3)
            batch1_3 = BatchNormalization()(conv1_3)
            prelu1_3 = PReLU()(batch1_3)
            drop1_3 = Dropout(0.45)(prelu1_3)
            pool1_3 = MaxPooling1D(pool_size=2)(drop1_3)
            flat1_3 = Flatten()(pool1_3)

            # merge
            # merged = Concatenate()([flat1_1, flat1_2, flat1_3, flat1_4, flat1_5, flat1_6])
            merged = Concatenate()([pool1_1, pool1_2, pool1_3])
            # interpretation
            preluFinal = PReLU()(merged)
            convFinal = Conv1D(filters=64, kernel_size=3, activation='linear',
                               kernel_initializer=he_normal(seed=seed), padding='SAME')(preluFinal)
            batchFinal = BatchNormalization()(convFinal)
            preluFinal = PReLU()(batchFinal)
            preluFinal2 = PReLU()(preluFinal)
            convFinal2 = Conv1D(filters=32, kernel_size=3, activation='linear',
                                kernel_initializer=he_normal(seed=seed), padding='SAME')(preluFinal2)

            batchFinal2 = BatchNormalization()(convFinal2)
            preluFinal2 = PReLU()(batchFinal2)
            flat1Final2 = Flatten()(preluFinal2)
            # dense1 = Dense(100, activation='relu')(merged)
            # outputs = Dense(num_classes, activation='softmax')(dense1)
            outputs = Dense(num_classes, activation='softmax')(flat1Final2)
            model = Model(inputs=[inputs1_1, inputs1_2, inputs1_3], outputs=outputs)

            if weights_path:
                model.load_weights(weights_path)

            return model
