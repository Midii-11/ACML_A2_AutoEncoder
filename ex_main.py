import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KERAS_BACKEND"] = "tensorflow"
kerasBKED = os.environ["KERAS_BACKEND"]


from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


import numpy as np


class One():
    def __init__(self):
        self.c10test = None
        self.model = None
        self.batch_size = 32
        self.epochs = 100

        # Initialize Train Test Val sets & normalize
        (x_train, _), (x_test, _) = cifar10.load_data()
        x_tot = np.concatenate((x_train, x_test))

        x_train, x_sub, _, _ = train_test_split(x_tot, x_tot, train_size=.8, test_size=.2, random_state=42,
                                                shuffle=True)
        x_test, x_val, _, _ = train_test_split(x_sub, x_sub, train_size=.5, test_size=.5, random_state=42,
                                               shuffle=True)
        self.x_train = x_train / 255
        self.x_test = x_test / 255
        self.x_val = x_val / 255

        print(x_train.shape)
        print(x_test.shape)
        print(x_val.shape)

    def base_model(self):
        # Initialize callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00005,
            patience=11,
            verbose=1,
            restore_best_weights=True,
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        )

        callbacks = [
            early_stopping,
            lr_scheduler,
        ]

        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(filters=8,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__1')(input_img)
        x = BatchNormalization(name='BatchNorm__1')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__1")(x)

        x = Conv2D(filters=12,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__2')(x)
        x = BatchNormalization(name='BatchNorm__2')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__2")(x)

        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__3')(x)
        encoded = BatchNormalization(name='BatchNorm__3')(x)

        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__1')(encoded)
        x = BatchNormalization(name='BatchNorm__4')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=12,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__4')(x)
        x = BatchNormalization(name='BatchNorm__5')(x)
        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__2')(x)
        x = BatchNormalization(name='BatchNorm__6')(x)
        x = Activation('sigmoid')(x)
        x = Conv2D(filters=3,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__5')(x)
        decoded = BatchNormalization(name='BatchNorm__7')(x)

        model = Model(input_img, decoded)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.x_train, self.x_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(self.x_val, self.x_val),
                            shuffle=True)
        # TODO: Save model
        model_yaml = model.to_yaml()
        with open("Basic\\model_base.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        model.save("Basic\\model_base.h5")

        return model, history

    def Alt_1_model(self):
        # Initialize callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00005,
            patience=11,
            verbose=1,
            restore_best_weights=True,
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        )

        callbacks = [
            early_stopping,
            lr_scheduler,
        ]

        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__1')(input_img)
        x = BatchNormalization(name='BatchNorm__1')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__1")(x)

        x = Conv2D(filters=32,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__2')(x)
        x = BatchNormalization(name='BatchNorm__2')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__2")(x)

        x = Conv2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__3')(x)
        encoded = BatchNormalization(name='BatchNorm__3')(x)

        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__1')(encoded)
        x = BatchNormalization(name='BatchNorm__4')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__4')(x)
        x = BatchNormalization(name='BatchNorm__5')(x)
        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__2')(x)
        x = BatchNormalization(name='BatchNorm__6')(x)
        x = Activation('sigmoid')(x)
        x = Conv2D(filters=3,
                   kernel_size=(3, 3),
                   padding='same',
                   activation='relu',
                   name='Conv2D__5')(x)
        decoded = BatchNormalization(name='BatchNorm__7')(x)

        model = Model(input_img, decoded)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.x_train, self.x_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(self.x_val, self.x_val),
                            shuffle=True)
        model_yaml = model.to_yaml()
        with open("Alt_1\\Alt_1_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        model.save("Alt_1\\Alt_1_model.h5")

        return model, history

    def Alt_2_model(self):

        # Initialize callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00005,
            patience=11,
            verbose=1,
            restore_best_weights=True,
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        )

        callbacks = [
            early_stopping,
            lr_scheduler,
        ]

        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(filters=8,
                   kernel_size=(5, 5),
                   padding='same',
                   activation='relu',
                   name='Conv2D__1')(input_img)
        x = BatchNormalization(name='BatchNorm__1')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__1")(x)

        x = Conv2D(filters=12,
                   kernel_size=(5, 5),
                   padding='same',
                   activation='relu',
                   name='Conv2D__2')(x)
        x = BatchNormalization(name='BatchNorm__2')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__2")(x)

        x = Conv2D(filters=16,
                   kernel_size=(5, 5),
                   padding='same',
                   activation='relu',
                   name='Conv2D__3')(x)
        encoded = BatchNormalization(name='BatchNorm__3')(x)

        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__1')(encoded)
        x = BatchNormalization(name='BatchNorm__4')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=12,
                   kernel_size=(5, 5),
                   padding='same',
                   activation='relu',
                   name='Conv2D__4')(x)
        x = BatchNormalization(name='BatchNorm__5')(x)
        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__2')(x)
        x = BatchNormalization(name='BatchNorm__6')(x)
        x = Activation('sigmoid')(x)
        x = Conv2D(filters=3,
                   kernel_size=(5, 5),
                   padding='same',
                   activation='relu',
                   name='Conv2D__5')(x)
        decoded = BatchNormalization(name='BatchNorm__7')(x)

        model = Model(input_img, decoded)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.x_train, self.x_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(self.x_val, self.x_val),
                            shuffle=True)

        model_yaml = model.to_yaml()
        with open("Alt_2\\Alt_2_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        model.save("Alt_2\\Alt_2_model.h5")

        return model, history

    def Alt_3_model(self):
        # Initialize callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00005,
            patience=11,
            verbose=1,
            restore_best_weights=True,
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        )

        callbacks = [
            early_stopping,
            lr_scheduler,
        ]

        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(filters=8,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   activation='relu',
                   name='Conv2D__1')(input_img)
        x = BatchNormalization(name='BatchNorm__1')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__1")(x)

        x = Conv2D(filters=12,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   activation='relu',
                   name='Conv2D__2')(x)
        x = BatchNormalization(name='BatchNorm__2')(x)

        x = MaxPooling2D(pool_size=(2, 2),
                         padding='same',
                         name="MaxPool__2")(x)

        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   activation='relu',
                   name='Conv2D__3')(x)
        encoded = BatchNormalization(name='BatchNorm__3')(x)

        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__1')(encoded)
        x = BatchNormalization(name='BatchNorm__4')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=12,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   activation='relu',
                   name='Conv2D__4')(x)
        x = BatchNormalization(name='BatchNorm__5')(x)
        x = UpSampling2D(size=(2, 2),
                         name='Upsampl__2')(x)
        x = BatchNormalization(name='BatchNorm__6')(x)
        x = Activation('sigmoid')(x)
        x = Conv2D(filters=3,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding='same',
                   activation='relu',
                   name='Conv2D__5')(x)
        decoded = BatchNormalization(name='BatchNorm__7')(x)

        model = Model(input_img, decoded)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(self.x_train, self.x_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(self.x_val, self.x_val),
                            shuffle=True)
        model_yaml = model.to_yaml()
        with open("Alt_3\\Alt_3_model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        model.save("Alt_3\\Alt_3_model.h5")

        return model, history



    def evaluate(self, model):
        # score = self.model.evaluate(self.x_test, self.x_test, verbose=1)
        # print(score)
        self.c10test = model.predict(self.x_test)


    def compare(self, history, path, alt):
        n = 10
        orig = self.x_test
        dec = self.c10test
        fig = plt.figure(figsize=(20, 5))
        # Plot images and predicted-images
        for i in range(n):
            # display original
            ax = fig.add_subplot(3, n, i + 1)
            plt.imshow(((orig[i].reshape(32, 32, 3)) * 255).astype(np.uint8))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = fig.add_subplot(3, n, i + 1 + n)
            if alt is False:
                plt.imshow(((dec[i].reshape(32, 32, 3)) * 255).astype(np.uint8))
            else:
                plt.imshow((dec[i] * 255).astype(np.uint8))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # Plot Accuracy of the model
        fig.add_subplot(3, 2, 5)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        # Plot Loss of the model
        fig.add_subplot(3, 2, 6)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')

        plt.savefig(path)
        # plt.show()


def base():
    path = 'Basic\\History_Basic.png'
    one = One()
    model, history = one.base_model()
    one.evaluate(model)
    one.compare(history, path, False)
def alternative_1():
    path = 'Alt_1\\Alt_1_model.png'
    two = One()
    model, history = two.Alt_1_model()
    two.evaluate(model)
    two.compare(history, path, False)
def alternative_2():
    path = 'Alt_2\\Alt_2_model.png'
    tree = One()
    model, history = tree.Alt_2_model()
    tree.evaluate(model)
    tree.compare(history, path, False)

def alternative_3():
    path = 'Alt_3\\Alt_3_model.png'
    four = One()
    model, history = four.Alt_3_model()
    four.evaluate(model)
    four.compare(history, path, True)



if __name__ == '__main__':
    base()
    alternative_1()
    alternative_2()
    alternative_3()
