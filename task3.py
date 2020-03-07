import os
import math
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from contextlib import redirect_stdout
from keras.callbacks import EarlyStopping


def load_images(image_folder):
    def load_image():
        for i in range(image_folder[1]):
            image = Image.open(os.path.abspath(
                '') + '/' + image_folder[0] + '/' + str(i) + '.png').convert('RGB')
            yield image
    return list(load_image())


folder_basketball_drill = ('BasketballDrill_832x480_50', 500)
folder_blowing_bubbles = ('BlowingBubbles_416x240_50', 500)
folder_race_horses = ('RaceHorses_416x240_30', 300)

imgs_basketball = load_images(folder_basketball_drill)
imgs_bubble = load_images(folder_blowing_bubbles)
imgs_horse = load_images(folder_race_horses)


def conv(orig_imgs, ratio, train_percentage, learning_rate, epochs, batch_size, patience, pic_name, method):
    input_img_array= np.asarray([np.asarray(img) for img in orig_imgs]) / 255
    train_amount = input_img_array.shape[0] * train_percentage // 100
    train_imgs, test_imgs = input_img_array[:train_amount], input_img_array[train_amount:]
    input_img = Input(shape=input_img_array[0].shape)

    decoded = None

    if method == 1:
        if ratio == 2:
            e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
            e = MaxPooling2D((2, 1))(e)
            e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 1), activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu')(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", activation='relu')(d)
        elif ratio == 4:
            e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
            e = MaxPooling2D((2, 2))(e)
            e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 2), activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu')(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", activation='relu')(d)
        elif ratio == 8:
            e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
            e = MaxPooling2D((2, 1))(e)
            e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
            e = MaxPooling2D((2, 2))(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 2), activation='relu')(d)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 1), activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu')(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", activation='relu')(d)
        elif ratio == 16:
            e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
            e = MaxPooling2D((2, 2))(e)
            e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
            e = MaxPooling2D((2, 2))(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 2), activation='relu')(d)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 2), activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu')(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", activation='relu')(d)
        elif ratio == 32:
            e = Conv2D(64, (7, 7), activation='relu', padding='same')(input_img)
            e = MaxPooling2D((2, 1))(e)
            e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
            e = MaxPooling2D((2, 2))(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = MaxPooling2D((2, 2))(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 2), activation='relu')(e)
            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 2), activation='relu')(d)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(1, kernel_size=(1, 1), padding="SAME", strides=(2, 1), activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu')(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", activation='relu')(d)
    elif method == 2:
        if ratio == 2:
            e = Conv2D(64, (7, 7), activation='relu', padding='same', strides=(2, 1))(input_img)
            e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu')(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", strides=(2, 1), activation='relu')(d)
        elif ratio == 4:
            e = Conv2D(64, (7, 7), activation='relu', padding='same', strides=(2, 2))(input_img)
            e = Conv2D(32, (5, 5), activation='relu', padding='same')(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu')(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", strides=(2, 2), activation='relu')(d)
        elif ratio == 8:
            e = Conv2D(64, (7, 7), activation='relu', padding='same', strides=(2, 1))(input_img)
            e = Conv2D(32, (5, 5), activation='relu', padding='same', strides=(2, 2))(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu', strides=(2, 2))(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", strides=(2, 1), activation='relu')(d)
        elif ratio == 16:
            e = Conv2D(64, (7, 7), activation='relu', padding='same', strides=(2, 2))(input_img)
            e = Conv2D(32, (5, 5), activation='relu', padding='same', strides=(2, 2))(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same')(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu')(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu', strides=(2, 2))(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", strides=(2, 2), activation='relu')(d)
        elif ratio == 32:
            e = Conv2D(64, (7, 7), activation='relu', padding='same', strides=(2, 1))(input_img)
            e = Conv2D(32, (5, 5), activation='relu', padding='same', strides=(2, 2))(e)
            e = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(e)
            e = Conv2D(3, (3, 3), activation='relu', padding='same')(e)

            d = Conv2DTranspose(16, kernel_size=(3, 3), padding="SAME", activation='relu')(e)
            d = Conv2DTranspose(32, kernel_size=(5, 5), padding="SAME", activation='relu', strides=(2, 2))(d)
            d = Conv2DTranspose(64, kernel_size=(7, 7), padding="SAME", activation='relu', strides=(2, 2))(d)
            decoded = Conv2DTranspose(3, kernel_size=(3, 3), padding="SAME", strides=(2, 1), activation='relu')(d)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    with open(pic_name + '_modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            autoencoder.summary()
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

    autoencoder.fit(train_imgs, train_imgs,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test_imgs, test_imgs),
                    callbacks=[early_stopping])

    decoded_img_array = autoencoder.predict(test_imgs)
    psnr_values = []

    psnr_file = open(pic_name + '_psnr.txt', 'w')
    for i in range(len(decoded_img_array)):
        img_array = decoded_img_array[i] * 255
        img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
        img.save(pic_name + '_' + str(train_amount + i) + '.png')

        orig_img32_array = np.asarray(orig_imgs[train_amount + i]).astype(np.float32)
        mse = np.square(orig_img32_array - img_array).mean(axis=None)
        psnr = 10 * math.log10((255 ** 2) / mse)
        psnr_values.append(psnr)
        psnr_file.write(str(psnr) + '\n')
        print(psnr)

    psnr_avg = 'Avg:' + str(sum(psnr_values) / len(psnr_values)) + '\n'
    psnr_file.write(psnr_avg)
    psnr_file.write('learning rate:' + str(learning_rate) + '\n')
    psnr_file.write('epochs:' + str(epochs) + '\n')
    psnr_file.close()
    print(psnr_avg)

# conv(imgs_basketball, 2, 80, 1e-3, 500, 32, 25, 'test_ball', 1)