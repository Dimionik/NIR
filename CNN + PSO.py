import os

import numpy as np
import pandas as pd

import librosa.display
import matplotlib.pyplot as plt

import tensorflow as tf

import pyswarms as ps

import datetime
'''
Раскоментировать все ниже, если работа алгоритма с нуля
'''
path_to_GTZAN = f'D:\\GTZAN\\Data'
# '''
# Определение жанров
# '''
# print('Определение жанров\n')
# # genres = "blues classical country disco hiphop jazz metal pop reggae rock".split()
# genres = "blues classical country disco".split()
# genres_arr = np.asarray(genres)
# '''
# Подсчёт композиций
# '''
# print('Подсчёт композиций\n')
# count_of_genres_and_compositions_in_genre = []
# for genre in genres_arr:
#     # dir_path = f'{path_to_GTZAN}\\genres_original\\'+genre
#     dir_path = f'{path_to_GTZAN}\\genres\\'+genre
#     count = 0
#     for path in os.listdir(dir_path):
#         if os.path.isfile(os.path.join(dir_path, path)):
#             count += 1
#     count_of_genres_and_compositions_in_genre.append([genre, count])
#     print('In genre', genre, 'have', count, 'compositions')
# print(count_of_genres_and_compositions_in_genre, '\n')
# '''
# Создание БД композиций
# '''
# print('Создание БД композиций\n')
# for genre in genres_arr:
#     if genre == genres_arr[0]:
#         audio_data = {}
#         audio_sr = {}
#         # dir_path = f'{path_to_GTZAN}\\genres_original\\'+genre
#         dir_path = f'{path_to_GTZAN}\\genres\\'+genre
#         genre_arr = []
#         sr_app = []
#         key_arr = []
#         for path in os.listdir(dir_path):
#             AUDIO_FILE = f'{dir_path}\\' + path
#             samples, sample_rate = librosa.load(AUDIO_FILE, mono=False, sr=None)
#             genre_arr.append(samples)
#             sr_app.append(sample_rate)
#             key = path.split(".")
#             key_arr.append(key[1])
#         index_key = []
#         for i in key_arr:
#             i = int(i)
#             index_key.append(i)
#         audio_data.update({f'{genre}': genre_arr})
#         audio_sr.update({f'{genre}-sample_rate': sr_app})
#         db_genres = pd.DataFrame(audio_data, index=index_key)
#         db_sr = pd.DataFrame(audio_sr, index=index_key)
#     elif genre != genres_arr[0]:
#         audio_data = {}
#         audio_sr = {}
#         # dir_path = f'{path_to_GTZAN}\\genres_original\\'+genre
#         dir_path = f'{path_to_GTZAN}\\genres\\'+genre
#         genre_arr = []
#         sr_app = []
#         key_arr = []
#         for path in os.listdir(dir_path):
#             AUDIO_FILE = f'{dir_path}\\' + path
#             samples, sample_rate = librosa.load(AUDIO_FILE, mono=False, sr=None)
#             genre_arr.append(samples)
#             sr_app.append(sample_rate)
#             key = path.split(".")
#             key_arr.append(key[1])
#         index_key = []
#         for i in key_arr:
#             i = int(i)
#             index_key.append(i)
#         audio_data.update({f'{genre}': genre_arr})
#         audio_sr.update({f'{genre}-sample_rate': sr_app})
#         db2 = pd.DataFrame(audio_data, index=index_key)
#         db_2_sr = pd.DataFrame(audio_sr, index=index_key)
#         db_genres = db_genres.join(db2)
#         db_sr = db_sr.join(db_2_sr)
#
# '''
# Проверка длины композиции
# '''
# print('Проверка длины композиции и их сокращение для одинковой длины\n')
# lowest_compose = 0
# for genre in genres_arr:
#     for ind in db_genres.index:
#         compose = db_genres[genre][ind]
#         len_compose = len(compose)
#         if lowest_compose == 0:
#             lowest_compose = len_compose
#         elif len_compose < lowest_compose:
#             lowest_compose = len_compose
#
#
# for genre in genres_arr:
#     for ind in db_genres.index:
#         compose = db_genres[genre][ind]
#         len_compose = len(compose)
#         if len_compose > lowest_compose:
#             db_genres[genre][ind] = compose[:-(len_compose - lowest_compose)]
# '''
# Проверка для галочки
# '''
# print('Проверка для галочки\n')
# lenlen = {}
# for genre in genres_arr:
#     len_array = []
#     len_x = 0
#     for ind in db_genres.index:
#         compose = db_genres[genre][ind]
#         len_compose = len(compose)
#         if len_compose != len_x:
#             len_array.append(len_compose)
#             len_x = len_compose
#     lenlen.update({genre: len_array})
# print(lenlen)
#
# '''
# Получение БД спектрограмм
# '''
# print('Получение БД спектрограмм\n')
#
# cmap = plt.cm.get_cmap('cool')
#
# mel_data = {}
# path_to_spec = f'{path_to_GTZAN}\\genres_spec'
# # path_to_spec = f'{path_to_GTZAN}\\genres_original_spec'
# os.makedirs(path_to_spec, exist_ok=True)
# for i, j in zip(db_genres.itertuples(), db_sr.itertuples()):
#     mel_arr = []
#     for x in range(1, len(i)):
#         index_sample = i[0]
#         # print(index_sample)
#         sample = i[x]
#         # print(sample)
#         index_sr = j[0]
#         sr = j[x]
#         sgram = librosa.stft(sample)
#         sgram_mag, sgram_phase = librosa.magphase(sgram)
#         mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
#         mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
#         mel_arr.append(mel_sgram)
#         # path_to_genre_spec = f'{path_to_spec}\\{genres_arr[x-1]}'
#         # os.makedirs(path_to_genre_spec, exist_ok=True)
#         # img = librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel', cmap=cmap)
#         # plt.savefig(f"{path_to_genre_spec}\\{genres_arr[x-1]}{index_sample}.jpg")
#         # print(f"Жанр {genres_arr[x-1]} заполнен")
#     mel_data.update({f'{i[0]}': mel_arr})
#     # print(f"Спектр композиций № {i[0]} сформирован")
#
# ax = db_genres.columns.to_list()
# db_mel_sgram = pd.DataFrame(mel_data, index=ax)
# db_mel_sgram = db_mel_sgram.transpose()

'''
CNN
'''

def CNN(chosen_optimizer, epo, numero_gen=1, preparation=None, plot=None):
    # global
    print(f'Начало CNN\n')
    
    epo = np.int32(epo)
    chosen_optimizer = np.int32(chosen_optimizer)

    path_to_spec = f'{path_to_GTZAN}\\genres_spec'

    spectr_height = 256
    spectr_width = 256
    batch_size = 16
    n_channels = 3
    n_classes = 10
    model_chape = (spectr_height, spectr_width, n_channels)

    # Make a dataset containing the training spectrograms
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        batch_size=batch_size,
        validation_split=0.2,
        directory=os.path.join(path_to_spec),
        shuffle=True,
        color_mode='rgb',
        image_size=(spectr_height, spectr_width),
        subset="training",
        seed=0)

    # Make a dataset containing the validation spectrogram
    valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        batch_size=batch_size,
        validation_split=0.2,
        directory=os.path.join(path_to_spec),
        shuffle=True,
        color_mode='rgb',
        image_size=(spectr_height, spectr_width),
        subset="validation",
        seed=0)

    if preparation == True:
        def prepare(ds, augment=False):
            rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)])
            flip_and_rotate = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
            ])

            ds = ds.map(lambda x, y: (rescale(x, training=True), y))
            if augment: ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))
            return ds

        train_dataset = prepare(train_dataset, augment=False)
        valid_dataset = prepare(valid_dataset, augment=False)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=model_chape))
    model.add(tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    if chosen_optimizer == 1:
        optimizer = tf.keras.optimizers.Adadelta()
    elif chosen_optimizer == 2:
        optimizer = tf.keras.optimizers.Adagrad()
    elif chosen_optimizer == 3:
        optimizer = tf.keras.optimizers.Adam()
    elif chosen_optimizer == 4:
        optimizer = tf.keras.optimizers.Adamax()
    elif chosen_optimizer == 5:
        optimizer = tf.keras.optimizers.Ftrl()
    elif chosen_optimizer == 6:
        optimizer = tf.keras.optimizers.Nadam()
    elif chosen_optimizer == 7:
        optimizer = tf.keras.optimizers.RMSprop()
    elif chosen_optimizer == 8:
        optimizer = tf.keras.optimizers.SGD()

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'],
    )

    history = model.fit(train_dataset,
                        epochs=epo,
                        validation_data=valid_dataset,
                        # verbose=0
                        )

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    path_to_CNN = f"{path_to_GTZAN}\\CNN"
    os.makedirs(path_to_CNN, exist_ok=True)

    if plot == True:
        epochs = range(1, len(loss_values) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title(f'Training and validation loss #{numero_gen}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig(f"{path_to_CNN}\\loss_{numero_gen}.jpg")

        epochs = range(1, len(acc_values) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
        plt.title(f'Training and validation accuracy #{numero_gen}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.savefig(f"{path_to_CNN}\\accuracy_{numero_gen}.jpg")

    final_loss, final_acc = model.evaluate(valid_dataset, verbose=0)
    print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))

    return [loss_values, val_loss_values, final_loss, acc_values, val_acc_values, final_acc]

'''
Запуск основного алгоритма CNN для усреднения результатов без PSO
'''
# print(f'Запуск алгоритма предсказания CNN на случайных/выбранных настройках\n')
# epo = 500
# n_generation = 10
# loss_values_arr = []
# val_loss_values_arr = []
# final_loss_arr = []
# acc_values_arr = []
# val_acc_values_arr = []
# final_acc_arr = []

# for numero_gen in range(n_generation):
#     print(f'==============================\n'
#           f'==============================\n'
#           f'\n'
#           f'GENERATION # {numero_gen}\n'
#           f'- - - - - - - - - - - - - - - -\n')
#     optimist = [1, 2, 3, 4, 5, 6, 8, 9]
#     chosen_optimizer = np.random.choice(optimist)
#     print(f'optimizer = {chosen_optimizer}\n')
#     loss_values, val_loss_values, final_loss, acc_values, val_acc_values, final_acc = CNN(chosen_optimizer, epo, numero_gen, preparation=None, plot=True)
#     loss_values_arr.append(loss_values)
#     val_loss_values_arr.append(val_loss_values)
#     final_loss_arr.append(final_loss)
#     acc_values_arr.append(acc_values)
#     val_acc_values_arr.append(val_acc_values)
#     final_acc_arr.append(final_acc)
#
# columns = np.arange(start=0,
#                     stop=epo,
#                     step=1)
#
# db_loss_values = pd.DataFrame(data=loss_values_arr, columns=columns)
# db_val_loss_values = pd.DataFrame(data=val_loss_values_arr, columns=columns)
# db_acc_values = pd.DataFrame(data=acc_values_arr, columns=columns)
# db_val_acc_values = pd.DataFrame(data=val_acc_values_arr, columns=columns)
#
# loss_values_mean_arr = []
# val_loss_values_mean_arr = []
# acc_values_mean_arr = []
# val_acc_values_mean_arr = []
#
# for i in range(epo):
#     loss_values_mean = np.mean(db_loss_values[i])
#     val_loss_values_mean = np.mean(db_val_loss_values[i])
#     acc_values_mean = np.mean(db_acc_values[i])
#     val_acc_values_mean = np.mean(db_val_acc_values[i])
#     loss_values_mean_arr.append(loss_values_mean)
#     val_loss_values_mean_arr.append(val_loss_values_mean)
#     acc_values_mean_arr.append(acc_values_mean)
#     val_acc_values_mean_arr.append(val_acc_values_mean)
#
# final_loss_mean = np.mean(final_loss_arr)
# final_acc_mean = np.mean(final_acc_arr)
#
# path_to_CNN = f"{path_to_GTZAN}\\CNN"
# os.makedirs(path_to_CNN, exist_ok=True)
#
# plt.figure(figsize=(8, 6))
# plt.plot(columns+1, loss_values_mean_arr, 'bo', label='Training loss')
# plt.plot(columns+1, val_loss_values_mean_arr, 'b', label='Validation loss')
# plt.title(f'Mean training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# plt.savefig(f"{path_to_CNN}\\mean_loss.jpg")
#
# plt.figure(figsize=(8, 6))
# plt.plot(columns+1, acc_values_mean_arr, 'bo', label='Training accuracy')
# plt.plot(columns+1, val_acc_values_mean_arr, 'b', label='Validation accuracy')
# plt.title('Mean training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# plt.savefig(f"{path_to_CNN}\\mean_accuracy.jpg")
#
# print("Mean final loss: {0:.4f}, mean final accuracy: {1:.4f}".format(final_loss_mean, final_acc_mean))

'''
PSO
'''
print(f'Начало PSO\n')
def accur (accuracy_target, accuracy_test):
    difference = (accuracy_target - accuracy_test)
    return difference

swarm_size = 2
dim = 2  # Dimension of X
epsilon = 1.0
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}

CNN_constraints = (np.array([1, 1], dtype=np.int32),
                   np.array([8, 5], dtype=np.int32))

def fitness_func(X):
    n_particles = X.shape[0]  # number of particles
    accuracy_target = 1
    accur = [(CNN(X[i][0], X[i][1])[3], accuracy_target) for i in range(n_particles)]
    return np.array(accur)

# Call an instance of PSO
optimizerPSO = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                       dimensions=dim,
                                       options=options,
                                       bounds=CNN_constraints)

# Perform optimization
final_best_cost, final_best_pos = optimizerPSO.optimize(fitness_func, iters=5)

print('final_best_cost     final_best_pos\n'
      '{},                 {}'.format(final_best_cost, final_best_pos))
