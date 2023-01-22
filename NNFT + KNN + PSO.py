# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps
from pyswarms.backend import topology

import os

import numpy as np
import pandas as pd

import librosa.display
import matplotlib.pyplot as plt

import nn_fac.ntd as NTD
import barmuscomp

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
'''
Определение жанров
'''
print('Определение жанров\n')
# genres = "blues classical country disco hiphop jazz metal pop reggae rock".split()
'''Количество жанров до 5 штук в классификаторе предсказываются с вероятностью +90%'''
genres = "blues classical country disco hiphop".split()
genres_arr = np.asarray(genres)
'''
Подсчёт композиций
'''
print('Подсчёт композиций\n')
count_of_genres_and_compositions_in_genre = []
for genre in genres_arr:
    # dir_path = f'D:\\GTZAN\\Data\\genres_original\\'+genre
    dir_path = f'D:\\GTZAN\\Data\\genres\\'+genre
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    count_of_genres_and_compositions_in_genre.append([genre, count])
    print('In genre', genre, 'have', count, 'compositions')
print(count_of_genres_and_compositions_in_genre, '\n')
'''
Создание БД композиций
'''
print('Создание БД композиций\n')
for genre in genres_arr:
    if genre == genres_arr[0]:
        audio_data = {}
        audio_sr = {}
        # dir_path = f'D:\\GTZAN\\Data\\genres_original\\'+genre
        dir_path = f'D:\\GTZAN\\Data\\genres\\'+genre
        genre_arr = []
        sr_app = []
        key_arr = []
        for path in os.listdir(dir_path):
            AUDIO_FILE = f'{dir_path}\\' + path
            samples, sample_rate = librosa.load(AUDIO_FILE, mono=False, sr=None)
            genre_arr.append(samples)
            sr_app.append(sample_rate)
            key = path.split(".")
            key_arr.append(key[1])
        index_key = []
        for i in key_arr:
            i = int(i)
            index_key.append(i)
        audio_data.update({f'{genre}': genre_arr})
        audio_sr.update({f'{genre}-sample_rate': sr_app})
        db_genres = pd.DataFrame(audio_data, index=index_key)
        db_sr = pd.DataFrame(audio_sr, index=index_key)
    elif genre != genres_arr[0]:
        audio_data = {}
        audio_sr = {}
        # dir_path = f'D:\\GTZAN\\Data\\genres_original\\'+genre
        dir_path = f'D:\\GTZAN\\Data\\genres\\'+genre
        genre_arr = []
        sr_app = []
        key_arr = []
        for path in os.listdir(dir_path):
            AUDIO_FILE = f'{dir_path}\\' + path
            samples, sample_rate = librosa.load(AUDIO_FILE, mono=False, sr=None)
            genre_arr.append(samples)
            sr_app.append(sample_rate)
            key = path.split(".")
            key_arr.append(key[1])
        index_key = []
        for i in key_arr:
            i = int(i)
            index_key.append(i)
        audio_data.update({f'{genre}': genre_arr})
        audio_sr.update({f'{genre}-sample_rate': sr_app})
        db2 = pd.DataFrame(audio_data, index=index_key)
        db_2_sr = pd.DataFrame(audio_sr, index=index_key)
        db_genres = db_genres.join(db2)
        db_sr = db_sr.join(db_2_sr)

'''
Проверка длины композиции
'''
print('Проверка длины композиции и их сокращение для одинковой длины\n')
lowest_compose = 0
for genre in genres_arr:
    for ind in db_genres.index:
        compose = db_genres[genre][ind]
        len_compose = len(compose)
        if lowest_compose == 0:
            lowest_compose = len_compose
        elif len_compose < lowest_compose:
            lowest_compose = len_compose


for genre in genres_arr:
    for ind in db_genres.index:
        compose = db_genres[genre][ind]
        len_compose = len(compose)
        if len_compose > lowest_compose:
            db_genres[genre][ind] = compose[:-(len_compose - lowest_compose)]
'''
Проверка для галочки
'''
print('Проверка для галочки\n')
lenlen = {}
for genre in genres_arr:
    len_array = []
    len_x = 0
    for ind in db_genres.index:
        compose = db_genres[genre][ind]
        len_compose = len(compose)
        if len_compose != len_x:
            len_array.append(len_compose)
            len_x = len_compose
    lenlen.update({genre: len_array})
print(lenlen)
'''
Поиск неисправных композиций
'''
# db_null = db_genres.isna()
# nan_arr = []
# for genre in genres_arr:
#     nan_index = db_null.index[db_null[genre] == True].tolist()
#     nan_arr.append([genre, nan_index])
# print('nan_arr = ', nan_arr)

# nan_arr_remove = []
# for i in nan_arr:
#     if len(i[1]) == 0:
#         nan_arr_remove.append(i)
# print('to remove', nan_arr_remove)
# for i in nan_arr_remove:
#     nan_arr.remove(i)
# print('to zeros', nan_arr)

# for i in nan_arr:
#     genre = i[0]
#     for j in i[1]:
#         db_genres[i[0]][j] = np.zeros(lowest_compose)

# db_null = db_genres.isna()
# nan_arr = []
# for genre in genres_arr:
#     nan_index = db_null.index[db_null[genre] == True].tolist()
#     nan_arr.append([genre, nan_index])
# print('nan_arr = ', nan_arr)
'''
Сохранение БД в CSV
'''
# # csv_path = f'D:\\GTZAN\\Data\\genres_original_db\\genres_original.csv'
# # path_to_csv = f'D:\\GTZAN\\Data\\genres_original_db'
# csv_path = f'D:\\GTZAN\\Data\\genres_db\\genres.csv'
# path_to_csv = f'D:\\GTZAN\\Data\\genres_db'
# os.makedirs(path_to_csv, exist_ok=True)
# db_genres.to_csv(path_or_buf=csv_path, sep=';')

# # csv_path = f'D:\\GTZAN\\Data\\genres_original_db\\genres_original_sr.csv'
# # path_to_csv = f'D:\\GTZAN\\Data\\genres_original_db'
# csv_path = f'D:\\GTZAN\\Data\\genres_db\\genres_sr.csv'
# path_to_csv = f'D:\\GTZAN\\Data\\genres_db'
# os.makedirs(path_to_csv, exist_ok=True)
# db_sr.to_csv(path_or_buf=csv_path)
'''
Загрузка предзаписанных БД
'''
# db_genres_2 = pd.read_csv(f'D:\\GTZAN\\Data\\genres_db\\genres.csv', index_col=0, sep=';')
# db_sr_2 = pd.read_csv(f'D:\\GTZAN\\Data\\genres_db\\genres_sr.csv', index_col=0)
'''
Получение БД спектрограмм
'''
print('Получение БД спектрограмм\n')

cmap = plt.cm.get_cmap('cool')

mel_data = {}
path_to_spec = f'D:\\GTZAN\\Data\\genres_spec'
# path_to_spec = f'D:\\GTZAN\\Data\\genres_original_spec'
os.makedirs(path_to_spec, exist_ok=True)
for i, j in zip(db_genres.itertuples(), db_sr.itertuples()):
    mel_arr = []
    for x in range(1, len(i)):
        index_sample = i[0]
        # print(index_sample)
        sample = i[x]
        # print(sample)
        index_sr = j[0]
        sr = j[x]
        sgram = librosa.stft(sample)
        sgram_mag, sgram_phase = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sr)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        mel_arr.append(mel_sgram)
        # path_to_genre_spec = f'{path_to_spec}\\{genres_arr[x-1]}'
        # os.makedirs(path_to_genre_spec, exist_ok=True)
        # img = librosa.display.specshow(mel_sgram, sr=sr, x_axis='time', y_axis='mel', cmap=cmap)
        # plt.savefig(f"{path_to_genre_spec}\\{genres_arr[x-1]}{index_sample}.jpg")
        # print(f"Жанр {genres_arr[x-1]} заполнен")
    mel_data.update({f'{i[0]}': mel_arr})
    # print(f"Спектр композиций № {i[0]} сформирован")

ax = db_genres.columns.to_list()
db_mel_sgram = pd.DataFrame(mel_data, index=ax)
db_mel_sgram = db_mel_sgram.transpose()
'''
Выравнивание количества частот
'''
print('Выравнивание количества частот\n')
lowest_freq_dict = {}
for genre in genres:
    len_lowest_freq = 0
    print(f'Жанр = {genre}')
    for ind in db_mel_sgram.index:
        compose = db_mel_sgram[genre][ind]
        shape = list(compose.shape)
        len_freq = shape[1]
        if len_lowest_freq == 0:
            len_lowest_freq = len_freq
        elif len_freq < len_lowest_freq:
            len_lowest_freq = len_freq
    lowest_freq_dict.update({genre: len_lowest_freq})
    # print(f'len_lowest_freq = {len_lowest_freq}')
    for ind in db_mel_sgram.index:
        compose = db_mel_sgram[genre][ind]
        shape = list(compose.shape)
        len_freq = shape[1]
        freq_Arr = []
        for freq in compose:
            if len_freq > len_lowest_freq:
                db_mel_sgram[genre][ind][freq] = freq[:-(len_freq - len_lowest_freq)]
print(lowest_freq_dict)
'''
Функция вычисления NNFT
'''
def NNFT(ranks):
    global genres, db_mel_sgram, db_genres, db_sr, genres_arr
    '''
    NNFT
    '''
    print('NNFT\n')
    '''
    Расчёт ядер и факторов для создания отрывков аудио
    '''
    print('Расчёт ядер и факторов\n')
    data_cores = {}
    data_factors = {}
    data_value_vatiation = {}
    data_time_ntd = {}
    for genre in genres:
        print(f'genre = {genre}')
        pretensor = db_mel_sgram[genre].to_numpy()
        tensor = []
        for j in range(len(pretensor)):
            composition = pretensor[j]
            tensor.append(composition)
        tensor = np.asarray(tensor)
        # nb_modes = len(tensor.shape)
        # print(f'nb_modes = {nb_modes}')
        # shape = list(tensor.shape)
        # print(f'shape = {shape}')
        core, factors, value_variation, time = NTD.ntd(tensor, ranks=ranks, init="tucker", verbose=True,
                                sparsity_coefficients=[None, None, None, None],
                                normalize=[True, True, False, True],
                                return_costs=True)
        print(f'\n')
        data_cores.update({f'{genre}': [core]})
        data_factors.update({f'{genre}': [factors]})
        data_value_vatiation.update({f'{genre}': [value_variation]})
        data_time_ntd.update({f'{genre}': [time]})

    ax = db_genres.columns.to_list()
    db_cores = pd.DataFrame(data_cores)
    db_factors = pd.DataFrame(data_factors)
    db_value_vatiation = pd.DataFrame(data_cores)
    db_time_ntd = pd.DataFrame(data_factors)

    return [db_cores, db_factors, db_value_vatiation, db_time_ntd]
    '''
    Расчёт features
    '''
    # print('Расчёт features\n')
    # print(f'ranks = {ranks}')
    # feat_data = {}
    # barfeatures = barmuscomp.features
    # for i, j in zip(db_genres.itertuples(), db_sr.itertuples()):
    #     feat_arr = []
    #     for x in range(1, len(i)):
    #         # for x in range(1, 2):
    #         # print(f'Features для жанра {genres_arr[x - 1]}, index = {i[0]}')
    #         index_sample = i[0]
    #         signal = i[x]
    #         index_sr = j[0]
    #         sr = j[x]
    #         features_ntd = barfeatures.compute_hcqt_bittner(signal, sr)
    #         feat_arr.append(features_ntd)
    #     feat_data.update({f'{i[0]}': feat_arr})
    #     print(f'Features для index {i[0]} сформированы')
    #
    # print(f'Создание ДБ features\n')
    # ax = db_genres.columns.to_list()
    # db_feat = pd.DataFrame(feat_data, index=ax)
    # db_feat = db_feat.transpose()
    # # print(f'ДБ features = \n'
    # #       f'{db_feat}\n')
    #
    # '''
    # KNN предикт на основании features полученных с помощью NNFT
    # '''
    # print('KNN prediction for NNFT\n')
    # '''
    # Формирование данных для KNN
    # '''
    # print('Формирование данных для KNN\n')
    # '''
    # knn_for_db = []
    # max_len_feat = 0
    # for genre in ['blues', 'classical']:
    # # for genre in genres:
    #     print(f'РАБОТАЕМ по жанру {genre}')
    #     for ind in db_mel_sgram.index:
    #         song_feat_arr = []
    #         # song_feat_arr.append(genre)
    #         print(f'РАБОТАЕМ по индексу {ind}')
    #         song_feat = db_mel_sgram[genre][ind]
    #         for n_feat in song_feat:
    #             shape = n_feat.shape
    #             # print(f'shape[0] = \n'
    #             #       f'{shape[0]}')
    #             for sh in range(shape[0]):
    #                 song_feat_arr.append(n_feat[sh])
    #         print(f'len song_feat_arreat_arr (количество фичей в векторе по индексу) \n'
    #               f'{len(song_feat_arr)}')
    #         len_feat = len(song_feat_arr)
    #         if len_feat > max_len_feat:
    #             max_len_feat = len_feat
    #     print(f'song_feat_arr = \n'
    #           f'{song_feat_arr}')
    #     knn_for_db.append(song_feat_arr)
    # '''
    # knn_for_db = []
    # max_len_feat = 0
    # # for genre in ['blues', 'classical']:
    # for genre in genres:
    #     # print(f'РАБОТАЕМ по жанру {genre}')
    #     for ind in db_mel_sgram.index:
    #         song_feat_arr = []
    #         song_feat_arr.append(genre)
    #         # print(f'РАБОТАЕМ по индексу {ind}')
    #         song_feat = db_mel_sgram[genre][ind]
    #         for n_feat in song_feat:
    #             shape = n_feat.shape
    #             for sh in range(shape[0]):
    #                 song_feat_arr.append(n_feat[sh])
    #         # print(f'len song_feat_arr (количество фичей в векторе по индексу) \n'
    #         #       f'{len(song_feat_arr)}')
    #         len_feat = len(song_feat_arr)
    #         if len_feat > max_len_feat:
    #             max_len_feat = len_feat
    #         # print(f'song_feat_arr = \n'
    #         #       f'{song_feat_arr}')
    #         knn_for_db.append(song_feat_arr)
    #
    # '''
    # Формирование столбцов для KNN
    # '''
    # print('Формирование столбцов для KNN\n')
    #
    # arr_for_feat_cols = []
    # arr_for_feat_cols.append('Classes')
    # for i in range(max_len_feat - 1):
    #     arr_for_feat_cols.append(f'song_features_{i}')
    # db_feat_for_knn = pd.DataFrame(data=knn_for_db, columns=arr_for_feat_cols)
    # # print(db_feat_for_knn.info)
    # # print(f'ДБ features для КНН = \n'
    # #       f'{db_feat_for_knn.head()}\n')
    #
    # '''
    # Разделение на трейн и тест
    # '''
    # print(f'Разделение на трейн и тест\n')
    # X = db_feat_for_knn.iloc[:, 1:]
    # # print(f'X для трейн/теста = \n'
    # #       f'{X}\n')
    # Y = db_feat_for_knn['Classes'].values
    # # print(f'Y для трейн/теста = \n'
    # #       f'{Y}\n')
    # x_train, x_test, y_train, y_test = train_test_split(X, Y,
    #                                                     test_size=0.2,
    #                                                     shuffle=True)
    #
    # # scaler = StandardScaler()
    # # scaler.fit(x_train)
    # # x_train_scaled = scaler.transform(x_train)
    # # x_test_scaled = scaler.transform(x_test)
    #
    # '''
    # Выбор параметров KNN
    # '''
    # print(f'Выбор параметров KNN\n')
    # grid_params = {
    #     'n_neighbors': [3, 5, 7, 9, 11, 15],
    #     'weights': ['uniform', 'distance'],
    #     'metric': ['euclidean', 'manhattan']
    # }
    #
    # '''
    # Настройка модели
    # '''
    # print(f'Настройка модели\n')
    # model = GridSearchCV(KNeighborsClassifier(),
    #                      grid_params,
    #                      cv=2,
    #                      # n_jobs=-1,
    #                      error_score='raise',
    #                      verbose=3)
    # model.fit(x_train, y_train)
    # model_score = model.score(x_test, y_test)
    # # model.fit(x_train_scaled, y_train)
    # # model_score = model.score(x_test_scaled, y_test)
    # print(f'Model Score: \n'
    #       f'{model_score}\n')
    #
    # '''
    # Предикт модели
    # '''
    # print(f'Предикт модели\n')
    # y_predict = model.predict(x_test)
    # # y_predict = model.predict(x_train_scaled)
    # conf_matrix = confusion_matrix(y_pred=y_predict,
    #                                y_true=y_test)
    # print(f'Confusion Matrix: \n'
    #       f'{conf_matrix}\n')
    # return [model_score, conf_matrix]
    # # return model_score

'''
Запуск вручную
'''
ranks = np.array([5, 20, 10])
'''max ranks [10, 128, 1292]'''
n_generation = 1
cores_arr = []
factors_arr = []
value_vatiation_arr = []
time_ntd_arr = []
for numero_gen in range(n_generation):
    print(f'==============================\n'
          f'==============================\n'
          f'\n'
          f'GENERATION # {numero_gen}\n'
          f'- - - - - - - - - - - - - - - -\n')
    # model_score, conf_matrix = NNFT(ranks)
    # print(f'model score = {model_score}\n'
    #       f'matrix:\n'
    #       f'{conf_matrix}')
    db_cores, db_factors, db_value_vatiation, db_time_ntd = NNFT(ranks)
    print(f'time to ntd iterations = \n'
          f'{db_time_ntd}')



'''
PSO
'''
# def accur (accuracy_target, accuracy_test):
#     difference = (accuracy_target - accuracy_test)
#     return difference
#
# swarm_size = 2
# dim = 3  # Dimension of X
# epsilon = 1.0
# options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
# '''
# max ranks = [10, 128, 1292]
# '''
# rank_constraints = (np.array([1,     1,      1]),
#                     np.array([10,    128,    1292]))
#
# def fitness_func(X):
#     n_particles = X.shape[0]  # number of particles
#     accuracy_target = 1
#     accur = [(NNFT(X[i])[0], accuracy_target) for i in range(n_particles)]
#     return np.array(accur)
#
# # Call an instance of PSO
# optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
#                                     dimensions=dim,
#                                     options=options,
#                                     bounds=rank_constraints)
#
# # Perform optimization
# final_best_cost, final_best_pos = optimizer.optimize(fitness_func, iters=5)
#
# print(f'final_best_cost     final_best_pos\n'
#       f'{final_best_cost},  {final_best_pos}')