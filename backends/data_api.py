import keras
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Dropout, Conv1D, BatchNormalization, MaxPooling1D, Flatten
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def get_features(df, features):
    return pd.DataFrame(data=df, columns=features)


def get_data_with_info():
    return pd.concat([data, get_info_from_df(data)], axis=1)


def get_info_from_df(df):
    track_info_list_2d = df[['artists', 'track_name', 'album_name']].values.tolist()
    track_info_list = [str(i[0]) + ' - ' + str(i[1]) + ' | from "' + str(i[2]) + '"' for i in track_info_list_2d]
    track_info_df = pd.DataFrame(track_info_list, columns=['track_info'])
    return track_info_df


def get_id_by_info(input_info):
    data_with_info = get_data_with_info()
    return data_with_info.loc[data_with_info['track_info'] == input_info, ['track_id']]


def get_info_by_id(input_id):
    data_with_info = get_data_with_info()
    return data_with_info.loc[data_with_info['track_id'] == input_id, ['track_info']]


def get_genre_list():
    return data[['track_genre']].drop_duplicates()


def df_to_ndarray(df):
    return np.array(df)


def filter_df_by_genre(df, input_genre_list):
    return df.loc[df['track_genre'].isin(input_genre_list)]


def is_df_belong_to_genre(df, input_genre_list):
    return df['track_genre'].isin(input_genre_list).values[0]


@st.cache()
def read_data():
    # read data, get 114000 rows
    data = pd.read_csv('backends/datasets/dataset.csv')
    # remove rows containing NANs
    data.dropna()
    # remove duplicates
    data = data.drop_duplicates(subset=['artists', 'track_name', 'album_name'], keep='first', inplace=False)

    return data


if 'data' not in st.session_state:
    st.session_state['data'] = read_data()

data = st.session_state['data']


def get_recommendation_ids(id_input, n_recommendation, input_genre_list, retrained):
    """
    :param id_input: id of input track
    :param num_recommendation: number of recommendations made
    :return: dataframe of recommended track ids
    """
    # from input id dataframe to string
    input_id_str = str(id_input.astype(str)).split()[2]

    # from input id to feature dataframe, and if the input track does not belong to the input genre, return
    df_input = data.loc[data.track_id == input_id_str]
    if not is_df_belong_to_genre(df_input, input_genre_list):
        return -1

    # from input genre df to array
    input_genre_list = np.array(input_genre_list)

    # the quantity of input genres
    n_genres = input_genre_list.shape[0]

    # Spotify features, loudness is removed because of high correlation with energy
    features_column_name = ['danceability', 'acousticness', 'speechiness',
                            'instrumentalness', 'liveness', 'energy']

    # genre column name
    genre_column_name = ['track_genre']

    # filter all the data
    filtered_data = filter_df_by_genre(data, input_genre_list)

    # get feature data
    features_all_df = get_features(filtered_data, features_column_name)
    # convert feature dataframe to feature ndarray
    features_all = df_to_ndarray(features_all_df)
    # scale the features to make them standard
    scaler = StandardScaler()
    features_all = scaler.fit_transform(features_all)

    # get genre data
    genres_all_df = get_features(filtered_data, genre_column_name)
    # one-hot encode the genres
    genres_all_code = pd.get_dummies(genres_all_df['track_genre'])
    # convert genre dataframe to feature ndarray
    genres_all = df_to_ndarray(genres_all_code)

    # split the feature data into train set and validation set, use 70% for training and 30% for validation
    x_train, x_test, y_train, y_test = train_test_split(features_all, genres_all, test_size=0.3)

    # reshape x_train and x_test
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # create the models if there is no fit history and compile and fit
    for file_walk in os.walk('backends/models'):
        for file in file_walk[2]:
            if 'model' in file:
                st.session_state['model'] = keras.models.load_model('backends/models/' + file)
                if st.session_state['model'].input_shape == (x_train.shape[1], 1):
                    if not retrained:
                        break
                    current_model_count = int(file.replace('model', '').replace('.h5', ''))
                    model = Sequential([
                        Conv1D(512, 3, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)),
                        BatchNormalization(),
                        MaxPooling1D(3, strides=2, padding='same'),
                        Conv1D(256, 3, strides=1, padding='same', activation='relu'),
                        BatchNormalization(),
                        MaxPooling1D(3, strides=2, padding='same'),
                        Conv1D(128, 3, strides=1, padding='same', activation='relu'),
                        BatchNormalization(),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.2),
                        Dense(128, activation='relu'),
                        Dropout(0.2),
                        Dense(128, activation='relu'),
                        Dropout(0.2),
                        Dense(n_genres, activation='softmax')
                    ])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
                    plot_model(model, to_file='backends/img/model' + str(current_model_count + 1) + '.png',
                               show_shapes=True)

                    history = model.fit(x_train, y_train, epochs=50, batch_size=128,
                                        validation_data=(x_test, y_test))

                    st.session_state['model'] = model
                    model.save('backends/models/model' + str(current_model_count + 1) + '.h5')

                    st.session_state['fig'] = plt.figure()
                    plt.plot(history.history['acc'])
                    plt.plot(history.history['val_acc'])
                    plt.legend(['acc', 'val_acc'], loc='upper left')
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.savefig('backends/img/acc' + str(current_model_count + 1) + '.png')
                    break

    if 'fig' in st.session_state:
        fig = st.session_state['fig']
        st.pyplot(fig, height=300)

    model = st.session_state['model']

    # get input features
    features_input = get_features(df_input, features_column_name)
    features_input = df_to_ndarray(features_input)
    # reshape input
    features_input = np.expand_dims(features_input, axis=-1)

    # make predictions to get similar music genre's one hot code
    pred = model.predict(features_input)
    similar_label = np.argmax(pred)

    # get similar music index df
    similar_index_df = genres_all_code[genres_all_code.iloc[:, similar_label] == 1]

    # pick up recommendation indices in similar music
    recommendation_indices = similar_index_df.sample(n_recommendation, replace=False).index

    # from recommendation indices to id dataframe
    recommendation_ids_df = data.loc[recommendation_indices, 'track_id']

    # from recommendation id dataframe to id list of str
    recommendation_ids = recommendation_ids_df.values.tolist()

    return recommendation_ids
