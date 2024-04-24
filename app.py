import streamlit as st

# noinspection PyTypeChecker
st.set_page_config(page_title='home',
                   page_icon=':musical_note:',
                   layout='wide',
                   menu_items={
                       'Get Help': None,
                       'Report a bug': None,
                       'About': 'K. Wong'
                   }
                   )

exec(open("page_setting/setting.py").read())

import pandas as pd

from backends.data_api import get_data_with_info, get_info_from_df, get_id_by_info, \
    get_info_by_id, get_recommendation_ids, get_genre_list, filter_df_by_genre
from backends.spotipy_api import init_client, get_image_data_from_track_id, \
    get_external_url_from_track_id

# body
with st.container():
    st.title('Music Recommendation System')

    recommendation_number = st.slider(label='Please choose the number of recommendations:',
                                      value=1, min_value=1, max_value=200, step=1)

    input_genre_list = st.multiselect('Please choose genres of ' + str(get_genre_list().shape[0]) + ' genres:',
                                      get_genre_list())

    retrained = st.checkbox(' Retrain the model', value=False)

    if not input_genre_list or len(input_genre_list) < 2:
        st.error('Please choose at least two genres.')
    else:
        data_with_info = get_data_with_info()
        data_with_info = filter_df_by_genre(data_with_info, input_genre_list)
        track_info_list = get_info_from_df(data_with_info)

        input_track_info = st.selectbox(
            "Please choose a track from dataset of " + str(data_with_info.shape[0]) + " tracks:",
            track_info_list)

        st.session_state['is_start'] = st.button('Start')

        if not input_track_info:
            st.error('Please choose one track.')
        else:
            input_id = get_id_by_info(input_track_info)
            if st.session_state['is_start']:
                st.session_state['is_start'] = False
                with st.container():
                    recommendation_ids = get_recommendation_ids(input_id, recommendation_number, input_genre_list,
                                                                retrained)
                    if recommendation_ids == -1:
                        st.error('The song input does not fit the genre input.')
                    else:
                        for recommendation_id in recommendation_ids:
                            with st.container():
                                recommendation_info = get_info_by_id(recommendation_id)
                                if init_client():
                                    if get_image_data_from_track_id(recommendation_id):
                                        recommendation_image = get_image_data_from_track_id(recommendation_id)
                                        recommendation_info = pd.concat([recommendation_image, recommendation_info],
                                                                        axis=1)

                                    if get_external_url_from_track_id(recommendation_id):
                                        recommendation_external_url = get_external_url_from_track_id(recommendation_id)
                                        recommendation_info = pd.concat(
                                            [recommendation_info, recommendation_external_url],
                                            axis=1)
                                else:
                                    st.error(
                                        'There\'s something wrong with the client. Please refresh or check your network.')

                                st.write(recommendation_info)
