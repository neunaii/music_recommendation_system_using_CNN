import pandas as pd

import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import requests
from PIL import Image
from io import BytesIO

SPOTIPY_CLIENT_ID = '-1'
SPOTIPY_CLIENT_SECRET = '-1'
SPOTIFY_SCOPE = 'user-library-read'

auth_manager = None
sp = None


def init_client():
    global auth_manager, sp

    try:
        auth_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_SECRET,
                                                client_secret=SPOTIPY_CLIENT_SECRET)
        sp = spotipy.Spotify(auth_manager=auth_manager)

        if auth_manager and sp:
            return True
    except (spotipy.SpotifyException, spotipy.SpotifyOauthError):
        return False


def get_image_data_from_track_id(track_id):
    global sp
    try:
        track = sp.track(track_id)
        image_url = track['album']['images'][0]['url']
        response = requests.get(image_url)
        content = response.content
        pil = Image.open(BytesIO(content))
        image_df = pd.DataFrame({'album_image': [pil]})
        return image_df
    except (spotipy.SpotifyException, spotipy.SpotifyOauthError) as e:
        return None


def get_external_url_from_track_id(track_id):
    global sp
    try:
        id_str = str(track_id)
        track = sp.track(id_str)
        external_url = track['external_urls']['spotify']
        external_url_df = pd.DataFrame({'external_url': [external_url]})
        return external_url_df
    except (spotipy.SpotifyException, spotipy.SpotifyOauthError) as e:
        return None
