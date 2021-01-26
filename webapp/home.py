import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image



def home():
    #st.set_page_config(layout="wide")
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    title = Image.open('title.PNG')
    st.image(title)
    poster = Image.open('poster.PNG')
    st.image(poster)
    st.sidebar.markdown('Created By:')
    st.sidebar.markdown("Aditya Sai Yerraguntla")
    st.sidebar.markdown("Hitesh Thadhani")
    st.sidebar.markdown("Surjit Singh")   
