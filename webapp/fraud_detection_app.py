import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from multiapp import MultiApp
import lr,rf,xgb,nb,home


st.set_page_config(layout="wide")
app = MultiApp()

app.add_app("Home", home.home)
app.add_app("Logistic Regression", lr.main)
app.add_app("Random Forest", rf.main)
app.add_app("Extreme Gradient Boosting", xgb.main)
app.add_app("Naive Bayes", nb.main)



app.run()
