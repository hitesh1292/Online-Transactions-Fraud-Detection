import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
from sklearn.metrics import accuracy_score,recall_score,classification_report,precision_score,roc_auc_score,f1_score,roc_curve, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve




def main():
        
    #st.set_page_config(layout="wide")
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    title = Image.open('title.PNG')
    st.image(title)
    st.sidebar.markdown('Created By:')
    st.sidebar.markdown("Aditya Sai Yerraguntla")
    st.sidebar.markdown("Hitesh Thadhani")
    st.sidebar.markdown("Surjit Singh")
    rf_grid = pd.read_csv("grid_otp_rf.csv")
    rf_X_test = pd.read_csv("X_test_over_imp.csv")
    rf_X_test = rf_X_test.drop(columns=['Unnamed: 0'])
    rf_X_test = rf_X_test.dropna()
    rf_y_test = pd.read_csv("y_test_over_imp.csv")
    rf_y_test = rf_y_test["isFraud"]
    
    rf_max_dept_list = list(set(rf_grid["param_max_depth"]))
    rf_max_feat_list = list(set(rf_grid["param_max_features"]))
    rf_min_samp_leaf_list = list(set(rf_grid["param_min_samples_leaf"]))
    rf_min_samp_split_list = list(set(rf_grid["param_min_samples_split"]))
    rf_n_est = list(set(rf_grid["param_n_estimators"]))
    
    rf_model = pickle.load(open('logoptm.pkl', 'rb'))
    y_score = rf_model.predict_proba(rf_X_test)[:,1]
    fpr, tpr, threshold = roc_curve(rf_y_test, y_score)
    #print('roc_auc_score for LR: ', roc_auc_score(lr_y_test, y_score))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    col1, col2, col3 = st.beta_columns([1,1,2])
    
    with col1:
        max_dept = st.selectbox('Max Depth', rf_max_dept_list)
        max_feat = st.selectbox('Max Features', rf_max_feat_list)
        min_leaf = st.selectbox('Min Samples Leaf', rf_min_samp_leaf_list)
        min_split = st.selectbox('Min Samles Split', rf_min_samp_split_list)
        n_est = st.selectbox('N Estimators', rf_n_est)
        submit = st.button("submit",key="submit")
    

    auc = list(rf_grid[(rf_grid["param_max_depth"]==max_dept) \
                       & (rf_grid["param_max_features"]==max_feat) \
                       & (rf_grid["param_min_samples_leaf"]==min_leaf) \
                       & (rf_grid["param_min_samples_split"]==min_split) \
                       & (rf_grid["param_n_estimators"]==n_est)].iloc[:,-1])
    with col2:
        st.write("Mean AUC Score:")
        if not submit:
            st.write("Click on Submit")
        if submit:
            
            if auc:
                output = "<div><span class='highlight blue'><span class='bold'>"+ str(round(auc[0],6))+"</span> </span></span></div>"
                #st.write(round(auc[0],6))
                st.write(output,unsafe_allow_html=True)
            else:
                output = "<div><span class='highlight blue'><span class='bold'>NA</span> </span></span></div>"
                #st.write(round(auc[0],6))
                st.write(output,unsafe_allow_html=True)
    
    with col3:
        plt.subplots(1, figsize=(5,5))
        plt.title('Receiver Operating Characteristic - Random Forest')
        plt.plot(fpr, tpr,linewidth=2, label="Random Forest Classifier : AUC = 0.865513")
        plt.plot(fpr, tpr)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.legend(loc = 'lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')    
        st.subheader("ROC Curve for Best Model")
        st.pyplot()

