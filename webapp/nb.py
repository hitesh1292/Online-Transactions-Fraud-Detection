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
    nb_grid = pd.read_csv("grid_otp_nb.csv")
    nb_X_test = pd.read_csv("X_test_over_imp.csv")
    nb_X_test = nb_X_test.drop(columns=['Unnamed: 0'])
    nb_X_test = nb_X_test.dropna()
    nb_y_test = pd.read_csv("y_test_over_imp.csv")
    nb_y_test = nb_y_test["isFraud"]
    nb_alpha_list = list(set(nb_grid["param_alpha"]))
    nb_fit_prior_list = list(set(nb_grid["param_fit_prior"]))
    nb_model = pickle.load(open('nboptm.pkl', 'rb'))
    y_score = nb_model.predict_proba(nb_X_test)[:,1]
    fpr, tpr, threshold = roc_curve(nb_y_test, y_score)
    print('roc_auc_score for MNB: ', roc_auc_score(nb_y_test, y_score))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    col1, col2, col3 = st.beta_columns([1,1,2])
    
    with col1:
        alpha = st.selectbox('alpha', nb_alpha_list)
        fit_prior = st.selectbox('fit_prior', nb_fit_prior_list)
        submit = st.button("submit",key="submit")
    
    st.write()
        
    with col2:
        st.write("Mean AUC Score:")
        if not submit:
            st.write("Click on Submit")
        if submit:
            auc = list(nb_grid[(nb_grid["param_alpha"]==alpha) & (nb_grid["param_fit_prior"]==fit_prior)].iloc[:,-1])
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
        plt.title('Receiver Operating Characteristic - Multinomial Naive Bayes')
        plt.plot(fpr, tpr,linewidth=2, label="Naive Bayes Classifier : AUC = 0.750583")
        #plt.plot(fpr, tpr)
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.legend(loc = 'lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')    
        st.subheader("ROC Curve for Best Model")
        st.pyplot()

#main()