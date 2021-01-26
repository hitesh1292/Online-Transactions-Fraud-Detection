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
    lr_grid = pd.read_csv("grid_otp_lr.csv")
    lr_X_test = pd.read_csv("X_test_over_imp.csv")
    lr_X_test = lr_X_test.drop(columns=['Unnamed: 0'])
    lr_X_test = lr_X_test.dropna()
    lr_y_test = pd.read_csv("y_test_over_imp.csv")
    lr_y_test = lr_y_test["isFraud"]
    lr_c_list = list(set(lr_grid["param_C"]))
    lr_iter_list = list(set(lr_grid["param_max_iter"]))
    lr_model = pickle.load(open('logoptm.pkl', 'rb'))
    y_score = lr_model.predict_proba(lr_X_test)[:,1]
    fpr, tpr, threshold = roc_curve(lr_y_test, y_score)
    #print('roc_auc_score for LR: ', roc_auc_score(lr_y_test, y_score))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    
    col1, col2, col3 = st.beta_columns([1,1,2])
    
    with col1:
        c = st.selectbox('C', lr_c_list)
        max_iter = st.selectbox('Max Iterations', lr_iter_list)
        submit = st.button("submit",key="submit")
    

    auc = list(lr_grid[(lr_grid["param_C"]==c) & (lr_grid["param_max_iter"]==max_iter)].iloc[:,-1])    
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
        plt.title('Receiver Operating Characteristic - Logistic Regression')
        plt.plot(fpr, tpr,linewidth=2, label="Logistic Classifier : AUC = 0.823409")
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        #plt.plot(fpr, tpr, )
        plt.legend(loc = 'lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')    
        st.subheader("ROC Curve for Best Model")
        st.pyplot()
        
        

        #st.markdown(t, unsafe_allow_html=True)
#main()

