from cProfile import label
import time
from turtle import width
import model
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

def evalts1 (df):

    xvar = df['text(string)']
    yvar = df['label']

    start = time.time()
    y_predval=model.modeleval(xvar)
    end = time.time()

    #Prediction
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Prediction Table :</h3>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Table (
        columnwidth = [700,700,700],
        header=dict(values=['Message','Actual', 'Predicted'], fill_color='paleturquoise', font=dict(color='black'), font_size = 20),
        cells=dict(values=[xvar, yvar, y_predval], fill_color='lavender', font=dict(color='black'), font_size = 14, height=30))
        ])
    st.plotly_chart(fig, use_container_width=True)

    #Classification Report
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Classification Report :</h3>", unsafe_allow_html=True)
    clfr = classification_report(yvar,y_predval)
    st.markdown(clfr)

    #Confusion Matrix
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Heatmap :</h3>", unsafe_allow_html=True)
    confusionmatrix = confusion_matrix(yvar,y_predval)
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(confusionmatrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    st.pyplot(fig)

    # Accuracy
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Accuracy :</h3>", unsafe_allow_html=True)
    accuracy = accuracy_score(yvar, y_predval)
    st.write('Accuracy-',accuracy)

    # Recall
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Recall :</h3>", unsafe_allow_html=True)
    from sklearn.metrics import recall_score
    recall = recall_score(yvar, y_predval, average=None)
    st.write('Anger-',recall[0])
    st.write('Fear-',recall[1])
    st.write('Joy-',recall[2])
    st.write('Love-',recall[3])
    st.write('Sadness-',recall[4])
    st.write('Surprise-',recall[5])

    # Precision
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Precision :</h3>", unsafe_allow_html=True)
    from sklearn.metrics import precision_score
    precision = precision_score(yvar, y_predval, average=None) 
    st.write('Anger-',precision[0])
    st.write('Fear-',precision[1])
    st.write('Joy-',precision[2])
    st.write('Love-',precision[3])
    st.write('Sadness-',precision[4])
    st.write('Surprise-',precision[5])

    #F-1 Score
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>F-1 Score :</h3>", unsafe_allow_html=True)
    f1s = 2 * (precision * recall) / (precision + recall)
    st.write('Anger-',f1s[0])
    st.write('Fear-',f1s[1])
    st.write('Joy-',f1s[2])
    st.write('Love-',f1s[3])
    st.write('Sadness-',f1s[4])
    st.write('Surprise-',f1s[5])

    #Time Elapsed
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Time Elapsed :</h3>", unsafe_allow_html=True)
    diff=(end-start)
    diffms = diff*1000
    st.write("Time Taken to Evalueate Model:",diffms, "ms")

    #LineChart
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Line Chart Comparison :</h3>", unsafe_allow_html=True)
    df_pred = pd.DataFrame({"Actual" : yvar  ,  "Predicted" : y_predval , "Message" : xvar  } )
    df_lc =df_pred.sample(50)
    df_lc['S.no'] = np.arange(len(df_lc)) 
    fig = px.line(df_lc, x="S.no", y=["Actual","Predicted"], width = 1000)
    st.plotly_chart(fig, use_container_width=True)
    fig = plt.figure()

    #Countplot
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Countplot Comparison :</h3>", unsafe_allow_html=True)
    sns.countplot(df_pred['Actual'])  
    plt.title("Countplot of Actual Emotions")    
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(df_pred['Predicted']) 
    plt.title("Countplot of Predicted Emotions")    
    st.pyplot(fig)



def evalts2 (df):
    xvar = df['text(string)']
    df['textlabel']=df.label.replace({0:"sadness",1:"joy",2:"love",3:"anger",4:"fear",5:"surprise"})
    yvar = df['textlabel']

    start = time.time()
    y_predval=model.modeleval(xvar)
    end = time.time()

    #Prediction
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Prediction Table :</h3>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Table (
        columnwidth = [700,700,700],
        header=dict(values=['Message','Actual', 'Predicted'], fill_color='paleturquoise', font=dict(color='black'), font_size = 20),
        cells=dict(values=[xvar, yvar, y_predval], fill_color='lavender', font=dict(color='black'), font_size = 14, height=30))
        ])
    st.plotly_chart(fig, use_container_width=True)

    #Classification Report
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Classification Report :</h3>", unsafe_allow_html=True)
    clfr = classification_report(yvar,y_predval)
    st.markdown(clfr)

    #Confusion Matrix
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>HeatMap :</h3>", unsafe_allow_html=True)
    confusionmatrix = confusion_matrix(yvar,y_predval)
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(confusionmatrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    st.pyplot(fig)

    # Accuracy
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Accuracy :</h3>", unsafe_allow_html=True)
    accuracy = accuracy_score(yvar, y_predval)
    st.write('Accuracy-',accuracy)

    # Recall
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Recall :</h3>", unsafe_allow_html=True)
    from sklearn.metrics import recall_score
    recall = recall_score(yvar, y_predval, average=None)
    st.write('Anger-',recall[0])
    st.write('Fear-',recall[1])
    st.write('Joy-',recall[2])
    st.write('Love-',recall[3])
    st.write('Sadness-',recall[4])
    st.write('Surprise-',recall[5])

    # Precision
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Precision :</h3>", unsafe_allow_html=True)
    from sklearn.metrics import precision_score
    precision = precision_score(yvar, y_predval, average=None) 
    st.write('Anger-',precision[0])
    st.write('Fear-',precision[1])
    st.write('Joy-',precision[2])
    st.write('Love-',precision[3])
    st.write('Sadness-',precision[4])
    st.write('Surprise-',precision[5])

    #F-1 Score
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>F-1 Score :</h3>", unsafe_allow_html=True)
    f1s = 2 * (precision * recall) / (precision + recall)
    st.write('Anger-',f1s[0])
    st.write('Fear-',f1s[1])
    st.write('Joy-',f1s[2])
    st.write('Love-',f1s[3])
    st.write('Sadness-',f1s[4])
    st.write('Surprise-',f1s[5])

    #Time Elapsed
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Time Elapsed :</h3>", unsafe_allow_html=True)
    diff=(end-start)
    diffms = diff*1000
    st.write("Time Taken to Evalueate Model:",diffms, "ms")

    #LineChart
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Line Chart Comparison :</h3>", unsafe_allow_html=True)
    df_pred = pd.DataFrame({"Actual" : yvar  ,  "Predicted" : y_predval , "Message" : xvar  } )
    df_lc =df_pred.sample(50)
    df_lc['S.no'] = np.arange(len(df_lc)) 
    fig = px.line(df_lc, x="S.no", y=["Actual","Predicted"], width = 1000)
    st.plotly_chart(fig, use_container_width=True)
    fig = plt.figure()

    #Countplot
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Countplot Comparison :</h3>", unsafe_allow_html=True)
    sns.countplot(df_pred['Actual'])  
    plt.title("Countplot of Actual Emotions")    
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(df_pred['Predicted']) 
    plt.title("Countplot of Predicted Emotions")    
    st.pyplot(fig)