
import pandas as pd
import numpy as np
import neattext.functions as nfx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


def algosumdisplay(df) :
    
    #User handles  
    df['Clean_Text'] = df['text(string)'].apply(nfx.remove_userhandles)
    # Stopwords
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

    #Dataframe
    st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: white;'>Dataset in Use</h2>", unsafe_allow_html=True)
    df_disp = df.sample(100)
    df_disp.rename(columns = {'text(string)' : 'Message', 'label' : 'Emotion', 'Clean_Text' : 'Summary'}, inplace = True)
    st.dataframe(df_disp,width=840,height= 300)
    st.write("Truncated to 100 rows")

    #Data Visualization
    st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: white;'>Visualization of current data :</h2>", unsafe_allow_html=True)

    #Countplot
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Countplot :</h3>", unsafe_allow_html=True)
    fig = plt.figure(figsize=(10, 4))
    sns.countplot(df['label'])  
    plt.title("Countplot of Emotions")    
    st.pyplot(fig)

    #Scatterplot
    st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Scatterplot :</h3>", unsafe_allow_html=True)
    df_scatterplot = df.sample(100)
    df_scatterplot['S.no'] = np.arange(len(df_scatterplot)) 
    df_scatterplot['numlabel']=df_scatterplot.label.replace({"sadness":1, "joy":2, "love":3, "anger":4, "fear":5, "surprise":6})
    fig = px.scatter(df_scatterplot,x="S.no", y="label", color="label",hover_data=['Clean_Text'])
    fig.update_layout(title_text='Scatterplot of Emotions', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    #Pie Chart
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Pie Chart :</h3>", unsafe_allow_html=True)
    fig = px.pie(df_scatterplot, values='numlabel', names='label')
    fig.update_layout(title_text='Distribution of Emotions', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    #Pairplot
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Pairplot :</h3>", unsafe_allow_html=True)
    figr = sns.pairplot(df_scatterplot, x_vars="S.no", y_vars="numlabel",kind="kde")
    st.pyplot(figr)

    #Linechart
    st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Line Chart :</h3>", unsafe_allow_html=True)
    fig = px.line(df_scatterplot, x="S.no", y="numlabel")
    st.plotly_chart(fig, use_container_width=True)