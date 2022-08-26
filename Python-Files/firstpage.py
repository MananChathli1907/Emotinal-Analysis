from logging import PlaceHolder
import streamlit as st
import pandas as pd
import webbrowser
import customtest
import algosummary
import testsample
import more

st.markdown("<h1 style='text-align: center; color: lightblue;'>Emotional Analysis Model</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: skyblue;'>Web Application for Emotion Detection via Predictive Modelling</h4>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: yellow ;'>⚡Model⚡</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 3, 3])
b1 = col1.button('Expression 1 📇',key=1)
b2 = col2.button('Expression 2 📇',key=2)
b3 = col3.button('Expression 3 📇',key=3)  

st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)
placeholder = st.empty()
input = placeholder.text_input('🖥️ Emotional Analysis ', key=4)
if b1 :
    input = placeholder.text_input('🖥️ Emotional Analysis ',value = 'I feel greedy wrong',key=5)
    st.markdown("<h6 style='text-align: left; color: white ;'>Emotion : Anger 😡</h6>", unsafe_allow_html=True)
if b2 :
    input = placeholder.text_input('🖥️ Emotional Analysis ',value = 'I dance I should feel pretty',key=5)
    st.markdown("<h6 style='text-align: left; color: white ;'>Emotion : Joy 😄</h6>", unsafe_allow_html=True)
if b3 :
    input = placeholder.text_input('🖥️ Emotional Analysis ',value = 'I am left feeling impressed by more than a few companies',key=5)
    st.markdown("<h6 style='text-align: left; color: white ;'>Emotion : Surprise 😮</h6>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: yellow ;'></h1>", unsafe_allow_html=True)
st.markdown("<h6 style='color: skyblue;'>Description: </h6>", unsafe_allow_html=True)
st.write("Emotion analysis is the process of identifying and analyzing the underlying emotions expressed in textual data. Emotion analytics can extract", 
        "the text data from multiple sources to analyze the subjective information and understand the emotions behind it.",
        "This project focuses on doing the same by taking a text message as an input and analysing it to deduce the emotion behind the message written",
        "This has application in feedback analysis as simple comments can indicate whether the user is satisfied or unsatisfied, whether the feedback is ",
        "positive or negative, and should be shown.")

rad =st.sidebar.radio("Navigation",["Home","Datasets","Evaluation Sample-1","Evaluation Sample-2","Custom Input", "Learn More"])

if rad == "Datasets":
    st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: white;'>Train Dataset</h1>", unsafe_allow_html=True)
    df_train= pd.read_csv("Dataset//train.csv")
    algosummary.algosumdisplay(df_train)
    url = 'C:\\Users\\91982\\Desktop\\NTCC_MANAN\\datasetsummary\\train.html';
    if st.button('View Summary Train DataSet'):
        webbrowser.open_new_tab(url)

    st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: white;'>Test Sample-1</h1>", unsafe_allow_html=True)
    data = pd.read_csv("Dataset//test.csv")
    df = data.sample(50)
    df.rename(columns = {'text(string)' : 'Message', 'label' : 'Emotion'}, inplace = True)
    st.dataframe(df,width=840,height= 300)
    url = 'C:\\Users\\91982\\Desktop\\NTCC_MANAN\\datasetsummary\\test.html';
    if st.button('View Summary Test Sample 1'):
        webbrowser.open_new_tab(url)
    
    st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; color: white;'>Test Sample-2</h1>", unsafe_allow_html=True)
    data = pd.read_csv("Dataset//customtest.csv")
    df = data.sample(50)
    df.rename(columns = {'text(string)' : 'Message', 'label' : 'Emotion'}, inplace = True)
    st.dataframe(df,width=840,height= 300)
    url = 'C:\\Users\\91982\\Desktop\\NTCC_MANAN\\datasetsummary\\test.html';
    if st.button('View Summary Test Sample 2'):
        webbrowser.open_new_tab(url)
    
if rad == "Evaluation Sample-1":
    df_test1 = pd.read_csv("Dataset//test.csv")
    testsample.evalts1(df_test1)
       
if rad == "Evaluation Sample-2":
    df_test2 = pd.read_csv("Dataset//customtest.csv")
    testsample.evalts2(df_test2)

if rad == "Custom Input":
    customtest.custompredict()
    
if rad == "Learn More":
    more.more()
