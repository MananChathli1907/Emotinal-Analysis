from turtle import width
import streamlit as st
import model
import time


def custompredict() :
    
    st.markdown("<h3 style='text-align: left; color: white;</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; color: white;'>Custom Prediction :</h3>", unsafe_allow_html=True)\

    placeholder = st.empty()
    input = placeholder.text_input('Please enter the text here', key=1)
    showpred  = st.button('Show Prediction', key=4)
    click_clear = st.button('Refresh', key=3)

    if click_clear:
        input = placeholder.text_input('text', value='', key=2)

    if showpred :
        xvar = {input}
        start = time.time()
        prediction = model.modeleval(xvar)
        end = time.time()
        diff=(end-start)
        diffms = diff*1000

        if(prediction[0]=="joy")  :
            st.markdown("<h3 style='text-align: center; color: white;'>Predicted Emotion is Joy</h3>", unsafe_allow_html=True)
            from PIL import Image
            image = Image.open('joy.jpeg')
            col1, col2, col3 = st.columns([3, 3, 3])
            col2.image(image, use_column_width = True)

        elif(prediction[0]=="sadness") :
            st.markdown("<h3 style='text-align: center; color: white;'>Predicted Emotion is Sadness</h3>", unsafe_allow_html=True)
            from PIL import Image
            image = Image.open('sad.jpeg')
            col1, col2, col3 = st.columns([3, 3, 3])
            col2.image(image, use_column_width = True)

        elif(prediction[0]=="anger")  :
            st.markdown("<h3 style='text-align: center; color: white;'>Predicted Emotion is Anger</h3>", unsafe_allow_html=True)
            from PIL import Image
            image = Image.open('anger.jpeg')
            col1, col2, col3 = st.columns([3, 3, 3])
            col2.image(image, use_column_width = True)

        elif(prediction[0]=="fear")  :
            st.markdown("<h3 style='text-align: center; color: white;'>Predicted Emotion is Fear</h3>", unsafe_allow_html=True)
            from PIL import Image
            image = Image.open('fear.jpeg')
            col1, col2, col3 = st.columns([3, 3, 3])
            col2.image(image, use_column_width = True)

        elif(prediction[0]=="surprise")  :
            st.markdown("<h3 style='text-align: center; color: white;'>Predicted Emotion is Surprise</h3>", unsafe_allow_html=True)
            from PIL import Image
            image = Image.open('surprise.jpeg')
            col1, col2, col3 = st.columns([3, 3, 3])
            col2.image(image, use_column_width = True)

        elif(prediction[0]=="love")  :
            st.markdown("<h3 style='text-align: center; color: white;'>Predicted Emotion is Love</h3>", unsafe_allow_html=True)
            from PIL import Image
            image = Image.open('love.jpeg')
            col1, col2, col3 = st.columns([3, 3, 3])
            col2.image(image, use_column_width = True)

        # image = Image.open('joy.jpeg')
        st.write("Message :",input)
        st.write("The sentence is predicted to depict the emotion : ", prediction[0])
        st.write("Time Taken to predict: ",diffms, "ms")
