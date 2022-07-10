import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import neattext.functions as nfx
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

def modeleval(xlabel) :
    df_trainmodel= pd.read_csv("Dataset//train.csv")
    #User handles  
    df_trainmodel['Clean_Text'] = df_trainmodel['text(string)'].apply(nfx.remove_userhandles)
    # Stopwords
    df_trainmodel['Clean_Text'] = df_trainmodel['Clean_Text'].apply(nfx.remove_stopwords)
    xlabeltrain = df_trainmodel['Clean_Text']
    ylabeltrain = df_trainmodel['label']
    pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
    pipe_lr.fit(xlabeltrain,ylabeltrain)
    y_predval=pipe_lr.predict(xlabel)
    return y_predval
