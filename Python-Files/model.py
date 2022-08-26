import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

df_trainmodel= pd.read_csv("Dataset//train.csv")
#User handles  
df_trainmodel['Clean_Text'] = df_trainmodel['text(string)'].apply(nfx.remove_userhandles)
# Stopwords
df_trainmodel['Clean_Text'] = df_trainmodel['Clean_Text'].apply(nfx.remove_stopwords)
xlabeltrain = df_trainmodel['Clean_Text']
ylabeltrain = df_trainmodel['label']

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(xlabeltrain,ylabeltrain)

def modeleval(xlabel) :
    y_predval=pipe_lr.predict(xlabel)
    return y_predval
