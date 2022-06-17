import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import pickle
import sklearn


first_part = st.container()
data_set = st.container()
inputs = st.container()
modeling = st.container()

with first_part : 
    st.title("Program is to predict whether a person survived in Sinking of the RMS Titanic or not")
    st.markdown("* **Fisrt Part** You can understand the DataSet")
    st.markdown("* **Second Part** You can give inputs to find out that if this person survived or not")
    
with data_set : 
    st.header("The Titanic Dataset")
    df = pd.read_csv("X_test Dataset")
    st.write(df.head())

st.sidebar.header("Inputs Giving") 
pclass = st.sidebar.selectbox("Please select input for the Pclass vairable" , options = [1,2,3] , index = 0)
sex = st.sidebar.selectbox("Please select input for the Sex variable" , options = ["male" , "female"] , index = 0)
age = st.sidebar.slider("Please select input for the Age variable" , min_value = 1 , max_value = 71 , value = 1 , step = 1)
sibsp = st.sidebar.selectbox("Please selet input for the Sibsp variable" , options = [0,1,2,3,4] , index = 0)
parch = st.sidebar.selectbox("Please select input for the Parch variable" , options = [0,1,2,3,4,5,6] , index = 0)
fare = st.sidebar.slider("Please select input for the Fare variable" , min_value = 0 , max_value = 513, value = 0 , step = 1)
who = st.sidebar.selectbox("Please select input for the Who variable" , options = ["man" , "woman" , "child"] , index = 0)
adult = st.sidebar.selectbox("Please select input for the Adult_male variable" , options = ["Yes" , "No"])
embark_town = st.sidebar.selectbox("Please select input for the Embark Town variable" , options = ['Southampton', 'Cherbourg', 'Queenstown'] , index = 0)
alive = st.sidebar.selectbox("Please select input for the Alive variable" , options = ["no" , "yes"] , index = 0)
alone = st.sidebar.selectbox("Please select input for the Alone variable" , options = ["No" , "Yes"] , index = 0)


with inputs : 
    st.header("You have entered these inputs")
    data_input = pd.DataFrame(data = {"Pclass" : [pclass], "Sex" : [sex],
                                  "Age" : [age] , "Sibsp" : [sibsp],
                                  "Parch" : [parch] , "Fare" : [fare],
                                  "Who" : [who] , "Adult_male" : [adult],
                                  "Embark_town" : [embark_town] , "Alive" : [alive],
                                  "Alone" : [alone]})
    st.write(data_input)

btn = st.sidebar.button("Predict")

if btn : 
    model = pickle.load(open("Model saved with pickle" , "rb"))
    y_pred = model.predict(X = pd.DataFrame(data = {"Pclass" : [pclass], "Sex" : [sex],
                                  "Age" : [age] , "Sibsp" : [sibsp],
                                  "Parch" : [parch] , "Fare" : [fare],
                                  "Who" : [who] , "Adult_male" : [adult],
                                  "Embark_town" : [embark_town] , "Alive" : [alive],
                                  "Alone" : [alone]}))
    y_pred_prob = model.predict_proba(X = pd.DataFrame(data = {"Pclass" : [pclass], "Sex" : [sex],
                                  "Age" : [age] , "Sibsp" : [sibsp],
                                  "Parch" : [parch] , "Fare" : [fare],
                                  "Who" : [who] , "Adult_male" : [adult],
                                  "Embark_town" : [embark_town] , "Alive" : [alive],
                                  "Alone" : [alone]}))

    
    if y_pred == [1] : 
        st.markdown("This person **Survived** in titanic accident and probablity is {}".format(np.round(y_pred_prob[:,1][0])*100),2)
    else : 
        st.markdown("This person **Died** in titanic accident and probablity is {}".format(np.round(y_pred_prob[: , 0][0])*100),2)
    

    