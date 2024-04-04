import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

st.title("Iris Flower Classification")
st.text("")
df = pd.read_csv('iris.csv')
flower = st.sidebar.multiselect("Filter by Flower", options=df['species'].unique(), default=df['species'][0])
df_filtered = df.query("species == @flower")
st.dataframe(df_filtered.sample(5))
st.text("")
st.text("")
df_chart = df_filtered.sample(50)
st.bar_chart(data=df_chart.iloc[:, :4], use_container_width=True)

x = df.iloc[:, :4]
y = df.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=110)

model = LogisticRegression()
model.fit(x_train, y_train)

file = open('model.pkl', 'wb')
pickle.dump(model, file)

s_length = st.slider('Sepal Length', float(df['sepal_length'].min()), float(df['sepal_length'].max()), float(df['sepal_length'].mean()), step=0.1)
s_width = st.slider('Sepal Width', float(df['sepal_width'].min()), float(df['sepal_width'].max()), float(df['sepal_width'].mean()), step=0.1)
p_length = st.slider('Petal Length', float(df['petal_length'].min()), float(df['petal_length'].max()), float(df['petal_length'].mean()), step=0.1)
p_width = st.slider('Petal Width', float(df['petal_width'].min()), float(df['petal_width'].max()), float(df['petal_width'].mean()), step=0.1)

button = st.button('Predict')
if button:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Preprocess the input data to match the format used during training
    input_features = [[s_length, s_width, p_length, p_width]]
    y_pred = model.predict(input_features)
    if y_pred[0] == 'Iris-setosa':
        st.subheader(y_pred[0])
        st.image('images/setosa.jpg', width=300)
    elif y_pred[0] == 'Iris-virginica':
        st.subheader(y_pred[0])
        st.image('images/virginica.jpg', width=300)
    elif y_pred[0] == 'Iris-versicolor':
        st.subheader(y_pred[0])
        st.image('images/versicolor.jpg', width=300)
