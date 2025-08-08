import streamlit as st
import pandas as pd
import joblib

st.title("Iris Project")
st.write("This is a simple Streamlit app that displays the Iris dataset.")

# Inference Function
# == Inference Function
model = joblib.load("model.joblib")
def get_prediction(data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# User Input
left, right = st.columns(2, gap="medium", border=True)
# -- Sepal Input
left.subheader("Sepal")
sepal_length = left.slider('Sepal Length', min_value=1.0, max_value=10.0, value=5.4, step=0.1)
sepal_width = left.slider('Sepal Width', min_value=1.0, max_value=10.0, value=5.4, step=0.1)

# -- Petal Input
right.subheader("Petal")
petal_length = right.slider('Petal Length', min_value=1.0, max_value=10.0, value=5.4, step=0.1)
petal_width = right.slider('Petal Width', min_value=1.0, max_value=10.0, value=5.4, step=0.1)

data = pd.DataFrame({"sepal length (cm)": [sepal_length],
                     "sepal width (cm)": [sepal_width], 
                     "petal length (cm)": [petal_length], 
                     "petal width (cm)": [petal_width]})
st.dataframe(data, use_container_width=True)

# Prediction Button
button = st.button("Predict", use_container_width=True)
if button:
    st.write("Prediksi Berhasil !")
    pred, pred_proba = get_prediction(data)

    label_map = {0: "Setosa", 1: "Versicolor",2: "Virginica"}
    
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    output = f"Iris Anda diklasifikasikan sebagai {label_proba:.0%} {label_pred}"
    st.write(output)