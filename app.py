import gradio as gr
import pandas as pd
import pickle


with open("insurance_ridge_pipeline.pkl","rb") as file:
    model = pickle.load(file)

def predict_charge(age, sex, bmi, children, smoker, region):


    input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                            columns=['age','sex','bmi','children','smoker','region'])
    
  

    input_df['age_bmi_interaction'] = input_df['age'] * input_df['bmi']
    
   
   
    prediction = model.predict(input_df)[0]
    return prediction



inputs = [
    gr.Number(label="Age", value=30),
    gr.Radio(["female", "male"], label="Sex"),
    gr.Number(label="BMI", value=25.0),
    gr.Number(label="Children", value=1),
    gr.Radio(["yes","no"], label="Smoker"),
    gr.Radio(["southwest", "southeast", "northwest", "northeast"], label="Region")
]



app = gr.Interface(
    fn=predict_charge,
    inputs=inputs,
    outputs="number",
    title="Insurance Charges Predictor"
)


app.launch()
