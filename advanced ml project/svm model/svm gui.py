import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import filedialog
from sklearn.linear_model import LinearRegression
import joblib

mean=[33.65733219,0,1.63039761, 72.76036768 , 8.11276186  ,2.00406156 , 4.87302266,0]
std=[ 7.6178177,1,0.87991671, 34.00311322 , 6.037298  ,  1.41896117  ,3.39801565,1]
class MLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Machine Learning Model GUI")

        self.model = joblib.load('svm_model4.pkl')  # Load your pre-trained regression model

        # Sample data
        self.data = pd.DataFrame(columns=[
            'Age',
            'Gender',
            'Education Level' ,
            'Job Title',
            'Years of Experience',
            'Country',
            'Race',
            'Senior' 
        ])

        self.labels = []
        self.entries = []

        self.create_input_fields()
        
        self.btn_predict = tk.Button(self, text="Predict", command=self.predict)
        self.btn_predict.grid(row=len(self.data.columns), columnspan=2, padx=5, pady=5)

        self.label_result = tk.Label(self, text="")
        self.label_result.grid(row=len(self.data.columns) + 1, columnspan=2, padx=5, pady=5)

    def create_input_fields(self):
        # Create labels and entry fields for each feature
        for i, column in enumerate(self.data.columns):
            label = tk.Label(self, text=column)
            label.grid(row=i, column=0, padx=5, pady=5)
            entry = tk.Entry(self)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.labels.append(label)
            self.entries.append(entry)

    def predict(self):
        # Get input values from entry fields
        input_values = [float(entry.get()) for entry in self.entries]
        
        # Scale input values
        scaled_values = [(val - mean_val) / std_val for val, mean_val, std_val in zip(input_values, mean, std)]
        
        input_array = np.array(scaled_values).reshape(1, -1)  # Reshape to match model input shape

        # Make prediction using the model
        prediction = self.model.predict(input_array)

        # Display prediction
        self.label_result.config(text=f"{int(prediction)}")


if __name__ == "__main__":
    app = MLApp()
    app.mainloop()
