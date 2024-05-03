import tkinter as tk
import pandas as pd
import numpy as np
from tkinter import filedialog
from sklearn.tree import DecisionTreeClassifier
import joblib


class MLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Machine Learning Model GUI")

        self.model = joblib.load('decision_tree_model.pkl')  # Load your pre-trained decision tree model
        self.data = pd.DataFrame(columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])

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
        input_array = np.array(input_values).reshape(1, -1)  # Reshape to match model input shape

        # Make prediction using the model
        prediction = self.model.predict(input_array)

        # Display prediction
        if prediction >0.5:
            self.label_result.config(text=f"Survived")
        else:
            self.label_result.config(text=f"did not Survived")

if __name__ == "__main__":
    app = MLApp()
    app.mainloop()
