# Description: This script is used to predict the price of a diamond based on its carat, x, y, and z values.
# Authors: Kaedyn Quinn (u3261368) and Tom Hawke (u3239388)
# Tutorial Times: Tom - Wednesday 9:30 am, Kaedyn - Thursday 1:30 pm

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class diamondPricePrediction:
    def __init__(self, master):
        self.master = master
        self.master.title('Diamond Price Prediction')

        self.data = pd.read_csv('diamonds.csv')
        self.logData["price"] = np.log(self.data["price"])
        self.logData = self.logData.drop(["Unnamed: 0"], axis=1)
        self.logData = self.logData[(self.logData["y"] < 30)]
        self.logData = self.logData[(self.logData["z"] < 30) & (self.logData["z"] > 2)]
        self.logData = self.logData[(self.logData["table"] < 80) & (self.logData["table"] > 40)]
        self.logData = self.logData[(self.logData["depth"] < 75) & (self.logData["depth"] > 45)]
        self.objects = (self.logData.dtypes == "object")
        self.categories = list(self.objects[self.objects].index)
        self.numData = self.logData.copy()
        self.sliders = []

        self.makeNumData()

        self.finalData = self.numData.drop(["price"], axis = 1)
        self.finalData = self.finalData.drop(["cut"], axis = 1)
        self.finalData = self.finalData.drop(["clarity"], axis = 1)
        self.finalData = self.finalData.drop(["color"], axis = 1)
        self.finalData = self.finalData.drop(["table"], axis = 1)
        self.finalData = self.finalData.drop(["depth"], axis = 1)

        self.predictors = self.finalData.values
        self.target = self.numData['price'].values

        self.predictors_train, self.predictors_test, self.target_train, self.target_test = train_test_split(self.predictors, self.target, test_size=0.3, random_state=42)

        self.model = XGBRegressor()
        self.model.fit(self.predictors_train, self.target_train)

        self.createWidgets()

    def makeNumData(self):
        for columns in self.categories:
            self.numData[columns] = LabelEncoder().fit_transform(self.numData[columns])

    def createWidgets(self):
        for i, column in enumerate(self.finalData.columns[:4]):
            label = tk.Label(self.master, text = column + ': ')
            label.grid(row = i, column = 0)
            current_val_label = tk.Label(self.master, text = '0.0')
            current_val_label.grid(row = i, column = 2)
            slider = ttk.Scale(self.master, from_ = self.finalData[column].min(), to = self.finalData[column].max(), orient = "horizontal",
                               command = lambda val, label = current_val_label: label.config(text = f'{float(val):.2f}'))
            slider.grid(row = i, column = 1)
            self.sliders.append((slider, current_val_label))

        predict_button = tk.Button(self.master, text = 'Predict Price', command = self.predictPrice)
        predict_button.grid(row = len(self.finalData.columns[:4]), columnspan = 3)

    def predictPrice(self):
        inputs = [float(slider.get()) for slider, _ in self.sliders]
        price = np.exp(self.model.predict([inputs]))
        messagebox.showinfo('Predicted Price', f'The predicted diamond\'s price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    app = diamondPricePrediction(root)
    root.mainloop()
    
