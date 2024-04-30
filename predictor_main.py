# Description: This script is used to predict the price of a diamond based on its carat, x, y, and z values.
# Authors: Kaedyn Quinn (u3261368) and Tom Hawke (u3239388)
# Tutorial Times: Tom - Wednesday 9:30 am, Kaedyn - Thursday 10:30 am

import pickle
import numpy as np
import pandas as pd
import tkinter as tk
import seaborn as sns
import warnings as warn
from tkinter import ttk
import matplotlib as mpl
from sklearn import metrics
from tkinter import messagebox
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Supressing the warning messages
warn.filterwarnings('ignore')

# Reading the dataset.
diamondData = pd.read_csv('diamonds.csv')

# Printing sample data.
print("### Viewing the data.")
print(diamondData.head())

graph = sns.histplot(data = diamondData, x = "price")
graph.set_title("price distribution")
graph.set_ylabel("count")
plt.show()

# The first column is just an index, which is unneeded.
diamondData = diamondData.drop(["Unnamed: 0"], axis=1)
print("### Viewing the data after removing the index column.")
print(diamondData.head())

# Describing the data to identify problems with the dataset.
print("### Describing the data to identify problems with the dataset.")
print(diamondData.describe())

# Dropping dimensionless (impossible) diamonds.
diamondData = diamondData.drop(diamondData[diamondData["x"]==0].index)
diamondData = diamondData.drop(diamondData[diamondData["y"]==0].index)
diamondData = diamondData.drop(diamondData[diamondData["z"]==0].index)
diamondData.describe()

# Printing the cleaned data.
print("### Printing the cleaned data.")
print('Shape:', diamondData.shape)

# Creating a pair plot for the dataset, using Cut as the hue. Additionally, we can explore other hue attributes with fewer instances.
# graph = sns.pairplot(diamondData, hue = "cut", diag_kind = 'hist')
# plt.show()

# Creating a regression plot for the price versus the y-axis.
graph = sns.regplot(data = diamondData, x = "price", y = "y", fit_reg = True, line_kws = {"color": "#000000"})
graph.set_title("price vs. y-axis")
plt.show()

# Creating a regression plot for the price versus the z-axis.
graph = sns.regplot(data = diamondData, x = "price", y = "z", fit_reg = True, line_kws = {"color": "#000000"})
graph.set_title("price vs. z-axis")
plt.show()

# Creating a regression plot for the price versus the depth.
graph = sns.regplot(data = diamondData, x = "price", y = "depth", fit_reg = True, line_kws = {"color": "#000000"})
graph.set_title("price vs. depth")
plt.show()

# Creating a regression plot for the price versus the table.
graph = sns.regplot(data = diamondData, x = "price", y = "table", fit_reg = True, line_kws = {"color": "#000000"})
graph.set_title("price vs. table")
plt.show()

# Creating a violin plot for the price to look for outliers.
plt.figure(figsize=(12,8))
graph = sns.violinplot(data = diamondData, x = "price", density_norm = "count")
graph.set_title("price")
graph.set_ylabel("count")
graph.set_xlabel("price")
plt.show()

# Creating a violin plot for the cut versus the price.
plt.figure(figsize=(12,8))
graph = sns.violinplot(data = diamondData, x = "cut", y = "price", density_norm = "count")
graph.set_title("cut vs. price")
graph.set_ylabel("price")
graph.set_xlabel("cut")
plt.show()

# Creating a violin plot for the colour versus the price.
plt.figure(figsize=(12,8))
graph = sns.violinplot(data = diamondData, x = "color", y = "price", density_norm = "count")
graph.set_title("color vs. price")
graph.set_ylabel("price")
graph.set_xlabel("color")
plt.show()

# Creating a violin plot for the clarity versus the price.
plt.figure(figsize=(12,8))
graph = sns.violinplot(data = diamondData, x = "clarity", y = "price", density_norm = "count")
graph.set_title("clarity vs. price")
graph.set_ylabel("price")
graph.set_xlabel("clarity")
plt.show()

# Removing the outliers.
diamondData = diamondData[(diamondData["y"] < 30)]
diamondData = diamondData[(diamondData["z"] < 30) & (diamondData["z"] > 2)]
diamondData = diamondData[(diamondData["price"] < 10500)]
diamondData = diamondData[(diamondData["table"] < 80) & (diamondData["table"] > 40)]
diamondData = diamondData[(diamondData["depth"] < 75) & (diamondData["depth"] > 45)]
diamondData.shape
# Creating a pair plot for the dataset, using Cut as the hue.
# graph = sns.pairplot(diamondData, hue = "cut", diag_kind = "hist")
# plt.show()

# Creating a list of the categorical variables.
objects = (diamondData.dtypes =="object")
categories = list(objects[objects].index)

# Creating a new copy of the table to avoid destroying original data.
numericalDiamondData = diamondData.copy()

# Applying the label encoder to each column containing categorical data.
for columns in categories:
    numericalDiamondData[columns] = LabelEncoder().fit_transform(numericalDiamondData[columns])
numericalDiamondData.head()

numericalDiamondData.describe()

# Creating a correlation matrix in order to discern what attributes should and should not be dropped.
correlation = numericalDiamondData.corr()
graph = plt.subplots(figsize=(12,12))
graph = sns.heatmap(correlation, annot = True)
graph.set_title("correlation matrix")
plt.show()

# Saving the final dataset as a pickle file.
columns = ['carat', 'x', 'y', 'z']
finalData = numericalDiamondData[columns]
finalData.to_pickle('finalData.pkl')

# Assigning all non-price attributes as the features, and the price attribute as the target.
predictors = finalData
targetVariable = numericalDiamondData["price"]
x_train, x_test, y_train, y_test = train_test_split(predictors, targetVariable, test_size = 0.3, random_state = 42)

# Building pipelines, using two different scalers and five different regression algorithms to find the best normaliser and regressor.
pipeline_lrss = Pipeline([("scaler1", StandardScaler()), ("lr_classifier", LinearRegression())])
pipeline_lrmm = Pipeline([("scaler2", MinMaxScaler()), ("lr_classifier", LinearRegression())])
pipeline_dtss = Pipeline([("scaler3", StandardScaler()), ("dt_classifier", DecisionTreeRegressor())])
pipeline_dtmm = Pipeline([("scaler4", MinMaxScaler()), ("dt_classifier", DecisionTreeRegressor())])
pipeline_rfss = Pipeline([("scaler5", StandardScaler()), ("rf_classifier", RandomForestRegressor())])
pipeline_rfmm = Pipeline([("scaler6", MinMaxScaler()), ("rf_classifier", RandomForestRegressor())])
pipeline_knss = Pipeline([("scaler7", StandardScaler()), ("rf_classifier", KNeighborsRegressor())])
pipeline_knmm = Pipeline([("scaler8", MinMaxScaler()), ("rf_classifier", KNeighborsRegressor())])
pipeline_xgbss = Pipeline([("scaler9", StandardScaler()), ("rf_classifier", XGBRegressor())])
pipeline_xgbmm = Pipeline([("scaler10", MinMaxScaler()), ("rf_classifier", XGBRegressor())])

# Listing all the pipelines.
pipelines = [pipeline_lrss, pipeline_lrmm, pipeline_dtss, pipeline_dtmm, pipeline_rfss, pipeline_rfmm, pipeline_knss, pipeline_knmm, pipeline_xgbss, pipeline_xgbmm]

# Creating a dictionary.
pipe_dict = {0: "Linear Regression, Standard Normalisation", 1: "Linear Regression, Minmax Normalisation", 2: "Decision Tree, Standard Normalisation", 3: "Decision Tree, Minmax Normalisation", 4: "Random Forest, Standard Normalisation", 5: "Random Forest, Minmax Normalisation", 6: "K-Nearest Neighbors, Standard Normalisation", 7: "K-Nearest Neighbors, Minmax Normalisation", 8: "XGBoost Regressor, Standard Normalisation", 9: "XGBoost Regressor, Minmax Normalisation"}

# Fitting the pipelines.
for pipe in pipelines:
    pipe.fit(x_train, y_train)

# Cross validating the pipelines using negative root mean squared error as the scoring metric.
cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, x_train, y_train, scoring = "neg_root_mean_squared_error", cv = 10)
    cv_results_rms.append(cv_score)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))

# Running model predictions using the Random Forest regressor with minmax normalisation.
rfmmprediction = pipeline_rfmm.predict(x_test)

# Running model predictions using the XGBoost regressor with standard normalisation.
xgbssprediction = pipeline_xgbss.predict(x_test)

# Printing the model evaluations for Random Forest with minmax normalisation.
print("R2:", metrics.r2_score(y_test, rfmmprediction))
print("Adjusted R2:", 1 - (1 - metrics.r2_score(y_test, rfmmprediction)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1]-1))
print("MAE:", metrics.mean_absolute_error(y_test, rfmmprediction))
print("MSE:", metrics.mean_squared_error(y_test, rfmmprediction))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, rfmmprediction)))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test, rfmmprediction))

# Printing the model evaluations for the XGBoost regressor with standard normalisation.
print("R2:", metrics.r2_score(y_test, xgbssprediction))
print("Adjusted R2:", 1 - (1 - metrics.r2_score(y_test, xgbssprediction)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1]-1))
print("MAE:", metrics.mean_absolute_error(y_test, xgbssprediction))
print("MSE:", metrics.mean_squared_error(y_test, xgbssprediction))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, xgbssprediction)))
print("MAPE:", metrics.mean_absolute_percentage_error(y_test, xgbssprediction))

# Assigning all non-price attributes as the features, and the price attribute as the target.

predictors = finalData
targetVariable = numericalDiamondData["price"]
x_train, x_test, y_train, y_test = train_test_split(predictors, targetVariable, test_size = 0.3, random_state = 42)

# Storing the fit object for later reference
predictorScaler = MinMaxScaler().fit(predictors)

# Generating the standardized values of X
predictors = predictorScaler.transform(predictors)

print(predictors.shape)
print(targetVariable.shape)

# Training the model, using 100 per cent of the available data.
model = RandomForestRegressor(max_depth = 2)
finalModel = model.fit(predictors, targetVariable)

# Saving the model as a serialised file.
with open('finalModel.pkl', 'wb') as fileWriteStream:
    pickle.dump(finalModel, fileWriteStream)
    fileWriteStream.close()

def generatePredictionTest(inputDataTest):
    inputNo = inputDataTest.shape[0]

    # Appending the input data with the final dataset.
    finalDataTest = pd.read_pickle('finalData.pkl')
    inputDataTest = pd.concat([inputDataTest, finalDataTest], ignore_index=True)

    featuresTest = ['carat', 'x', 'y', 'z']

    # Generating the input values for the model.
    predictorsTest = inputDataTest[featuresTest].values[0:inputNo]

    # Normalising the generated input values.
    predictorsTest = predictorScaler.transform(predictorsTest)

    # Loading the model.
    with open('finalModel.pkl', 'rb') as fileReadStream:
        finalModelTest = pickle.load(fileReadStream)
        # Don't forget to close the filestream!
        fileReadStream.close()

    # Generating a prediction.
    predictionTest = finalModelTest.predict(predictorsTest)
    resultTest = pd.DataFrame(predictionTest, columns = ['Prediction'])
    return(resultTest)

# Creating some data to predict the price for.
testData = pd.DataFrame(data=[[0.23, 3.95, 3.98, 2.43],[0.21, 3.89, 3.84, 2.31],[0.71, 5.65, 5.68, 3.57],[1.02, 6.28, 6.23, 4.02]], columns = ['carat', 'x', 'y', 'z'])

# Calling the prediction function.
generatePredictionTest(inputDataTest = testData)

# Creating the function which can take in inputs, and return a prediction.
def generatePrediction(inp_carat, inp_x, inp_y, inp_z):

    # Creating a data frame for the model input.
    testData = pd.DataFrame(data = [[inp_carat, inp_x, inp_y, inp_z]], columns = ['carat', 'x', 'y', 'z'])

    # Calling the function defined above using the input parameters.
    predictions = generatePredictionTest(testData)

    # Returning the predictions.
    return(predictions.to_json())

# Calling the function.
generatePrediction(inp_carat = 0.23, inp_x = 3.95, inp_y = 3.98, inp_z = 2.43)

# Installing the prerequisite micro web framework.

app = Flask(__name__)

@app.route('/prediction_api', methods=["GET"])
def prediction_api():
  try:
    # Getting the paramters from API call
    carat_value = float(request.args.get('carat'))
    x_value = float(request.args.get('x'))
    y_value = float(request.args.get('y'))
    z_value = float(request.args.get('z'))

    # Calling the funtion to get predictions
    prediction_from_api = generatePrediction(inp_carat = carat_value, inp_x = x_value, inp_y = y_value, inp_z = z_value)
    return (prediction_from_api)
  except Exception as error:
    return('Something is not right!:' + str(error))

# Running the Flask app at localhost on port 9000.
print("###### Click the following link: http://127.0.0.1:9000/prediction_api?carat=0.23&x=3.95&y=3.98&z=2.43 ######")
if __name__ =="__main__":
  # Do NOT use the URL that is outputted in this code block. Use the URL above.
  app.run(host = '127.0.0.1', port = 9000)
