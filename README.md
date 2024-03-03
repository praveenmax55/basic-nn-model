# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY


1. Developing a Neural Network Regression Model AIM To develop a neural network regression model for the given dataset. THEORY Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it.

2. Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.

3. First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.


## Neural Network Model


![neural_model](https://github.com/A-Thiyagarajan/basic-nn-model/assets/118707693/d75d0892-15b8-4ae2-ae28-8f03b8bfb325)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
# Reg.no: 212222240076
# Name: Praveen D
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Data').sheet1

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df=df.astype({'Input':'float'})
df=df.astype({'Output':'float'})
X=df[['Input']].values
Y=df[['Output']].values


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_t_scaled = scaler.transform(x_train)
x_t_scaled

ai_brain = Sequential([
    Dense(3,activation='relu'),
    Dense(4,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_t_scaled,y=y_train,epochs=5)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

scal_x_test=scaler.transform(x_test)
ai_brain.evaluate(scal_x_test,y_test)
input=[[120]]
inp_scale=scaler.transform(input)
inp_scale.shape
ai_brain.predict(inp_scale)
```

## Dataset Information

![Data](https://github.com/A-Thiyagarajan/basic-nn-model/assets/118707693/956839cb-4c08-4f95-9530-d5af6332fc54)



## OUTPUT

### Training Loss Vs Iteration Plot


![graph](https://github.com/A-Thiyagarajan/basic-nn-model/assets/118707693/d9f78e78-e5b4-4b3e-aba1-a5022ac105cd)


### Test Data Root Mean Squared Error



![Epochs](https://github.com/A-Thiyagarajan/basic-nn-model/assets/118707693/b7096f86-3bc7-4039-a317-d61071ecbb2e)


### New Sample Data Prediction


![sample](https://github.com/A-Thiyagarajan/basic-nn-model/assets/118707693/f7d5ad8d-5c6b-40f2-9544-a0cb466ff24f)



## RESULT

Thus a Neural network for Regression model is Implemented.
