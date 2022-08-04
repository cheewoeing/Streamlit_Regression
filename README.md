# Machine Learning with CatBoost Regressor Made Easy

Hi. Welcome to my app.

I have made this app to help you preprocess your dataset and build ML model efficiently with just few simple clicks.

This app is designed such that it can process any CSV files. To help you better understand how this app works, you may
use the sample data provided and follow along with the instructions below.

You can access this app via the link below.
https://streamlit-regression-ncw.herokuapp.com/

Please note that this app is deployed to Heroku platform and computation power allocated can be low.  As such, I will
recommend you to run this app on your localhost and install required python packages as indicated in requirement.txt.
You can access the source code via this Github repo
https://github.com/cheewoeing/Streamlit_Regression

## 1.1 Upload your CSV file
You can get the sample data via the link below.
https://github.com/cheewoeing/Streamlit_Regression/blob/master/Sample%20Data/insurance.csv
Please upload the insurance.csv file that you have just downloaded.
Our objective is to predict the insurance charges.

## 1.2 Deal with missing data
I have delibetely deleted some fields on the first 4 rows of data for demo purpose. 
Please check "Drop rows with missing data?" and on the drop down menu please select "sex". 
You will notice that first 2 rows has been deleted as their "sex" field is empty.
Also, please check "fill missing data with mean?" and on the drop down menu please select "bmi". You will
notice that row 2 and 3 has been filled with mean value of the column.

![Screenshot 2022-08-04 at 5 11 44 PM](https://user-images.githubusercontent.com/104248593/182810133-8376729d-94b7-46e6-ac6b-d7dc7a76e84e.png)


Click "Proceed to next step"

## 2.1 Select features(X)
Please select "age","sex", "bmi", "children", "smoker", "region".

## 2.2 Select label(y)
Please select "charges".

## 2.3 One-hot encode features
Please check "One-hot encode features?" and on the drop down menu please select "sex", "smoker","region".

![Screenshot 2022-08-04 at 5 14 54 PM](https://user-images.githubusercontent.com/104248593/182810788-a3f7aa26-8f7b-4338-bda4-4cc10a089e6e.png)

Click "Proceed to next step".

## 3.1 Encode label
You may skip this step for regression problem.

Click "Proceed to next step".

## 4.1 Splitting data into train and test set
Please select the size of the test set as you like. Here I would recommend 0.2.

## 4.2 Scaling features
Please select "age", "bmi", "children".

You can also preview and download X_train, y_train, X_test, y_test in .csv format here

![Screenshot 2022-08-04 at 5 15 54 PM](https://user-images.githubusercontent.com/104248593/182810948-fc1925d1-c624-4737-b233-fd5ffd428337.png)

Click "Proceed to next step".

## 5.1 Build and evaluate model
After the model is built, you can view the chart that compares y_test and y_pred. Moreoveer, you can download the model
tree visualisation and model file in pkl format here.

![Screenshot 2022-08-04 at 5 16 26 PM](https://user-images.githubusercontent.com/104248593/182811062-68d7022e-a2cc-4c30-bec6-59c69ce36e0f.png)

Once you are satisfied with the result, click "Proceed to next step".

## 6.1 Make prediction with user input
Now our model is ready for making prediction, input the feature values and the prediction will be generated below.

![Screenshot 2022-08-04 at 5 19 45 PM](https://user-images.githubusercontent.com/104248593/182811740-910b668b-75d3-4957-8bd4-e72d9f7080ef.png)

