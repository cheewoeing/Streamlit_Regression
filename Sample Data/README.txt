Hi. Welcome to my app.

This app is designed such that it can process any CSV files. To help you better understand how this app works, you may
use the sample data provided and follow along with the instructions below.

You can access this app via the link below.
https://streamlit-classification-ncw.herokuapp.com/

Please note that this app is deployed to Heroku platform and computation power allocated can be low.  As such, I will
recommend you to run this app on your localhost and install required python packages as indicated in requirement.txt.
You can access the source code via this Github repo
https://github.com/cheewoeing/Streamlit_Classification


Our objective is to predict the species of the penguins.

1.1 Upload your CSV file
Please upload the insurance.csv file that you have just downloaded.
You can get the sample data via the link below.
https://github.com/cheewoeing/Streamlit_Classification/blob/master/Sample_Data/penguins.csv

1.2 Deal with missing data
I have delibetely deleted some fields on the first 4 rows of data. Please check "Drop rows with missing data?" and on the
drop down menu please select "sex". You will notice that first 2 rows has been deleted as their "sex" field is empty.
Also, please check "fill missing data with mean?" and on the drop down menu please select "bill_length_mm". You will
notice that row 2 and 3 has been filled with mean value of the column.

Click "Proceed to next step"

2.1 Select features(X)
Please select "island", "bill_length_mm", "bill_depth_mm","flipper_length_mm", "body_mass_g" and "sex".

2.2 Select label(y)
Please select "species".

2.3 One-hot encode features
Please check "One-hot encode features?" and on the drop down menu please select "island" and "sex".

Click "Proceed to next step".

3.1 Encode label
Please check "Encode label?"

Click "Proceed to next step".

4.1 Splitting data into train and test set
Please select the size of the test set as you like. Here I would recommend 0.2.

4.2 Scaling features
Please select "bill_length_mm", "bill_depth_mm", "flipper_length_mm", and "body_mass_g".

You can also preview and download X_train, y_train, X_test, y_test in .csv format here

Click "Proceed to next step".

5.1 Build and evaluate model
After the model is built, you can view the confusion matrix and tree visualisation.

Once you are satisfied with the result, click "Proceed to next step".

6.1 Make prediction with user input
Now our model is ready for making prediction, input the feature values and the prediction will be generated below.





