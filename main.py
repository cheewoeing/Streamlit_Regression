import pickle
import graphviz
import pandas as pd
import streamlit as st
from dtreeviz.trees import *
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
import re
from catboost import CatBoostRegressor

# The following lines are meant to allow us to add multiple buttons on the same page
################################################################
button_list = ['proceed1', 'proceed2', 'proceed3', 'proceed4', 'proceed5']
for button in button_list:
    if button not in st.session_state:
        st.session_state[button] = False


def proceed1_button():
    st.session_state.proceed1 = True


def proceed2_button():
    st.session_state.proceed2 = True


def proceed3_button():
    st.session_state.proceed3 = True


def proceed4_button():
    st.session_state.proceed4 = True


def proceed5_button():
    st.session_state.proceed5 = True


def cat_boost_model_building(X, y):
    with st.spinner('Model building in progress'):
        try:
            model = CatBoostRegressor()
            model.fit(X, y)
        except:
            st.error("Please make sure all the string variables are encoded.")
    return model


################################################################

st.set_page_config(page_title='Machine Learning with Streamlit', layout='wide')
st.title("Machine Learning with CatBoost Regressor Made Easy")

st.subheader("This tool helps you preprocess your data and build your model efficiently with just few simple clicks.")
st.markdown("Follow these few steps and you will have your machine learning model in no time.")
st.markdown("""
Step 1: Upload your csv file and deal missing data if  there is any. \n 
Step 2: Define your features and label. At the same time it's a good idea to encode your categorical features.\n
Step 3: Encode you label if its has not been done on your csv file.\n
Step 4: Now let's choose the size of the test set. Also, you can scale your features here. Machine learning model 
that are trained on scaled features are generally deliver better result.\n
Step 5: It's time to build your model!!! Here you get to visualize and evaluate your model. Moreover, you can download 
your model in pkl format ans its tree visualization.\n
Step 6: Ready to make some predictions? Input the feature values and generate prediction as output. Isn't it great?\n
\n""")
st.markdown("""---""")
st.markdown("""To help you better understand how this app works, I have prepared a sample dataset
<a href="https://github.com/cheewoeing/Streamlit_Regression/blob/master/Sample%20Data/insurance.csv">insurance.csv</a>
. You can use this dataset and follow along with the step-by-step instructions given in this
<a href="https://github.com/cheewoeing/Streamlit_Regression/blob/master/Sample%20Data/README.txt">README.txt</a>.""",
            unsafe_allow_html=True)
st.markdown("""---""")
st.markdown("""
This app is made with  <a href="https://streamlit.io">Streamlit API</a>. \n
I have taken some reference from this Udemy course 
<a href="https://www.udemy.com/course/machinelearning/">Machine Learning A-Z</a>. This course gave a very good 
explanation on various machine learning algorithms.\n
Also, I strongly recommend this 
<a href="https://www.youtube.com/watch?v=JwSS70SZdyM&t=9068s">video</a> on YouTube by 
<a href="https://www.youtube.com/dataprofessor">Data Professor</a>. Ths video helps me get a very good understanding of 
various types of Streamlit implementations. \n
""", unsafe_allow_html=True)

# initial the layout of the app

e1 = st.container()
c1, c2 = st.columns(2)
e2 = st.container()
c3, c4 = st.columns(2)
e3 = st.container()
c5, c6 = st.columns(2)
e4 = st.container()
c7, c8 = st.columns(2)
e5 = st.container()
c9, c10 = st.columns(2)
e6 = st.container()
c11, c12 = st.columns(2)

e1.markdown("""---""")
e1.header('Section 1')

c1.subheader('1.1 Upload your CSV file')
uploaded_file = c1.file_uploader(label='', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    columns = df.columns
    c1.subheader("1.2 Deal with missing data")
    drop_missing_data = c1.checkbox('Drop rows with missing data?')

    if drop_missing_data:
        drop_missing_data_columns = c1.multiselect('Drop rows that has empty value for following fields',
                                                   columns)

        if len(drop_missing_data_columns) > 0:
            df = df.dropna(subset=drop_missing_data_columns)

    fill_missing_data = c1.checkbox('Fill missing numerical data with mean?')

    if fill_missing_data:
        fill_missing_data_columns = c1.multiselect('Features to fill missing data with mean.',
                                                   columns)

        if len(fill_missing_data_columns) > 0:
            try:
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer.fit(df[fill_missing_data_columns].values)
                df[fill_missing_data_columns] = imputer.transform(df[fill_missing_data_columns].values)

            except:
                c1.error("Please select numerical features only to fill missing data.")

    c2.subheader('Preview your uploaded data')
    c2.dataframe(df)
    proceed1 = (c1.button('Proceed to next step', disabled=(uploaded_file is None), on_click=proceed1_button,
                          key=1) or st.session_state.proceed1)

    if proceed1:

        e2.markdown("""---""")  # Draw a line seperator
        e2.header('Section 2')

        # Select features and label

        c3.subheader("2.1 Select features (X)")
        features = c3.multiselect('Select features(X)', columns)
        if len(features) > 0:
            c3.subheader("2.2 Select label (y)")
            label = c3.radio('Select label(y)', columns[~columns.isin(features)])
            X = df[features]
            y = df[label]
            # Check whether the column values are numeric, if it's not, save the unique values of the column
            isColumnNumeric = {}
            columnUniqueValues = {}
            class_names = y.unique()
            for feature in features:
                isColumnNumeric[feature] = is_numeric_dtype(X[feature])
                if not isColumnNumeric[feature]:
                    columnUniqueValues[feature] = X[feature].unique()

            c3.subheader("2.3 One-hot encode features")
            one_hot_encode = c3.checkbox('One-Hot encode features?')

            if one_hot_encode:
                one_hot_column = c3.multiselect("Features to be one-hot encoded", features)
                ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), one_hot_column)], remainder=
                'passthrough')
                try:
                    X = np.array(ct.fit_transform(X))
                except:
                    c3.error("This column cannot be one-hot encoded.")
                columns_name = ct.get_feature_names()
                X = pd.DataFrame(X, columns=columns_name)

            c4.subheader('Preview your features(X)')
            c4.dataframe(X)

            proceed2 = (c3.button('Proceed to next step', disabled=(features is None), on_click=proceed2_button,
                                  key=2) or st.session_state.proceed2)
            if proceed2:
                e3.markdown("""---""")
                e3.header('Section 3')
                c5.subheader("3.1 Encode label")

                # Select label to be encoded
                label_encode = c5.checkbox('Encode label?')

                if label_encode:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    y = pd.DataFrame(y, columns=[label])
                    c6.subheader('Preview your data')
                    c6.write(pd.concat([X, y], axis=1))
                    c6.write(pd.DataFrame((zip(le.classes_, le.transform(le.classes_))), columns=['Value', 'Code']))

                proceed3 = (c5.button('Proceed to next step', on_click=proceed3_button,
                                      key=3) or st.session_state.proceed3)

                if proceed3:
                    e4.markdown("""---""")
                    e4.header('Section 4')
                    c7.subheader("4.1 Splitting data into train and test set")
                    # Split the data into training and test sets
                    test_size = c7.slider('Select the size of the test set.', 0.0, 1.0, 0.2)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
                    c7.markdown(
                        f"There are {int(len(X_train))} records in the train set and {int(len(X_test))} records in the "
                        f"test set.")

                    # Select features to be scaled
                    c7.subheader("4.2 Scaling features")
                    r = re.compile("^(?!encoder.*$).*")
                    features_scaled_option = list(filter(r.match, X.columns))
                    features_scaled = c7.multiselect("Features to be scaled(Optional)", features_scaled_option)

                    if len(features_scaled) > 0:
                        sc = StandardScaler()
                        X_train[features_scaled] = sc.fit_transform(X_train[features_scaled])
                        X_test[features_scaled] = sc.transform(X_test[features_scaled])

                    X_train_csv = X_train.to_csv().encode('utf-8')
                    X_test_csv = X_test.to_csv().encode('utf-8')
                    y_train_csv = y_train.to_csv().encode('utf-8')
                    y_test_csv = y_test.to_csv().encode('utf-8')

                    data_to_be_preview = ['X_train', 'y_train', 'X_test', 'y_test']
                    c8.subheader("Preview your data")
                    data_selected = c8.selectbox('Select data to be preview', data_to_be_preview)

                    if data_selected == 'X_train':
                        c8.dataframe(X_train)
                        c8.download_button(label='Download X_train.csv', data=X_train_csv, file_name='X_train.csv',
                                           mime='text/csv')

                    if data_selected == 'X_test':
                        c8.dataframe(X_test)
                        c8.download_button(label='Download X_test.csv', data=X_test_csv, file_name='X_test.csv', mime=
                        'text/csv')

                    if data_selected == 'y_train':
                        c8.dataframe(y_train)
                        c8.download_button(label='Download y_train.csv', data=y_train_csv, file_name='y_train.csv',
                                           mime='text/csv')

                    if data_selected == 'y_test':
                        c8.dataframe(y_test)
                        c8.download_button(label='Download y_test.csv', data=y_test_csv, file_name='y_test.csv', mime=
                        'text/csv')

                    proceed4 = (c7.button('Proceed to next step', on_click=proceed4_button,
                                          key=4) or st.session_state.proceed4)

                    if proceed4:
                        e5.markdown("""---""")
                        e5.header('Section 5')
                        e5.subheader('5.1 Build and evaluate model')

                        model = None
                        model = cat_boost_model_building(X_train, y_train)
                        if model:
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = mean_squared_error(y_test, y_pred, squared=False)
                            r_2 = r2_score(y_test,y_pred)
                            c10.subheader(f"Mean squared error: {round(mse, 4)} ")
                            c10.subheader(f"Root mean squared error: {round(rmse, 4)}")
                            c10.subheader(f"R-square: {round(r_2, 4)}")

                            viz_df = pd.DataFrame(list(zip(y_test, y_pred)), columns=['y_test', 'y_pred'])
                            c9.line_chart(viz_df)

                            graph = model.plot_tree(tree_idx=0)
                            graph.format = 'svg'
                            graph.render()
                            with open("Digraph.gv.svg", "rb") as file:
                                btn = c10.download_button(
                                    label="Download tree visualization",
                                    data=file,
                                    file_name="Digraph.gv.svg",
                                    mime="image/svg"
                                )

                            c10.download_button("Download model.pkl", data=pickle.dumps(model), file_name="model.pkl",
                                                disabled=not model)
                            proceed5 = (c9.button('Proceed to next step', on_click=proceed5_button,
                                                  key=5) or st.session_state.proceed5)

                            if proceed5:
                                e6.markdown("""---""")
                                e6.header('Section 6')
                                c11.subheader('6.1 Make Prediction with User input')

                                user_input_dict = {}
                                for feature in features:
                                    if isColumnNumeric[feature]:
                                        user_input_dict[feature] = [c11.number_input(f"{feature}:")]
                                    else:
                                        user_input_dict[feature] = [
                                            c11.selectbox(f"{feature}", columnUniqueValues[feature])]

                                user_input_df = pd.DataFrame.from_dict(user_input_dict)

                                if one_hot_encode:
                                    user_input_df = np.array(ct.transform(user_input_df))
                                    user_input_df = pd.DataFrame(user_input_df, columns=columns_name)

                                if features_scaled:
                                    user_input_df[features_scaled] = sc.transform(
                                        user_input_df[features_scaled])

                                try:
                                    pred = model.predict(user_input_df)
                                    if label_encode:
                                        st.subheader(
                                            f"The prediction for {label} is {(le.inverse_transform(pred))[0]}.")

                                    else:
                                        st.subheader(f"The prediction for {label} is {pred[0]}.")

                                except:
                                    st.error('Couldn\'t make prediction at this moment.')
