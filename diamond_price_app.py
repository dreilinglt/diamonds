# Diamond Price ML App
# Import libraries
import streamlit as st
import pandas as pd
import pickle

# Set up the app title and image
st.title(':blue[Diamond Prices Predictor] :gem:')
st.write('This app helps you estimate diamond prices based on selected features.')
st.image('diamond_image.jpg', use_column_width = True)

# Reading the pickle file that we created before 
model_pickle = open('diamond_price.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('diamonds.csv')

st.sidebar.image('diamond_sidebar.jpg')
st.sidebar.write('Diamond Price Predictor')
st.sidebar.header('Diamond Features Input')
st.sidebar.write('You can either upload your data file or manually enter the diamond features.')

with st.sidebar.expander("Option 1: Upload CSV file:"):
    user_file = st.file_uploader('Upload a CSV File containig the diamond details.')
    st.header('Sample File Upload:')
    st.write(default_df.head())
    st.write('Ensure your CSV file has the same column names and data types as shown above.')
with st.sidebar.expander('Option 2: Fill out a form:'):
    with st.form('Enter the diamond details using the form below:'):
        cut = st.selectbox('Cut Quality', options=default_df['cut'].unique(), help='Quality of the cut (Fair, Good, Very Good, Premium, Ideal)')
        color = st.selectbox('Diamond Color', options=default_df['color'].unique(), help='Diamond color, from J (worst) to D (best)')
        clarity = st.selectbox('Clarity', options=default_df['clarity'].unique(), help='A measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))')
        carat = st.number_input('Carat Weight', min_value=0.0, max_value=default_df['carat'].max(), step=.01, help='Weight of the diamond')
        depth = st.number_input('Depth (%)', min_value=0.0, max_value=100.0, step=.01, help='Total depth percentage = z / mean(x, y)')
        table = st.number_input('Table (%)', min_value=0.0, max_value=100.0, step=.01, help='Width of top of diamond relative to widest point')
        x = st.number_input('Length', min_value=0.0, max_value=default_df['x'].max(), step=.01, help='Length in mm')
        y = st.number_input('Width', min_value=0.0, max_value=default_df['y'].max(), step=.01, help='Width in mm')
        z = st.number_input('Depth', min_value=0.0, max_value=default_df['z'].max(), step=.01, help='Depth in mm')
        submit_button = st.form_submit_button('Submit Form')


if submit_button:
    st.success('Form data submitted successfully')
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['price'])
    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [carat, cut, color, clarity, depth, table, x, y, z]
    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)
    user_encoded_df = encode_dummy_df.tail(1)
    # Confidence Interval Slider
    alpha = st.slider('Select Alpha Values for prediction intervals:', min_value=.01, max_value=.5, step=.01)
    # Get the prediction with its intervals
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    if intervals[0,0]<0:
        lower_limit=0
    else:
        lower_limit = intervals[0, 0]
    upper_limit = intervals[0, 1]
    # limit 2 decimal places by using ': .2f'
    # Display price predictions
    st.header('Pedicting Prices...')
    st.write('Predicted Price:')
    st.header(f'${prediction[0]: .2f}')
    confidence_interval = (1 - alpha)*100
    st.write(f'**Confidence Interval** ({confidence_interval}%): [{lower_limit: .2f}, {upper_limit[0]: .2f}]')
elif user_file is not None:
    # report success
    st.success('CSV file successfully uploaded')
    # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['price'])
    # Combine the list of user data as rows to default_df
    user_file_df = pd.read_csv(user_file)
    encode_df_combined = pd.concat([encode_df, user_file_df]) 
    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df_combined)
    user_file_length = len(user_file_df)
    user_encoded_df = encode_dummy_df.tail(user_file_length)
    # Confidence Interval Slider
    alpha = st.slider('Select Alpha Values for prediction intervals:', min_value=.01, max_value=.5, step=.01)
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1]
    # Get the prediction with its intervals
    user_file_df['Predicted Price'] = prediction
    user_file_df['Lower Price Limit'] = lower_limit
    user_file_df['Upper Price Limit'] = upper_limit
    # Ensure lowest value is 0 and not negative
    user_file_df['Lower Price Limit'] = user_file_df['Lower Price Limit'].clip(lower=0)
    # Display price predictions
    st.header('Pedicting Prices...')
    st.write('Predicted Prices:')
    st.write(user_file_df)
else:
    st.info('Please choose a data input method to proceed')
    # Confidence Interval Slider
    alpha = st.slider('Select Alpha Values for prediction intervals:', min_value=.01, max_value=.5, step=.01)

# Additional tabs for DT model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('dp_feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('dp_residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('dp_pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('dp_coverage.svg')
    st.caption("Range of predictions with confidence intervals.")