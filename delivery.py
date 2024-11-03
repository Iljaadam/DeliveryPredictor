import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import altair as alt


# Function to load data with caching to improve performance
@st.cache_data
def load_data():
    data = pd.read_csv('delivery_data.csv')
    return data

# Load the dataset
data = load_data()

# Display the raw data
st.header('Raw Delivery Data')
st.write("""
This **raw data** represents the initial dataset collected from moodle, which encompasses vast amounts of details about the deliveries.
Displaying the raw data aids us in many things:
- Allows us to grasp the chaotic structure and contents of the dataset.
- To identify any glaring data quality issues, such as missing values or incorrect data types.
- To get insights into the data in order to figure out how to preprocess it.
""")
st.write(data.head())

# Preprocess the data
def preprocess_data(df):
    df = df.copy()

    # Convert date and time columns to datetime
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    df['Time_Orderd'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce')
    df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], format='%H:%M', errors='coerce')

    # Calculate the preparation time in minutes
    df['Preparation_Time'] = (df['Time_Order_picked'] - df['Time_Orderd']).dt.total_seconds() / 60

    # Handle missing values in 'Preparation_Time' by replacing them with the mean value
    df['Preparation_Time'].fillna(df['Preparation_Time'].mean(), inplace=True)

    # Encode categorical variables using Label Encoding
    categorical_cols = [
        'Delivery_person_ID', 'Weather_conditions', 'Road_traffic_density',
        'Type_of_order', 'Type_of_vehicle', 'Festival', 'City'
    ]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Fill missing numerical values with the mean
    numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Calculate the distance between restaurant and delivery location using the Haversine formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + \
            np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    df['Distance_km'] = haversine_distance(
        df['Restaurant_latitude'],
        df['Restaurant_longitude'],
        df['Delivery_location_latitude'],
        df['Delivery_location_longitude']
    )

    # Drop irrelevant or redundant columns
    df.drop(['ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked',
             'Restaurant_latitude', 'Restaurant_longitude',
             'Delivery_location_latitude', 'Delivery_location_longitude'], axis=1, inplace=True)

    return df

# Apply preprocessing to the data
processed_data = preprocess_data(data)

# Display the processed data
st.header('Processed Data')
st.write("""
Right after preprocessing, the data is transformed into an optimal format for the model.
The key steps her are:
- To ensure the completeness of the data by taking care of missing fields and so on.
- To encode the categorical variables into the ones the model could understand, like one-hot-encoding.
- Other variables that need additional calculation must be ready for use.
- To remove any insignificant columns, that may introduce unwanted noise.
""")
st.write(processed_data.head())

# Define features and target variable
X = processed_data.drop('Time_taken (min)', axis=1)
y = processed_data['Time_taken (min)']

st.header('Data Splitting')
st.write("""
In order for us to be able to evaluate the model properly, we need to split the dataset into the train and yet unseen by the model, testing data.
This is crucial, because it minimizes the risk of overfitting and gives us feedback on the applications of the model.""")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write(f"**Training set size:** {X_train.shape[0]} samples")
st.write(f"**Testing set size:** {X_test.shape[0]} samples")

st.header('Model Training and Cross-Validation')
st.write("""
We use common **Random Forest Regressor** model because of its ability to handle non-linear relationships.
And in order to assess the performance of the model, we use cross validated MAE Scores.""")
model = RandomForestRegressor(random_state=42)

# Perform cross-validation on the training set using Mean Absolute Error (MAE) as the scoring metric
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_scores = -cv_scores  # Convert to positive MAE
st.write(f"**Cross-Validation MAE Scores:** {cv_scores}")
st.write(f"**Mean CV MAE:** {cv_scores.mean():.2f} minutes")

# Fit the model on the full training data
model.fit(X_train, y_train)

st.header('Model Evaluation on Test Set')
st.write("""
After training, we test the model's performance on the undseen data to figure out how well it predicts on new data.
""")
# Predict on the test set
y_pred_test = model.predict(X_test)

# Compute evaluation metrics on the test set
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

col1, col2, col3 = st.columns(3)
col1.metric('Mean Absolute Error (MAE)', f'{mae:.2f} minutes')
col2.metric('Root Mean Squared Error (RMSE)', f'{rmse:.2f} minutes')
col3.metric('R² Score', f'{r2:.2f}')

st.write("""
**Evaluation Metrics:**
- We would say that the MAE is reasonably low, and usually the lower it is, the better performance the model shows.
- RMSE also shows a good indication, but it seems that the errors might be on the larger side, due to it going up rather drastically. Which is hardly a good sign. 
- Values of R squared closer to 1 indicate a better fit, which is 0.82 in our case, which is not ideal, but acceptable.
""")

st.header('Residual Analysis')
st.write("""
Residuals analysis allows us to visualize the distribution of errors.
Ideally, residuals should be randomly distributed around zero, meaning that there is no undiscovered pattern. But in our case it's unfortunately rather normally distributed.
Which tends to mean that our model has not accounted for all patterns in the data.
""")
residuals = y_test - y_pred_test
residuals_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test, 'Residuals': residuals})

st.write("**Residuals Histogram:**")
hist_chart = alt.Chart(residuals_df).mark_bar().encode(
    alt.X("Residuals", bin=alt.Bin(maxbins=30)),
    y='count()',
).properties(
    width=600,
    height=400
)
st.altair_chart(hist_chart, use_container_width=True)

st.header('Feature Importance')
st.write("""
Feature list is intresting and significant, because it allows us to see which features affect the prediction the most.
Allowing us to tap into the underlying patterns and reasonings.
""")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feature_importances.nlargest(5).reset_index()
top_features.columns = ['Feature', 'Importance']

chart = alt.Chart(top_features).mark_bar().encode(
    x='Importance',
    y=alt.Y('Feature', sort='-x'),
    color='Feature'
).properties(
    width=600,
    height=400
)
st.altair_chart(chart, use_container_width=True)

st.write("""
- The features with the highest importance scores are the most influential in predicting delivery time.
""")

st.header('Delivery Time Distribution')
st.write("""
Visualizing the distribution of delivery times, allowing us to spot outliers and paint us a broader picture.
""")
delivery_time_df = pd.DataFrame({'Delivery Time (min)': y})
hist_chart = alt.Chart(delivery_time_df).mark_bar().encode(
    alt.X("Delivery Time (min)", bin=alt.Bin(maxbins=30)),
    y='count()',
).properties(
    width=600,
    height=400
)
st.altair_chart(hist_chart, use_container_width=True)

st.header('Predictive Analysis on Test Set')
st.write("""
Comparison of actual and predicted delivery times. 
""")
# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual Delivery Time': y_test,
    'Predicted Delivery Time': y_pred_test
}).reset_index(drop=True)

st.write(results_df.head())

st.write("""
- The table shows individual predictions compared to actual delivery times.
- The less the difference is, the more accurate the model is.
""")

st.header('Solutions for Improving Delivery Time')
st.write("""
Based on the analysis, the following solutions can help reduce delivery times:
- **Potential optimizaztion of the delivery routes:**
  - With the use of advanced routing algorithms it may prove feasible to simply optimize the physical aspect of the delivery.
- **Traffic Monitoring in real time:**
  - It may be possible to avoid high traffic areas with real time monitoring to minimize the time spent there.
- **Weather Conditions:**
  - Some advanved weather predictions may be employed and/or improvement of the equipment to negate the weather disadvantage.
- **Improve Order Logistics:**
  - Order pick ups and preparation times might be bottlenecking the delivery times, which could be solved by minimizing the time driver waits for delivery to be ready for pick up.
""")

st.header('Conclusion')
st.write(f"""
The predictive model demonstrates a good ability to estimate delivery times, with a Mean Absolute Error of **{mae:.2f} minutes**.
Also the Residual Analysis shows seemingly normal distribution around 0, which means that some of the relationships still need to be found.
This research and the model do not only enhance customer satisfaction but can also lead to cost savings and better resource management.
""")
