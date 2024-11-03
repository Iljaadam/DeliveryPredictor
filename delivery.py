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
The **raw data** represents the initial dataset collected from various sources before any processing.
Displaying the raw data helps us to:
- Understand the chaotic structure and contents of the dataset.
- Identify any obvious data quality issues, such as missing values or incorrect data types.
- Gain initial insights into the data to figure out how to preprocess it.
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
After preprocessing, the data is transformed into a suitable format for modeling.
Key steps:
- Handling missing values to ensure data completeness.
- Encoding categorical variables to numerical values for model compatibility.
- Calculating additional features (e.g., **Preparation_Time**, **Distance_km**) that may influence delivery time.
- Dropping irrelevant columns to reduce noise and improve model performance.
""")
st.write(processed_data.head())

# Define features and target variable
X = processed_data.drop('Time_taken (min)', axis=1)
y = processed_data['Time_taken (min)']

st.header('Data Splitting')
st.write("""
Splitting the dataset into training and testing sets allows us to evaluate the model's performance on unseen data.
This helps in assessing the model's generalization ability and prevents overfitting.
""")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

st.write(f"**Training set size:** {X_train.shape[0]} samples")
st.write(f"**Testing set size:** {X_test.shape[0]} samples")

st.header('Model Training and Cross-Validation')
st.write("""
We use a **Random Forest Regressor** model due to its robustness and ability to handle non-linear relationships.
Cross-validation is employed to assess the model's performance across different subsets of the training data.
""")
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
After training, we evaluate the model's performance on the test set to see how well it generalizes to new data.
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
- **Mean Absolute Error (MAE):** The average absolute difference between the predicted and actual delivery times.
  - Lower MAE indicates better model performance.
- **Root Mean Squared Error (RMSE):** The square root of the average squared differences between predicted and actual values.
  - Penalizes larger errors more than MAE.
- **R² Score:** Represents the proportion of variance in the dependent variable that is predictable from the independent variables.
  - Values closer to 1 indicate a better fit.
""")

st.header('Residual Analysis')
st.write("""
Analyzing the residuals (differences between actual and predicted values) helps us understand the model's errors.
Ideally, residuals should be randomly distributed around zero, indicating no patterns left unexplained by the model.
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
Understanding which features have the most influence on the delivery time can provide valuable insights.
This can help in focusing efforts on the most impactful areas.
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
- This information can guide operational improvements.
""")

st.header('Delivery Time Distribution')
st.write("""
Visualizing the distribution of delivery times helps us understand the overall performance and identify any outliers.
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
Comparing actual and predicted delivery times on the test set allows us to see how well the model performs on unseen data.
""")
# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual Delivery Time': y_test,
    'Predicted Delivery Time': y_pred_test
}).reset_index(drop=True)

st.write(results_df.head())

st.write("""
- The table shows individual predictions compared to actual delivery times.
- Smaller differences indicate better predictions.
""")

st.header('Solutions for Improving Delivery Time')
st.write("""
Based on the analysis, the following solutions can help reduce delivery times:
- **Optimize Delivery Routes:**
  - Use of advanced routing algorithms to find the shortest and fastest paths.
- **Real-Time Traffic Monitoring:**
  - Integrate traffic data to avoid congested areas, especially during peak hours.
- **Vehicle Maintenance Programs:**
  - Regularly service delivery vehicles to prevent delays due to breakdowns.
- **Weather Preparedness:**
  - Provide equipment or training to handle adverse weather conditions effectively.
- **Streamline Order Preparation:**
  - Improve coordination between order placement and preparation to reduce waiting times.
""")

st.header('Conclusion')
st.write(f"""
The predictive model demonstrates a good ability to estimate delivery times, with a Mean Absolute Error of **{mae:.2f} minutes**.
Also the Residual Analysis shows seemingly normal distribution around 0, which means that some of the relationships still need to be found.
This not only enhances customer satisfaction but can also lead to cost savings and better resource management.
""")
