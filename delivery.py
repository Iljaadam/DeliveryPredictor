import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt
from math import radians, cos, sin, asin, sqrt

@st.cache_data
def load_data():
    data = pd.read_csv('delivery_data.csv')
    return data

data = load_data()

# Display the raw data
st.title('Delivery Time Prediction Dashboard')
st.header('Raw Data')
st.write(data.head())

st.header('Data Preprocessing')

# Convert 'Order_Date' to datetime
data['Order_Date'] = pd.to_datetime(data['Order_Date'], format='%d-%m-%Y')

# Handle missing values
st.write('**Handling Missing Values**')
missing_values = data.isnull().sum()
st.write(missing_values[missing_values > 0])

# For simplicity, drop rows with missing target variable
data = data.dropna(subset=['Time_taken (min)'])

# Fill missing values in 'multiple_deliveries' with the mode
data['multiple_deliveries'] = data['multiple_deliveries'].fillna(data['multiple_deliveries'].mode()[0])

# Convert 'multiple_deliveries' to integer
data['multiple_deliveries'] = data['multiple_deliveries'].astype(int)

# Calculate distance using Haversine formula
def haversine_distance(row):
    lon1, lat1, lon2, lat2 = map(radians, [row['Restaurant_longitude'], row['Restaurant_latitude'],
                                           row['Delivery_location_longitude'], row['Delivery_location_latitude']])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

data['distance_km'] = data.apply(haversine_distance, axis=1)

st.header('Exploratory Data Analysis')

# Distribution of Delivery Time
st.subheader('Distribution of Delivery Time')
hist = alt.Chart(data).mark_bar().encode(
    alt.X('Time_taken (min)', bin=alt.Bin(maxbins=30)),
    y='count()',
).properties(
    width=600,
    height=400
)
st.altair_chart(hist, use_container_width=True)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr = data[numeric_cols].corr()
corr = corr.reset_index().melt('index')
corr_chart = alt.Chart(corr).mark_rect().encode(
    x='index:O',
    y='variable:O',
    color='value:Q',
    tooltip=['index', 'variable', 'value']
).properties(
    width=600,
    height=600
)
st.altair_chart(corr_chart, use_container_width=True)

st.header('Feature Encoding')

# Select categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int64', 'float64', 'datetime64']).columns.tolist()

st.write('**Categorical Columns:**', categorical_cols)
st.write('**Numerical Columns:**', numerical_cols)

# Label Encoding for ordinal categories
label_enc_cols = ['Weather_conditions', 'Road_traffic_density', 'Festival', 'City']
for col in label_enc_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# One-Hot Encoding for nominal categories
one_hot_cols = ['Type_of_order', 'Type_of_vehicle']
data = pd.get_dummies(data, columns=one_hot_cols)

st.header('Feature Selection')

# Define features and target
X = data.drop(columns=['ID', 'Delivery_person_ID', 'Time_taken (min)', 'Order_Date', 'Time_Orderd', 'Time_Order_picked',
                       'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude'])
y = data['Time_taken (min)']

st.write('**Selected Features:**')
st.write(X.columns.tolist())

st.header('Model Training')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42, n_estimators=100)

# Train the model
model.fit(X_train, y_train)

st.header('Model Evaluation')

# Predict on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric('MAE', f'{mae:.2f} min')
col2.metric('RMSE', f'{rmse:.2f} min')
col3.metric('R² Score', f'{r2:.2f}')

st.write("""
**Interpretation:**
- **Mean Absolute Error (MAE)**: The average error in minutes between the predicted and actual delivery times.
- **Root Mean Squared Error (RMSE)**: Similar to MAE but penalizes larger errors more.
- **R² Score**: Proportion of variance in the dependent variable that's predictable from the independent variables. An R² score closer to 1 indicates a better fit.
""")

# Feature Importance
st.subheader('Feature Importance')
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.nlargest(10).reset_index()
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
**Interpretation:**
- The top features influencing delivery time help in understanding what factors most affect delivery durations.
- **Distance** is expected to be one of the most significant features.
- Businesses can focus on these areas to optimize delivery efficiency.
""")

st.header('Predict Delivery Time')

st.write('You can input new delivery data to predict the delivery time.')

# Create input widgets for user to input new data
def user_input_features():
    Delivery_person_Age = st.number_input('Delivery Person Age', min_value=18, max_value=65, value=30)
    Delivery_person_Ratings = st.slider('Delivery Person Ratings', 0.0, 5.0, 4.5, 0.1)
    Restaurant_latitude = st.number_input('Restaurant Latitude', value=data['Restaurant_latitude'].mean())
    Restaurant_longitude = st.number_input('Restaurant Longitude', value=data['Restaurant_longitude'].mean())
    Delivery_location_latitude = st.number_input('Delivery Location Latitude', value=data['Delivery_location_latitude'].mean())
    Delivery_location_longitude = st.number_input('Delivery Location Longitude', value=data['Delivery_location_longitude'].mean())
    Weather_conditions = st.selectbox('Weather Conditions', data['Weather_conditions'].unique())
    Road_traffic_density = st.selectbox('Road Traffic Density', data['Road_traffic_density'].unique())
    Vehicle_condition = st.slider('Vehicle Condition', 0, 5, 3)
    multiple_deliveries = st.selectbox('Multiple Deliveries', sorted(data['multiple_deliveries'].unique()))
    Festival = st.selectbox('Festival', data['Festival'].unique())
    City = st.selectbox('City', data['City'].unique())
    Type_of_order = st.selectbox('Type of Order', data['Type_of_order'].unique())
    Type_of_vehicle = st.selectbox('Type of Vehicle', data['Type_of_vehicle'].unique())

    # Calculate distance
    distance_km = haversine_distance({
        'Restaurant_longitude': Restaurant_longitude,
        'Restaurant_latitude': Restaurant_latitude,
        'Delivery_location_longitude': Delivery_location_longitude,
        'Delivery_location_latitude': Delivery_location_latitude
    })

    input_data = {
        'Delivery_person_Age': Delivery_person_Age,
        'Delivery_person_Ratings': Delivery_person_Ratings,
        'Weather_conditions': Weather_conditions,
        'Road_traffic_density': Road_traffic_density,
        'Vehicle_condition': Vehicle_condition,
        'multiple_deliveries': multiple_deliveries,
        'Festival': Festival,
        'City': City,
        'distance_km': distance_km,
        # One-hot encoded features will be added later
        'Type_of_order': Type_of_order,
        'Type_of_vehicle': Type_of_vehicle,
    }
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Preprocess the input data
input_processed = input_df.copy()

# Label Encoding
for col in label_enc_cols:
    le = LabelEncoder()
    le.fit(data[col])
    input_processed[col] = le.transform(input_processed[col])

# One-Hot Encoding
input_processed = pd.get_dummies(input_processed, columns=one_hot_cols)

# Align the input data to match the training data columns
input_processed = input_processed.reindex(columns=X.columns, fill_value=0)

# Predict
if st.button('Predict'):
    prediction = model.predict(input_processed)
    st.success(f'Estimated Delivery Time: {prediction[0]:.2f} minutes')

st.header('Conclusion')
st.write("""
Including the **distance** between the restaurant and the delivery location significantly improves the model's predictive ability. The distance is one of the most important factors affecting delivery time, as expected. This dashboard allows you to explore the factors affecting delivery times and predict delivery durations based on various inputs. By analyzing the feature importance and model performance, businesses can optimize operations to improve delivery efficiency.
""")
