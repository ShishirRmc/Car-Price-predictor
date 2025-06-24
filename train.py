import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def train_model():
    # Load the data
    df = pd.read_csv('car.csv')
    
    # Data cleaning and preprocessing (identical to notebook)
    df['car_age'] = 2025 - df['year']  # new feature
    
    # Handle missing values
    nulls = ['engine', 'seats', 'mileage(km/ltr/kg)']
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    nulls2 = ['max_power']
    df.fillna(df[nulls].median(), inplace=True)
    df.fillna(df[nulls2].mean(), inplace=True)
    
    # Label encoding for categorical features
    le_fuel = LabelEncoder()
    le_seller = LabelEncoder()
    le_transmission = LabelEncoder()
    le_owner = LabelEncoder()
    
    df['fuel'] = le_fuel.fit_transform(df['fuel'])
    df['seller_type'] = le_seller.fit_transform(df['seller_type'])
    df['transmission'] = le_transmission.fit_transform(df['transmission'])
    df['owner'] = le_owner.fit_transform(df['owner'])
    
    # Drop unnecessary columns
    droping = ['year', 'name']
    df = df.drop(droping, axis=1)
    
    # Prepare features and target
    x = df.drop(['selling_price'], axis=1)
    y = df['selling_price']
    
    # Scale the target variable
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y_scaled, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(x_train, y_train)
    
    # Evaluate the model
    y_pred = rfr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Save the model and preprocessing objects
    model_data = {
        'model': rfr,
        'scaler': scaler,
        'label_encoders': {
            'fuel': le_fuel,
            'seller_type': le_seller,
            'transmission': le_transmission,
            'owner': le_owner
        },
        'feature_columns': x.columns.tolist()
    }
    
    with open('car_price_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully as 'car_price_model.pkl'")
    print(f"Feature columns: {x.columns.tolist()}")
    
    return model_data

if __name__ == "__main__":
    train_model()