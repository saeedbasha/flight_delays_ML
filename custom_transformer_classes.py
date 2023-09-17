import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic


class ColumnNameFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.columns = X.columns.str.replace(' ', '_').str.lower().str.replace('-', '_')
        return X

class CalculateDistance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def calculate_distance(row):
            dep_coords = (row['dep_lat'], row['dep_lon'])
            arr_coords = (row['arr_lat'], row['arr_lon'])
            distance = geodesic(dep_coords, arr_coords).kilometers
            return int(round(distance, 0))

        X['flight_distance_in_km'] = X.apply(calculate_distance, axis=1)
        return X

class CustomFeaturesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['sta'] = pd.to_datetime(X['sta'], format='%Y-%m-%d %H.%M.%S')
        X['std'] = pd.to_datetime(X['std'], format='%Y-%m-%d %H:%M:%S')
        X['datop'] = pd.to_datetime(X['datop'], format='%Y-%m-%d')
        X['std_time'] = X['std'].dt.time
        X['sta_time'] = X['sta'].dt.time
        X['std_time'] = X['std_time'].astype(str).str.replace(':', '').astype(int)
        X['sta_time'] = X['sta_time'].astype(str).str.replace(':', '').astype(int)
        
        X['elevation_dif'] = (X['arr_elevation'] - X['dep_elevation'])
        X['flight_time_in_min'] = (X['sta'] - X['std']).dt.total_seconds() / 60
        X['average_flight_speed_km_h'] = (X['flight_distance_in_km'] * 60 / X['flight_time_in_min']).round().astype(int)
        X['international_flight'] = np.where(X['arr_country'] != X['dep_country'], 1, 0)
        X['airline_code'] = X['fltid'].str[:2]
        # Extract year, month, and day components
        X['year'] = X['datop'].dt.year
        X['month'] = X['datop'].dt.month
        X['day'] = X['datop'].dt.day
        X['datop'] = X['datop'].astype(str).str.replace('-', '').astype(int)
        
        # Create the seasons column
        X.loc[(X['month'] < 3) | (X['month'] == 12), 'season'] = 'winter'
        X.loc[(X['month'] >= 3) & (X['month'] < 6), 'season'] = 'spring' 
        X.loc[(X['month'] >= 6) & (X['month'] < 9), 'season'] = 'summer' 
        X.loc[(X['month'] >= 9) & (X['month'] < 12), 'season'] = 'autumn'
        
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X.drop(self.columns_to_drop, axis=1, inplace=True)
        return X

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            label_encoder = LabelEncoder()
            label_encoder.fit(X[col])
            self.label_encoders[col] = label_encoder
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col, label_encoder in self.label_encoders.items():
            X_encoded[col] = label_encoder.transform(X_encoded[col])
        return X_encoded