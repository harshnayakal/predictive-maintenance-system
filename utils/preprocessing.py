import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath):
    df = pd.read_csv(filepath)
    features = df[['Temperature', 'Vibration', 'Pressure', 'RPM']]
    target = df['Machine_Status']  # 0 = Healthy, 1 = Failure
    return train_test_split(features, target, test_size=0.2, random_state=42)
