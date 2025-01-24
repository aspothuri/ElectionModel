import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

def predict():
    # Load the data
    data = pd.read_csv('election_data.csv')

    data['polling_avg_current'].fillna(data['result_2020'], inplace=True)
    data['polling_avg_2020'].fillna(data['result_2020'], inplace=True)
    data['polling_avg_2016'].fillna(data['result_2016'], inplace=True)

    # features and target variables
    features = [
        'polling_avg_current', 'polling_avg_2020', 'polling_avg_2016',
        'inflation_current', 'unemployment_current', 
        'inflation_2020', 'unemployment_2020', 
        'inflation_2016', 'unemployment_2016'
    ]
    target_2020 = 'result_2020'
    target_2016 = 'result_2016'

    X_train_2016 = data[features].copy()
    X_train_2016['polling_avg_current'] = data['polling_avg_2016']
    X_train_2016['inflation_current'] = data['inflation_2016']
    X_train_2016['unemployment_current'] = data['unemployment_2016']
    y_train_2016 = data[target_2016]

    X_train_2020 = data[features].copy()
    X_train_2020['polling_avg_current'] = data['polling_avg_2020']
    X_train_2020['inflation_current'] = data['inflation_2020']
    X_train_2020['unemployment_current'] = data['unemployment_2020']
    y_train_2020 = data[target_2020]

    X_train = pd.concat([X_train_2016, X_train_2020])
    y_train = pd.concat([y_train_2016, y_train_2020])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(data[features])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    model.fit(X_train_scaled, y_train, epochs=500, batch_size=16, verbose=1, callbacks=[early_stopping])

    model.save('election_model.h5')

    # Predict 
    X_2024 = data[features].copy()
    X_2024_scaled = scaler.transform(X_2024)
    predictions_2024 = model.predict(X_2024_scaled).flatten()

    return pd.DataFrame({'state': data['state'], 'prediction_2024': predictions_2024})
