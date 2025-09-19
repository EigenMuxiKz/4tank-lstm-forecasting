# 4-Tank System LSTM Forecasting
   
   Neural network forecasting model for chemical engineering process control.
   
   ## Files
   - `lstm_4tank_model.h5` - Trained LSTM model
   - `x_scaler.pkl` - Input data scaler
   - `y_scaler.pkl` - Output data scaler  
   - `model_metadata.pkl` - Model parameters
   - `inputs_2.csv` - Training data
   
   ## Usage
```python
   from keras.models import load_model
   import joblib
   
   # Load model
   model = load_model('lstm_4tank_model.h5')
   x_scaler = joblib.load('x_scaler.pkl')
   y_scaler = joblib.load('y_scaler.pkl')