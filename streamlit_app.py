import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from tensorflow.keras.models import load_model
import warnings
import os
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING logs
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Set page configuration
st.set_page_config(
    page_title="4-Tank LSTM Forecasting",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🏭 4-Tank System LSTM Forecasting</h1>', unsafe_allow_html=True)
st.markdown("""
### Neural Network Demonstration for Chemical Process Control
This application demonstrates how LSTM neural networks can forecast tank levels in a coupled 4-tank system - 
a classic problem in chemical engineering process control.
""")

# Sidebar
st.sidebar.header("🎛️ Control Panel")
st.sidebar.markdown("---")

# Load model and components
@st.cache_resource
def load_trained_model():
    """Load the pre-trained LSTM model and associated components"""
    try:
        # Load the Keras model with custom objects to handle compatibility
        import tensorflow as tf
        
        # Define custom objects for backward compatibility
        custom_objects = {
            'time_major': False  # Handle deprecated parameter
        }
        
        # Try loading with compile=False to avoid optimizer issues
        model = load_model('lstm_4tank_model.h5', compile=False)
        
        # Recompile the model with current TensorFlow version
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Load the scalers
        x_scaler = joblib.load('x_scaler.pkl')
        y_scaler = joblib.load('y_scaler.pkl')
        
        # Load metadata
        metadata = joblib.load('model_metadata.pkl')
        
        return model, x_scaler, y_scaler, metadata, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Try alternative loading method
        try:
            st.info("Trying alternative model loading...")
            
            # Load just the weights and rebuild architecture
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            # Load metadata to get architecture info
            metadata = joblib.load('model_metadata.pkl')
            x_scaler = joblib.load('x_scaler.pkl')
            y_scaler = joblib.load('y_scaler.pkl')
            
            # Rebuild model architecture
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(metadata['window_size'], 2)),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(4, activation='linear')
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Try to load weights
            model.load_weights('lstm_4tank_model.h5')
            
            st.success("✅ Model loaded using alternative method!")
            return model, x_scaler, y_scaler, metadata, True
            
        except Exception as e2:
            st.error(f"Alternative loading also failed: {str(e2)}")
            return None, None, None, None, False

# Load training data
@st.cache_data
def load_training_data():
    """Load the training data for visualization"""
    try:
        # Load the CSV data
        df = pd.read_csv('inputs_2.csv')
        return df, True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, False

# Load everything
model, x_scaler, y_scaler, metadata, model_loaded = load_trained_model()
training_data, data_loaded = load_training_data()

if not model_loaded or not data_loaded:
    st.error("❌ Failed to load model or data files. Please check if all files are present.")
    st.stop()

# Success message
st.success("✅ Model and data loaded successfully!")

# Display model information
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Window Size", metadata['window_size'])
with col2:
    st.metric("Input Variables", len(metadata['input_cols']))
with col3:
    st.metric("Output Variables", len(metadata['output_cols']))
with col4:
    st.metric("Model Type", "LSTM")

st.markdown("---")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Make Predictions", "📈 Model Performance", "📚 Educational Info"])

with tab1:
    st.header("📊 Training Data Overview")
    
    if training_data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create interactive plots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Tank Levels Over Time', 'Input Voltages Over Time'),
                vertical_spacing=0.1,
                shared_xaxes=True
            )
            
            # Tank levels
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            tank_names = ['Tank1', 'Tank2', 'Tank3', 'Tank4']
            
            for i, (tank, color) in enumerate(zip(tank_names, colors)):
                if tank in training_data.columns:
                    fig.add_trace(
                        go.Scatter(x=training_data['Time'], y=training_data[tank], 
                                 name=tank, line=dict(color=color)),
                        row=1, col=1
                    )
            
            # Input voltages
            fig.add_trace(
                go.Scatter(x=training_data['Time'], y=training_data['v1'], 
                         name='v1', line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=training_data['Time'], y=training_data['v2'], 
                         name='v2', line=dict(color='purple')),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="Height (m)", row=1, col=1)
            fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Data Statistics")
            st.dataframe(training_data.describe(), use_container_width=True)
            
            st.subheader("🔍 Data Info")
            st.write(f"**Total Data Points:** {len(training_data):,}")
            st.write(f"**Time Range:** {training_data['Time'].min():.0f} - {training_data['Time'].max():.0f} seconds")
            st.write(f"**Duration:** {(training_data['Time'].max() - training_data['Time'].min())/3600:.1f} hours")

with tab2:
    st.header("🤖 LSTM Predictions")
    
    st.markdown("""
    <div class="prediction-box">
    <h4>🎯 How it works:</h4>
    <p>The LSTM model uses the last <strong>10 time steps</strong> of input voltages (v1, v2) to predict the current tank levels.
    Adjust the input sequence below and see the real-time predictions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🎛️ Input Voltage Sequence (Last 10 Time Steps)")
    
    # Create input sequence interface
    col1, col2 = st.columns(2)
    
    # Default values for demonstration
    default_v1 = [5.0, 5.1, 5.2, 5.0, 4.9, 5.0, 5.1, 5.0, 5.0, 5.0]
    default_v2 = [3.0, 3.1, 3.0, 2.9, 3.0, 3.1, 3.0, 3.0, 3.0, 3.0]
    
    with col1:
        st.write("**v1 Voltage Sequence (V):**")
        v1_sequence = []
        for i in range(10):
            v1_val = st.number_input(f"Step {i+1}", 
                                   min_value=0.0, max_value=10.0, 
                                   value=default_v1[i], step=0.1,
                                   key=f"v1_{i}")
            v1_sequence.append(v1_val)
    
    with col2:
        st.write("**v2 Voltage Sequence (V):**")
        v2_sequence = []
        for i in range(10):
            v2_val = st.number_input(f"Step {i+1}", 
                                   min_value=0.0, max_value=10.0, 
                                   value=default_v2[i], step=0.1,
                                   key=f"v2_{i}")
            v2_sequence.append(v2_val)
    
    # Quick preset buttons
    st.subheader("🚀 Quick Presets")
    col1, col2, col3, col4 = st.columns(4)
    
    if col1.button("🔄 Reset to Default"):
        st.rerun()
    
    if col2.button("📈 High Inputs"):
        for i in range(10):
            st.session_state[f"v1_{i}"] = 8.0
            st.session_state[f"v2_{i}"] = 7.0
        st.rerun()
    
    if col3.button("📉 Low Inputs"):
        for i in range(10):
            st.session_state[f"v1_{i}"] = 2.0
            st.session_state[f"v2_{i}"] = 1.5
        st.rerun()
    
    if col4.button("🌊 Oscillating"):
        for i in range(10):
            st.session_state[f"v1_{i}"] = 5.0 + 2*np.sin(i*0.5)
            st.session_state[f"v2_{i}"] = 3.0 + 1.5*np.cos(i*0.3)
        st.rerun()
    
    # Make prediction
    if st.button("🎯 Predict Tank Levels", type="primary"):
        try:
            # Prepare input sequence
            input_sequence = np.array([[v1, v2] for v1, v2 in zip(v1_sequence, v2_sequence)])
            
            # Scale the input
            input_scaled = x_scaler.transform(input_sequence)
            
            # Reshape for LSTM
            input_reshaped = input_scaled.reshape(1, metadata['window_size'], len(metadata['input_cols']))
            
            # Make prediction
            prediction_scaled = model.predict(input_reshaped, verbose=0)
            
            # Inverse transform
            prediction = y_scaler.inverse_transform(prediction_scaled)[0]
            
            # Display results
            st.subheader("🎯 Predicted Tank Levels")
            
            col1, col2, col3, col4 = st.columns(4)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (tank, color) in enumerate(zip(metadata['output_cols'], colors)):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid {color};">
                        <h3 style="color: {color}; margin: 0;">{tank}</h3>
                        <h2 style="margin: 0.5rem 0;">{prediction[i]:.4f} m</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualize input sequence and prediction
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Input Voltage Sequence', 'Predicted Tank Levels'),
                vertical_spacing=0.15
            )
            
            # Input sequence
            time_steps = list(range(1, 11))
            fig.add_trace(
                go.Scatter(x=time_steps, y=v1_sequence, name='v1', 
                         line=dict(color='orange', width=3), marker=dict(size=8)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_steps, y=v2_sequence, name='v2', 
                         line=dict(color='purple', width=3), marker=dict(size=8)),
                row=1, col=1
            )
            
            # Predicted levels (as bar chart)
            fig.add_trace(
                go.Bar(x=metadata['output_cols'], y=prediction, 
                      marker_color=colors, name='Predicted Levels'),
                row=2, col=1
            )
            
            fig.update_layout(height=500, showlegend=True)
            fig.update_xaxes(title_text="Time Step", row=1, col=1)
            fig.update_xaxes(title_text="Tank", row=2, col=1)
            fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
            fig.update_yaxes(title_text="Height (m)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab3:
    st.header("📈 Model Performance")
    
    # Load training history if available
    try:
        training_history = joblib.load('training_history.pkl')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Training history plot
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(y=training_history['loss'], name='Training Loss', line=dict(color='blue')))
            if 'val_loss' in training_history:
                fig_hist.add_trace(go.Scatter(y=training_history['val_loss'], name='Validation Loss', line=dict(color='red')))
            
            fig_hist.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Model architecture info
            st.subheader("🏗️ Model Architecture")
            arch = metadata['model_architecture']
            st.write(f"**LSTM Units:** {arch['lstm_units']}")
            st.write(f"**Dropout Rate:** {arch['dropout_rate']}")
            st.write(f"**Dense Units:** {arch['dense_units']}")
            
            st.subheader("🎛️ Training Parameters")
            params = metadata['training_params']
            st.write(f"**Epochs:** {params['epochs']}")
            st.write(f"**Batch Size:** {params['batch_size']}")
            st.write(f"**Validation Split:** {params['validation_split']}")
            
            # Final loss
            final_loss = training_history['loss'][-1]
            st.metric("Final Training Loss", f"{final_loss:.6f}")
            
    except:
        st.warning("Training history not available.")
        
        # Still show model info
        st.subheader("🏗️ Model Architecture")
        st.write(f"**Input Shape:** {metadata['input_shape']}")
        st.write(f"**Output Shape:** {metadata['output_shape']}")
        st.write(f"**Window Size:** {metadata['window_size']} time steps")

with tab4:
    st.header("📚 Educational Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Learning Objectives")
        st.markdown("""
        - **Time Series Forecasting**: Understanding how neural networks can predict future states
        - **Multivariate Systems**: Working with multiple inputs and outputs simultaneously  
        - **Process Control**: Applying ML to chemical engineering control problems
        - **Data Preprocessing**: Importance of scaling and sequence preparation
        - **Model Validation**: Evaluating neural network performance
        """)
        
        st.subheader("🏭 4-Tank System Physics")
        st.markdown("""
        The 4-tank system is a classic **multivariable control problem**:
        
        - **Tanks 1 & 2**: Lower tanks (outputs)
        - **Tanks 3 & 4**: Upper tanks (buffers)  
        - **v1, v2**: Input pump voltages
        - **Coupling**: Upper tanks drain into lower tanks
        - **Nonlinearity**: Square-root relationship in flow rates
        """)
    
    with col2:
        st.subheader("🧠 LSTM Advantages")
        st.markdown("""
        **Why LSTM for this problem?**
        
        - **Memory**: Remembers past input sequences
        - **Nonlinearity**: Captures complex tank interactions
        - **Multivariable**: Handles multiple inputs/outputs naturally
        - **Time Dependencies**: Models dynamic behavior
        """)
        
        st.subheader("🔍 Key Concepts")
        st.markdown("""
        - **Sequence Length**: How far back the model "looks"
        - **Scaling**: Normalizing inputs for better training
        - **Validation**: Using separate data to test performance
        - **Overfitting**: When model memorizes vs. generalizes
        """)
    
    st.markdown("---")
    st.subheader("🚀 Next Steps for Students")
    st.markdown("""
    1. **Experiment** with different input sequences and observe predictions
    2. **Analyze** how past inputs influence current tank levels  
    3. **Consider** what happens with noisy or missing sensor data
    4. **Think** about control strategies: How would you design v1, v2 to reach target levels?
    5. **Explore** other neural network architectures (GRU, Transformer, etc.)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏭 Chemical Engineering Neural Network Demonstration</p>
    <p>Built with Streamlit • TensorFlow • Keras</p>
</div>
""", unsafe_allow_html=True)