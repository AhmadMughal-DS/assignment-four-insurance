import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime
import os
import logging
import socket
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with a beautiful theme
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MLflow with error handling - quietly fail if not available
mlflow_available = False
try:
    # Get MLflow tracking URI from environment variable or use default
    default_uri = 'http://localhost:5000'
    
    # Check for EC2 environment - try multiple environment variables
    ec2_ip = os.getenv('EC2_PUBLIC_IP') or os.getenv('EC2_IP') or os.getenv('PUBLIC_IP')
    if ec2_ip:
        default_uri = f"http://{ec2_ip}:5000"
        logger.info(f"Using EC2 public IP: {ec2_ip}")
    
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', default_uri)
    logger.info(f"Setting MLflow tracking URI to: {mlflow_tracking_uri}")
    
    # Set MLflow tracking URI first
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Test connection to MLflow server
    try:
        # Extract host and port safely
        if '://' in mlflow_tracking_uri:
            host_part = mlflow_tracking_uri.split('://')[1]
        else:
            host_part = mlflow_tracking_uri
            
        if ':' in host_part:
            host = host_part.split(':')[0]
            port_str = host_part.split(':')[1].split('/')[0]
            try:
                port = int(port_str)
            except ValueError:
                port = 5000
        else:
            host = host_part
            port = 5000
            
        logger.info(f"Testing connection to MLflow at {host}:{port}")
        
        # Try to ping the server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info("Successfully connected to MLflow server")
        else:
            logger.error(f"Could not connect to MLflow server at {mlflow_tracking_uri}")
            raise ConnectionError("MLflow server connection failed")
    except Exception as e:
        logger.error(f"Failed to connect to MLflow server: {str(e)}")
        raise
    
    # Create mlruns directory
    try:
        os.makedirs('mlruns', exist_ok=True)
        logger.info("Created or confirmed mlruns directory")
    except PermissionError:
        logger.warning("Permission error creating mlruns directory")
    
    # Set experiment
    experiment_name = "Insurance Cost Prediction"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Creating new experiment: {experiment_name}")
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        mlflow_available = True
        logger.info("MLflow initialized successfully")
    except Exception as e:
        logger.error(f"Failed to set up MLflow experiment: {str(e)}")
        raise
        
except Exception as e:
    logger.error(f"Failed to initialize MLflow: {str(e)}")
    # Just log error, don't show to user
    logger.info("Running without MLflow experiment tracking.")
    mlflow_available = False

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with emoji
st.title("🏥 Insurance Cost Prediction")
st.markdown("### Train and compare different models for predicting insurance costs")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    df['sex'] = df['sex'].map({'female': 0, 'male': 1})
    df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
    df['region'] = pd.Categorical(df['region']).codes
    return df

# Functions to load and save models
@st.cache_resource
def load_model(model_name):
    """Load a saved model if it exists with caching for improved performance"""
    model_path = os.path.join('models', f"{model_name}.joblib")
    if os.path.exists(model_path):
        logger.info(f"Loading saved model: {model_path}")
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
    
    return None

def save_model(model, model_name):
    """Save a model to disk"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    model_path = os.path.join('models', f"{model_name}.joblib")
    logger.info(f"Saving model to {model_path}")
    try:
        joblib.dump(model, model_path)
        return True
    except Exception as e:
        logger.error(f"Error saving model {model_path}: {str(e)}")
        return False
            # Cached function to get model metrics for faster predictions
@st.cache_data
def get_model_metrics(model_name, X_test, y_test):
    """Get model metrics with caching to avoid recomputation"""
    model = load_model(model_name)
    if model is None:
        return None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'r2_score': r2
    }

# Fast predict function - efficiently makes prediction with a given model name
@st.cache_data
def fast_predict(model_name, input_data):
    """Make a prediction using a model without loading full model details"""
    model = load_model(model_name)
    if model is None:
        return None
        
    return model.predict(input_data)[0]

df = load_data()

# Sidebar for parameters with better styling
with st.sidebar:
    st.header("⚙️ Model Parameters")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 1, 100, 42)
    use_saved_models = st.checkbox("Use saved models if available", value=True, 
                                  help="If checked, will load saved models instead of training new ones when possible.")

# Create tabs with emojis
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Training", "📊 Data Analysis", "🔍 Feature Insights", "🎯 Predictions"])

# Function to train model with MLflow tracking, using saved models if available
def train_model(model_name, X_train, X_test, y_train, y_test, use_saved=True):
    # First try to load a saved model if requested
    saved_model = None
    if use_saved:
        if model_name == "linear_regression":
            saved_model = load_model("linear_regression")
        elif model_name == "ridge_regression":
            saved_model = load_model("ridge_regression")
        elif model_name == "random_forest":
            saved_model = load_model("random_forest")
    
    # If we successfully loaded a saved model, use it
    if saved_model is not None:
        logger.info(f"Using saved model: {model_name}")
        model = saved_model
        
        # Calculate metrics using saved model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model,
            'metrics': {
                'mse': mse,
                'r2_score': r2
            },
            'loaded_from_saved': True
        }
    
    # Otherwise, train a new model
    logger.info(f"Training new model: {model_name}")
    
    global mlflow_available
    if mlflow_available:
        try:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("model_type", model_name)
                
                # Create and train model
                if model_name == "linear_regression":
                    model = LinearRegression()
                elif model_name == "ridge_regression":
                    model = Ridge()
                else:  # random_forest
                    model = RandomForestRegressor(random_state=random_state)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Log metrics
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2_score", r2)
                
                # Log model
                mlflow.sklearn.log_model(model, model_name)
                
                # Save model locally
                save_model(model, model_name)
                
                return {
                    'model': model,
                    'metrics': {
                        'mse': mse,
                        'r2_score': r2
                    },
                    'loaded_from_saved': False
                }
        except Exception as e:
            # Log error but don't show in UI
            logger.error(f"MLflow tracking failed: {str(e)}")
            logger.info("Continuing without MLflow tracking.")
            mlflow_available = False
    
    # If MLflow is not available or failed, train without tracking
    if model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "ridge_regression":
        model = Ridge()
    else:  # random_forest
        model = RandomForestRegressor(random_state=random_state)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save the model
    save_model(model, model_name)
    
    return {
        'model': model,
        'metrics': {
            'mse': mse,
            'r2_score': r2
        }
    }

with tab1:
    st.header("🤖 Model Training")
    
    # Prepare data for training
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Create three columns for model training buttons
    col1, col2, col3 = st.columns(3)
    
    # Dictionary to store model results
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
      # Linear Regression Model
    with col1:
        st.subheader("📈 Linear Regression")
        button_text = "Load/Train Linear Regression"
        if st.button(button_text):
            with st.spinner("Loading or Training Linear Regression..."):
                result = train_model("linear_regression", X_train, X_test, y_train, y_test, use_saved_models)
                st.session_state.model_results['Linear Regression'] = {
                    'model': result['model'],
                    'mse': result['metrics']['mse'],
                    'r2_score': result['metrics']['r2_score']
                }
                if result.get('loaded_from_saved', False):
                    st.success("Linear Regression loaded from saved model!")
                else:
                    st.success("Linear Regression trained and saved successfully!")
                st.metric("MSE", f"{result['metrics']['mse']:.2f}")
                st.metric("R2 Score", f"{result['metrics']['r2_score']:.2f}")
    
    # Ridge Regression Model
    with col2:
        st.subheader("📊 Ridge Regression")
        button_text = "Load/Train Ridge Regression"
        if st.button(button_text):
            with st.spinner("Loading or Training Ridge Regression..."):
                result = train_model("ridge_regression", X_train, X_test, y_train, y_test, use_saved_models)
                st.session_state.model_results['Ridge Regression'] = {
                    'model': result['model'],
                    'mse': result['metrics']['mse'],
                    'r2_score': result['metrics']['r2_score']
                }
                if result.get('loaded_from_saved', False):
                    st.success("Ridge Regression loaded from saved model!")
                else:
                    st.success("Ridge Regression trained and saved successfully!")
                st.metric("MSE", f"{result['metrics']['mse']:.2f}")
                st.metric("R2 Score", f"{result['metrics']['r2_score']:.2f}")
    
    # Random Forest Model
    with col3:
        st.subheader("🌲 Random Forest")
        button_text = "Load/Train Random Forest"
        if st.button(button_text):
            with st.spinner("Loading or Training Random Forest..."):
                result = train_model("random_forest", X_train, X_test, y_train, y_test, use_saved_models)
                st.session_state.model_results['Random Forest'] = {
                    'model': result['model'],
                    'mse': result['metrics']['mse'],
                    'r2_score': result['metrics']['r2_score']
                }
                if result.get('loaded_from_saved', False):
                    st.success("Random Forest loaded from saved model!")
                else:
                    st.success("Random Forest trained and saved successfully!")
                st.metric("MSE", f"{result['metrics']['mse']:.2f}")
                st.metric("R2 Score", f"{result['metrics']['r2_score']:.2f}")
    
    # Model Comparison
    if len(st.session_state.model_results) > 0:
        st.header("📊 Model Comparison")
        
        # Create comparison metrics
        comparison_data = []
        for model_name, results in st.session_state.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'MSE': results['mse'],
                'R2 Score': results['r2_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R2 Score', ascending=False)
        
        # Plot comparison with better styling
        fig = make_subplots(rows=1, cols=2, subplot_titles=('MSE Comparison', 'R2 Score Comparison'))
        
        fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['MSE'], 
                            name='MSE', marker_color='#4CAF50'), row=1, col=1)
        fig.add_trace(go.Bar(x=comparison_df['Model'], y=comparison_df['R2 Score'], 
                            name='R2', marker_color='#2196F3'), row=1, col=2)
        
        fig.update_layout(
            height=500,
            showlegend=False,
            template='plotly_white',
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics table with better styling
        st.subheader("🏆 Model Rankings")
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        comparison_df = comparison_df[['Rank', 'Model', 'MSE', 'R2 Score']]
        st.dataframe(comparison_df.style.background_gradient(cmap='RdYlGn', subset=['R2 Score']))
        
        # Display best model
        best_model = comparison_df.iloc[0]
        st.success(f"🏆 Best Model: {best_model['Model']} (R2 Score: {best_model['R2 Score']:.4f})")

with tab2:
    st.header("📊 Data Analysis")
    
    # Distribution of charges with better styling
    fig = px.histogram(df, x='charges', nbins=50, 
                      title='Distribution of Insurance Charges',
                      color_discrete_sequence=['#4CAF50'])
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap with better styling
    corr = df.corr()
    fig = px.imshow(corr, 
                    labels=dict(color="Correlation"),
                    title="Feature Correlation Heatmap",
                    color_continuous_scale='RdYlBu')
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Box plots for categorical variables
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x='sex', y='charges', 
                    title='Charges by Sex',
                    color_discrete_sequence=['#2196F3'])
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x='smoker', y='charges', 
                    title='Charges by Smoking Status',
                    color_discrete_sequence=['#FF9800'])
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🔍 Feature Insights")
    
    # Scatter plots with regression lines
    selected_feature = st.selectbox("Select feature to analyze", df.columns.drop('charges'))
    
    # Create scatter plot with better styling
    fig = px.scatter(df, x=selected_feature, y='charges', 
                     title=f'Charges vs {selected_feature}',
                     color_discrete_sequence=['#4CAF50'])
    
    # Add regression line
    x = df[selected_feature].values.reshape(-1, 1)
    y = df['charges'].values
    reg = LinearRegression().fit(x, y)
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)
    
    fig.add_trace(go.Scatter(x=x_line.flatten(), y=y_line,
                            mode='lines',
                            name='Regression Line',
                            line=dict(color='#FF5722', width=3)))
    
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature statistics with better styling
    st.subheader("📈 Feature Statistics")
    stats_df = df.describe()
    st.dataframe(stats_df.style.background_gradient(cmap='RdYlGn'))
    
    # Pair plot for selected features
    st.subheader("🔗 Feature Relationships")
    pair_plot_features = list(df.columns)
    selected_features = st.multiselect("Select features for pair plot", 
                                     pair_plot_features, 
                                     default=['age', 'bmi'])
    if len(selected_features) > 1:
        fig = px.scatter_matrix(df, dimensions=selected_features,
                              color_discrete_sequence=['#4CAF50'])
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("🎯 Make Predictions")
    
    # Input form for prediction with better styling
    st.subheader("📝 Enter Insurance Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])    # Initialize models and metrics dictionaries for predictions
    # Only compute metrics if needed, skip full model loading
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fast loading of model results - only when needed
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    
    # Define the available models
    available_models = {
        'Linear Regression': 'linear_regression',
        'Ridge Regression': 'ridge_regression',
        'Random Forest': 'random_forest'
    }
      # Model selection
    selected_model = st.selectbox(
        "Select Model for Detailed Analysis",
        list(available_models.keys()),
        format_func=lambda x: f"🤖 {x}"
    )
    
    if st.button("Predict Insurance Cost"):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == 'male' else 0],
            'bmi': [bmi],
            'children': [children],
            'smoker': [1 if smoker == 'yes' else 0],
            'region': [['southwest', 'southeast', 'northwest', 'northeast'].index(region)]
        })
        
        # Get predictions from all models - load models only when needed
        with st.spinner("Generating predictions..."):
            predictions = {}
            
            for display_name, model_name in available_models.items():
                # Check if we already have this model in session state
                if display_name in st.session_state.model_results and 'model' in st.session_state.model_results[display_name]:
                    model = st.session_state.model_results[display_name]['model']
                    metrics = {
                        'mse': st.session_state.model_results[display_name]['mse'],
                        'r2_score': st.session_state.model_results[display_name]['r2_score']
                    }
                else:
                    # Load model directly (cached function)
                    model = load_model(model_name)
                    if model is None:
                        st.error(f"Could not load model: {model_name}. Please train it first.")
                        continue
                    
                    # Get metrics (cached function)
                    metrics = get_model_metrics(model_name, X_test, y_test)
                    if metrics is None:
                        metrics = {'mse': 0, 'r2_score': 0}
                  # Make prediction with fast cached function
                pred = fast_predict(model_name, input_data)
                if pred is not None:
                    predictions[display_name] = {
                        'prediction': pred,
                        'confidence': metrics['r2_score']
                    }
        
        # Display selected model's prediction prominently
        st.subheader(f"🎯 Selected Model: {selected_model}")
        if selected_model in predictions:
            selected_pred = predictions[selected_model]
            st.metric("Predicted Insurance Cost", f"${selected_pred['prediction']:.2f}")
            st.info(f"Model Confidence: {selected_pred['confidence']:.2%}")
        
        # Show comparison with other models
        st.subheader("📊 Comparison with Other Models")
        
        # Create comparison table
        comparison_data = []
        for model_name, pred_data in predictions.items():
            comparison_data.append({
                'Model': model_name,
                'Predicted Cost': pred_data['prediction'],
                'Confidence': pred_data['confidence']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Confidence', ascending=False)
        
        # Highlight selected model in the table
        def highlight_selected(s):
            return ['background-color: #4CAF50' if x == selected_model else '' for x in s]
        
        st.dataframe(
            comparison_df.style
            .apply(highlight_selected, subset=['Model'])
            .background_gradient(cmap='RdYlGn', subset=['Predicted Cost', 'Confidence'])
        )
        
        # Plot comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df['Model'],
            y=comparison_df['Predicted Cost'],
            marker_color=['#4CAF50' if x == selected_model else '#2196F3' for x in comparison_df['Model']]
        ))
        fig.update_layout(
            title='Predicted Insurance Costs by Model',
            xaxis_title='Model',
            yaxis_title='Predicted Cost ($)',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# Display data info in sidebar
st.sidebar.header("Dataset Information")
st.sidebar.write("Number of samples:", len(df))
st.sidebar.write("Features:", ", ".join(df.columns))