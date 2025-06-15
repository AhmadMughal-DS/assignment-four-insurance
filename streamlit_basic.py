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
import joblib
from datetime import datetime
import os

# Set page config with a beautiful theme
st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize MLflow with error handling
mlflow_available = False
try:
    # Get MLflow tracking URI from environment variable or use default
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("Insurance Cost Prediction")
    mlflow_available = True
except Exception as e:
    st.warning("MLflow tracking is not available. Running without experiment tracking.")

# Function to save model individually
def save_model_separately(model, model_name):
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save the model with a specific filename
        model_path = f"models/{model_name}.joblib"
        joblib.dump(model, model_path)
        # st.success(f"💾 Model saved to {model_path}")  # Remove this to avoid duplicate messages
    except Exception as e:
        st.warning(f"Could not save model: {str(e)}")

# Function to load saved model
def load_saved_model(model_name, use_saved=True):
    if not use_saved:
        return None
        
    try:
        # Check if there's a specific saved model file for this model type
        model_paths = {
            "linear_regression": "models/linear_regression.joblib",
            "ridge_regression": "models/ridge_regression.joblib", 
            "random_forest": "models/random_forest.joblib"
        }        # First try to load from models directory
        if model_name in model_paths and os.path.exists(model_paths[model_name]):
            import joblib
            model = joblib.load(model_paths[model_name])
            # st.info(f"✅ Loaded saved {model_name} model from models folder!")  # Remove this duplicate message
            return model
            
        # Fallback: try to load from MLflow artifacts (but this loads same model for all)
        # We'll disable this to force training fresh models
        # model_path = f"mlruns/606562906264104841/b7c3ccd120a3471b8be5e905d112c3fb/artifacts/insurance_model"
        # if os.path.exists(model_path):
        #     model = mlflow.sklearn.load_model(model_path)
        #     st.success(f"Successfully loaded saved {model_name} model!")
        #     return model
            
    except Exception as e:
        st.warning(f"Could not load saved model: {str(e)}")
    return None

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

df = load_data()

# Sidebar for parameters with better styling
with st.sidebar:
    st.header("⚙️ Model Parameters")
    
    # Option to use saved models or train fresh
    use_saved_models = st.checkbox("Use Saved Models", value=True, 
                                  help="Uncheck to train fresh models instead of using saved ones")
    
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 1, 100, 42)

# Create tabs with emojis
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Training", "📊 Data Analysis", "🔍 Feature Insights", "🎯 Predictions"])

# Function to train model with MLflow tracking
def train_model(model_name, X_train, X_test, y_train, y_test, use_saved=True):
    global mlflow_available
    
    # Show what we're trying to do (remove this to avoid duplicate messages)
    # if use_saved:
    #     st.info(f"🔍 Checking for saved {model_name} model...")
    # else:
    #     st.info(f"🆕 Training fresh {model_name} model...")
    
    # First try to load the saved model (only if use_saved is True)
    saved_model = load_saved_model(model_name, use_saved)
    if saved_model is not None:
        # Make predictions with saved model
        y_pred = saved_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': saved_model,
            'metrics': {
                'mse': mse,
                'r2_score': r2
            },
            'was_loaded': True  # Add flag to indicate model was loaded
        }
    
    # If we reach here, we need to train a fresh model
    # st.info(f"🔄 Training fresh {model_name} model...")  # Remove this duplicate message
    
    # If no saved model, train a new one
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
                
                # Save model separately for future use
                save_model_separately(model, model_name)
                
                return {
                    'model': model,
                    'metrics': {
                        'mse': mse,
                        'r2_score': r2
                    },
                    'was_loaded': False  # Fresh training with MLflow
                }
        except Exception as e:
            st.warning(f"MLflow tracking failed: {str(e)}. Continuing without tracking.")
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
      # Save model separately for future use
    save_model_separately(model, model_name)
    
    return {
        'model': model,
        'metrics': {
            'mse': mse,
            'r2_score': r2
        },
        'was_loaded': False  # Fresh training without MLflow
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
        st.session_state.model_results = {}    # Linear Regression Model
    with col1:
        st.subheader("📈 Linear Regression")
        if st.button("Train Linear Regression"):
            with st.spinner("Processing Linear Regression..."):
                result = train_model("linear_regression", X_train, X_test, y_train, y_test, use_saved_models)
                st.session_state.model_results['Linear Regression'] = {
                    'model': result['model'],
                    'mse': result['metrics']['mse'],
                    'r2_score': result['metrics']['r2_score']
                }
                # Show appropriate message based on what happened
                if result.get('was_loaded', False):
                    st.success("✅ Linear Regression model loaded from saved file!")
                else:
                    st.success("🆕 Linear Regression trained successfully!")
                st.metric("MSE", f"{result['metrics']['mse']:.2f}")
                st.metric("R2 Score", f"{result['metrics']['r2_score']:.2f}")
    
    # Ridge Regression Model
    with col2:
        st.subheader("📊 Ridge Regression")
        if st.button("Train Ridge Regression"):
            with st.spinner("Processing Ridge Regression..."):
                result = train_model("ridge_regression", X_train, X_test, y_train, y_test, use_saved_models)
                st.session_state.model_results['Ridge Regression'] = {
                    'model': result['model'],
                    'mse': result['metrics']['mse'],
                    'r2_score': result['metrics']['r2_score']
                }
                # Show appropriate message based on what happened
                if result.get('was_loaded', False):
                    st.success("✅ Ridge Regression model loaded from saved file!")
                else:
                    st.success("🆕 Ridge Regression trained successfully!")
                st.metric("MSE", f"{result['metrics']['mse']:.2f}")
                st.metric("R2 Score", f"{result['metrics']['r2_score']:.2f}")
    
    # Random Forest Model
    with col3:
        st.subheader("🌲 Random Forest")
        if st.button("Train Random Forest"):
            with st.spinner("Processing Random Forest..."):
                result = train_model("random_forest", X_train, X_test, y_train, y_test, use_saved_models)
                st.session_state.model_results['Random Forest'] = {
                    'model': result['model'],
                    'mse': result['metrics']['mse'],
                    'r2_score': result['metrics']['r2_score']
                }
                # Show appropriate message based on what happened
                if result.get('was_loaded', False):
                    st.success("✅ Random Forest model loaded from saved file!")
                else:
                    st.success("🆕 Random Forest trained successfully!")
                st.metric("MSE", f"{result['metrics']['mse']:.2f}")
                st.metric("R2 Score", f"{result['metrics']['r2_score']:.2f}")
                st.success("Random Forest trained successfully!")
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
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
    
    # Train all models if not already trained
    if 'model_results' not in st.session_state or len(st.session_state.model_results) == 0:
        X = df.drop('charges', axis=1)
        y = df['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)        
        # Train Linear Regression
        lr_result = train_model("linear_regression", X_train, X_test, y_train, y_test, use_saved_models)
        st.session_state.model_results['Linear Regression'] = {
            'model': lr_result['model'],
            'mse': lr_result['metrics']['mse'],
            'r2_score': lr_result['metrics']['r2_score']
        }
        
        # Train Ridge Regression
        ridge_result = train_model("ridge_regression", X_train, X_test, y_train, y_test, use_saved_models)
        st.session_state.model_results['Ridge Regression'] = {
            'model': ridge_result['model'],
            'mse': ridge_result['metrics']['mse'],
            'r2_score': ridge_result['metrics']['r2_score']
        }
        
        # Train Random Forest
        rf_result = train_model("random_forest", X_train, X_test, y_train, y_test, use_saved_models)
        st.session_state.model_results['Random Forest'] = {
            'model': rf_result['model'],
            'mse': rf_result['metrics']['mse'],
            'r2_score': rf_result['metrics']['r2_score']
        }
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model for Detailed Analysis",
        list(st.session_state.model_results.keys()),
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
        
        # Get predictions from all models
        predictions = {}
        for model_name, model_data in st.session_state.model_results.items():
            pred = model_data['model'].predict(input_data)[0]
            predictions[model_name] = {
                'prediction': pred,
                'confidence': model_data['r2_score']
            }
        
        # Display selected model's prediction prominently
        st.subheader(f"🎯 Selected Model: {selected_model}")
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