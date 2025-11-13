import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, roc_curve
import warnings
from datetime import datetime
import re
import base64
from io import BytesIO
# Voice features 
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(
    page_title="Comparative Analysis of ML Algorithms",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #1f77b4;
    }
    .eva-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .equipment-found {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .equipment-not-found {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# SESSION STATE INITIALIZATION
def initialize_session_state():
    defaults = {
        'data_uploaded': False,
        'trained_models': {},
        'model_performance': {},
        'predictions': None,
        'feature_importance': {},
        'analysis_type': None,
        'target_column': None,
        'feature_columns': [],
        'id_column': None,
        'current_df': None,
        'processed_data': None,
        'eva_voice_type': 'female',
        'target_encoder': None,
        'eva_conversation_history': [],
        'user_name': None,
        'last_equipment_search': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# UTILITY FUNCTIONS
@st.cache_data
def detect_column_types(df):
    """Enhanced column type detection"""
    column_info = {}
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            column_info[col] = {'type': 'empty', 'suggested_role': 'exclude', 'unique_values': 0}
            continue
        
        unique_ratio = col_data.nunique() / len(col_data)
        col_lower = col.lower()
        
        # Enhanced ID detection patterns
        id_patterns = [
            'equipment_id', 'equipmentid', 'machine_id', 'machineid', 
            'device_id', 'deviceid', 'unit_id', 'asset_id', 'id',
            'equipment_no', 'machine_no', 'unit_no', 'asset_no',
            'e_', 'm_', 'eq_', 'unit_', 'asset_'
        ]
        
        is_likely_id = any(pattern in col_lower for pattern in id_patterns)
        
        # Special handling for predictive maintenance columns
        if col_lower in ['fault detected', 'fault_detected', 'predictive maintenance trigger', 
                        'predictive_maintenance_trigger', 'operational status', 'operational_status',
                        'fault status', 'fault_status']:
            column_info[col] = {
                'type': 'target_maintenance',
                'suggested_role': 'target',
                'unique_values': col_data.nunique(),
                'sample_values': col_data.unique().tolist()
            }
        elif is_likely_id or (unique_ratio > 0.8 and col_data.nunique() > 5):
            column_info[col] = {
                'type': 'identifier',
                'suggested_role': 'id',
                'unique_values': col_data.nunique(),
                'sample_values': col_data.head(3).tolist()
            }
        elif pd.api.types.is_numeric_dtype(col_data):
            if col_data.nunique() <= 10 and col_data.dtype in ['int64', 'int32']:
                column_info[col] = {
                    'type': 'categorical_numeric',
                    'suggested_role': 'potential_target',
                    'unique_values': col_data.nunique(),
                    'sample_values': col_data.unique()[:3].tolist()
                }
            else:
                column_info[col] = {
                    'type': 'continuous_numeric',
                    'suggested_role': 'feature',
                    'unique_values': col_data.nunique(),
                    'range': f"{col_data.min():.2f} - {col_data.max():.2f}"
                }
        else:
            column_info[col] = {
                'type': 'categorical_text',
                'suggested_role': 'potential_target',
                'unique_values': col_data.nunique(),
                'sample_values': col_data.unique()[:3].tolist()
            }
    
    return column_info

@st.cache_data
def preprocess_data_optimized(df, target_col, feature_cols, id_col=None):
    """Optimized preprocessing with caching"""
    processed_df = df.copy()
    
    # Handle missing values efficiently
    for col in feature_cols:
        if col not in processed_df.columns:
            continue
        if processed_df[col].dtype in ['object', 'string']:
            processed_df[col] = processed_df[col].fillna('missing')
        else:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # Encode categorical variables
    encoders = {}
    for col in feature_cols:
        if col not in processed_df.columns:
            continue
        if processed_df[col].dtype in ['object', 'string'] or processed_df[col].nunique() < 10:
            try:
                le = LabelEncoder()
                processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
                encoders[col] = le
                # Update feature_cols list
                idx = feature_cols.index(col)
                feature_cols[idx] = f'{col}_encoded'
            except:
                pass
    
    # Handle target variable
    target_encoder = None
    if processed_df[target_col].dtype in ['object', 'string']:
        target_encoder = LabelEncoder()
        processed_df[f'{target_col}_encoded'] = target_encoder.fit_transform(processed_df[target_col].astype(str))
        target_col = f'{target_col}_encoded'
    
    return processed_df, feature_cols, target_col, encoders, target_encoder

def determine_analysis_type(target_data):
    """Fast analysis type detection"""
    if target_data.dtype in ['object', 'string']:
        return 'classification'
    elif target_data.nunique() <= 10 and target_data.dtype in ['int64', 'int32']:
        return 'classification'
    else:
        return 'regression'

def train_fast_models(df, target_col, feature_cols, analysis_type):
    """Optimized model training"""
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, 
                                                        stratify=y if analysis_type == 'classification' else None)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if analysis_type == 'classification':
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
            'SVM': SVC(probability=True, random_state=42, kernel='rbf', C=1.0),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        }
    else:
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf', C=1.0),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        }
    
    model_results = {}
    trained_models = {}
    feature_importance = {}
    roc_data = {}
    
    for name, model in models.items():
        try:
            if name in ['Logistic Regression', 'Linear Regression', 'SVM', 'SVR', 'K-Nearest Neighbors']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                if analysis_type == 'classification' and hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if analysis_type == 'classification':
                    y_pred_proba = model.predict_proba(X_test)
            
            if analysis_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                model_results[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                }
                
                if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                    try:
                        if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
                    except:
                        pass
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                model_results[name] = {
                    'R² Score': r2,
                    'MSE': mse,
                    'RMSE': np.sqrt(mse),
                    'MAE': mae
                }
            
            scaler_to_store = scaler if name in ['Logistic Regression', 'Linear Regression', 'SVM', 'SVR', 'K-Nearest Neighbors'] else None
            trained_models[name] = {'model': model, 'scaler': scaler_to_store}
            
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                feature_importance[name] = dict(zip(feature_cols, abs(coef)))
                    
        except Exception as e:
            st.warning(f"Failed to train {name}: {str(e)}")
            continue
    
    return model_results, trained_models, feature_importance, roc_data

def generate_predictions_fast(df, trained_models, feature_columns, analysis_type, id_col=None):
    """Enhanced prediction generation"""
    
    if id_col and id_col in df.columns:
        predictions_df = df[[id_col]].copy()
        predictions_df.rename(columns={id_col: 'Equipment_ID'}, inplace=True)
        predictions_df['Equipment_ID'] = predictions_df['Equipment_ID'].astype(str)
    else:
        predictions_df = pd.DataFrame({
            'Equipment_ID': [f'EQ{str(i+1).zfill(3)}' for i in range(len(df))]
        })
    
    if not trained_models:
        return predictions_df
    
    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        st.error("No valid feature columns found in the dataset")
        return predictions_df
    
    try:
        X = df[available_features]
        
        model_name = 'Random Forest'
        if model_name not in trained_models:
            model_name = list(trained_models.keys())[0]
        
        model_info = trained_models[model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
        if scaler:
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
            if analysis_type == 'classification' and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
        else:
            predictions = model.predict(X)
            if analysis_type == 'classification' and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
        
        if analysis_type == 'classification':
            predictions_df['Predicted_Class'] = predictions
            
            target_col_name = st.session_state.target_column.lower() if st.session_state.target_column else ""
            
            if st.session_state.target_encoder:
                original_labels = st.session_state.target_encoder.classes_
                predicted_labels = st.session_state.target_encoder.inverse_transform(predictions)
                predictions_df['Predicted_Label'] = predicted_labels
                
                health_status = []
                for label in predicted_labels:
                    label_lower = str(label).lower()
                    if any(healthy_word in label_lower for healthy_word in ['no fault', 'operational', 'healthy', 'normal']):
                        health_status.append('Healthy')
                    elif any(fault_word in label_lower for fault_word in ['fault', 'maintenance', 'failure', 'risk']):
                        health_status.append('At Risk')
                    else:
                        health_status.append('At Risk' if label == 1 else 'Healthy')
                
                predictions_df['Health_Status'] = health_status
            else:
                if any(target_word in target_col_name for target_word in 
                       ['fault detected', 'predictive maintenance trigger', 'maintenance', 'fault']):
                    predictions_df['Health_Status'] = ['At Risk' if p == 1 else 'Healthy' for p in predictions]
                else:
                    predictions_df['Health_Status'] = ['Healthy' if p == 0 else 'At Risk' for p in predictions]
            
            if hasattr(model, 'predict_proba') and 'probabilities' in locals():
                predictions_df['Confidence'] = np.max(probabilities, axis=1)
                if probabilities.shape[1] == 2:
                    predictions_df['Risk_Score'] = probabilities[:, 1]
                    predictions_df['Risk_Level'] = pd.cut(predictions_df['Risk_Score'], 
                                                        bins=[0, 0.3, 0.7, 1.0], 
                                                        labels=['Low', 'Medium', 'High'])
                else:
                    predictions_df['Risk_Score'] = 1 - predictions_df['Confidence']
                    predictions_df['Risk_Level'] = pd.cut(predictions_df['Risk_Score'], 
                                                        bins=[0, 0.3, 0.7, 1.0], 
                                                        labels=['Low', 'Medium', 'High'])
        else:
            predictions_df['Predicted_Value'] = predictions
            predictions_df['Health_Status'] = 'Operational'
            
            if len(predictions) > 0:
                percentile_33 = np.percentile(predictions, 33)
                percentile_67 = np.percentile(predictions, 67)
                predictions_df['Risk_Level'] = pd.cut(predictions, 
                                                    bins=[-np.inf, percentile_33, percentile_67, np.inf], 
                                                    labels=['High', 'Medium', 'Low'])
                predictions_df['Risk_Score'] = 1 - (predictions - predictions.min()) / (predictions.max() - predictions.min())
    
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        predictions_df['Health_Status'] = 'Error'
        predictions_df['Risk_Level'] = 'Unknown'
        predictions_df['Risk_Score'] = 0.5
    
    return predictions_df

def search_equipment_status(df, equipment_id, id_column='Equipment_ID', predictions_df=None):
    """Enhanced equipment search with better ID matching"""
    if df is None:
        return None
    
    available_id_columns = []
    
    if id_column and id_column in df.columns:
        available_id_columns.append(id_column)
    
    id_patterns = ['id', 'equipment_id', 'equipmentid', 'machine_id', 'machineid', 
                   'device_id', 'unit_id', 'asset_id', 'equipment_no', 'machine_no',
                   'e_', 'm_', 'eq_', 'unit_']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in id_patterns) and col not in available_id_columns:
            available_id_columns.append(col)
    
    if not available_id_columns:
        available_id_columns = [df.columns[0]]
    
    equipment_data = None
    actual_id_column = None
    
    for search_col in available_id_columns:
        equipment_id_str = str(equipment_id).strip()
        df_search = df.copy()
        df_search[search_col] = df_search[search_col].astype(str).str.strip()
        
        # Try exact match first
        exact_match = df_search[df_search[search_col] == equipment_id_str]
        if not exact_match.empty:
            equipment_data = exact_match
            actual_id_column = search_col
            break
        
        # Try case-insensitive match
        case_insensitive_match = df_search[df_search[search_col].str.lower() == equipment_id_str.lower()]
        if not case_insensitive_match.empty:
            equipment_data = case_insensitive_match
            actual_id_column = search_col
            break
        
        # Try partial match
        partial_match = df_search[df_search[search_col].str.contains(equipment_id_str, case=False, na=False)]
        if not partial_match.empty:
            equipment_data = partial_match
            actual_id_column = search_col
            break
    
    if equipment_data is None or equipment_data.empty:
        return {
            'status': 'not_found',
            'message': f'Equipment ID {equipment_id} not found in any ID column',
            'searched_columns': available_id_columns,
            'available_ids': [str(id_val) for id_val in df[available_id_columns[0]].head(5).tolist()] if available_id_columns else []
        }
    
    equipment_record = equipment_data.iloc[0]
    
    result = {
        'equipment_id': equipment_id,
        'status': 'found',
        'id_column_used': actual_id_column,
        'data': equipment_record.to_dict()
    }
    
    if predictions_df is not None and 'Equipment_ID' in predictions_df.columns:
        pred_df_search = predictions_df.copy()
        pred_df_search['Equipment_ID'] = pred_df_search['Equipment_ID'].astype(str).str.strip()
        
        pred_match = pred_df_search[pred_df_search['Equipment_ID'] == str(equipment_id).strip()]
        
        if pred_match.empty:
            equipment_index = equipment_data.index[0]
            if equipment_index < len(predictions_df):
                pred_record = predictions_df.iloc[equipment_index].to_dict()
                result['predictions'] = pred_record
        else:
            pred_record = pred_match.iloc[0].to_dict()
            result['predictions'] = pred_record
        
        if 'predictions' in result:
            pred_record = result['predictions']
            if 'Risk_Score' in pred_record:
                risk_score = pred_record['Risk_Score']
                health_status = pred_record.get('Health_Status', 'Unknown')
                
                result['health_status'] = {
                    'status': health_status,
                    'risk_score': risk_score,
                    'confidence': pred_record.get('Confidence', 0),
                    'risk_level': pred_record.get('Risk_Level', 'Unknown')
                }
                
                if risk_score > 0.7:
                    result['maintenance_recommendation'] = {
                        'priority': 'HIGH',
                        'action': 'Immediate maintenance required',
                        'risk_level': 'Critical',
                        'days_until_maintenance': 1
                    }
                elif risk_score > 0.4:
                    result['maintenance_recommendation'] = {
                        'priority': 'MEDIUM',
                        'action': 'Schedule maintenance within 7 days',
                        'risk_level': 'Moderate',
                        'days_until_maintenance': 7
                    }
                else:
                    result['maintenance_recommendation'] = {
                        'priority': 'LOW',
                        'action': 'Normal operation - routine maintenance',
                        'risk_level': 'Low',
                        'days_until_maintenance': 30
                    }
    
    return result

# EVA VOICE ASSISTANT FUNCTIONS
def recognize_speech_from_mic():
    if not VOICE_AVAILABLE:
        st.error("Voice recognition not available")
        return None
        
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 EVA is listening...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=3, phrase_time_limit=8)
            
        user_input = r.recognize_google(audio)
        st.success(f"You said: '{user_input}'")
        return user_input.lower()
    except Exception as e:
        st.error("I couldn't understand that. Please try again.")
        return None

def eva_speak(text, voice_gender="female"):
    """EVA's text-to-speech function"""
    if not VOICE_AVAILABLE:
        return
        
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if voice_gender.lower() == "female" and len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        else:
            engine.setProperty('voice', voices[0].id)
            
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def get_current_time_greeting():
    """Get appropriate greeting based on current time"""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 22:
        return "Good evening"
    else:
        return "Good evening"

def process_eva_command(user_text):
    """Enhanced EVA command processing"""
    response = ""
    user_text_lower = user_text.lower().strip()
    
    # Store user name if mentioned
    name_patterns = [
        r'my name is (\w+)',
        r"i'm (\w+)",
        r'call me (\w+)',
        r"i am (\w+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, user_text_lower)
        if match:
            st.session_state.user_name = match.group(1).title()
            response = f"Nice to meet you, {st.session_state.user_name}! I'm EVA, your AI maintenance assistant. I can help you search equipment, analyze data, and monitor system health. How can I help you today?"
            break
    
    # Enhanced Greeting responses
    if not response:
        greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greeting in user_text_lower for greeting in greeting_words):
            time_greeting = get_current_time_greeting()
            if st.session_state.user_name:
                response = f"{time_greeting}, {st.session_state.user_name}! I'm EVA, ready to assist with your equipment monitoring needs. What would you like to check today?"
            else:
                response = f"{time_greeting}! I'm EVA, your AI maintenance assistant. I can search equipment status, analyze data quality, and help with predictive maintenance. What can I do for you?"
        
        elif "how are you" in user_text_lower or "how's it going" in user_text_lower:
            if st.session_state.data_uploaded and st.session_state.trained_models:
                response = "I'm functioning optimally! All systems are running smoothly, models are trained, and I'm monitoring your equipment data. Everything looks good on my end. How are your operations today?"
            elif st.session_state.data_uploaded:
                response = "I'm doing well! Your data is loaded and I'm ready to help. We just need to train the AI models to start equipment monitoring. How can I assist you?"
            else:
                response = "I'm ready and waiting! My systems are fully operational. Upload your equipment data and I'll help you get started with predictive maintenance analysis."
        
        elif any(word in user_text_lower for word in ["thank you", "thanks", "appreciate"]):
            if st.session_state.user_name:
                response = f"You're very welcome, {st.session_state.user_name}! I'm always happy to help with your maintenance needs. Is there anything else I can assist you with?"
            else:
                response = "You're very welcome! I'm always here to help with equipment monitoring and predictive maintenance. What else can I do for you?"
        
        elif "what can you do" in user_text_lower or "help me" in user_text_lower or user_text_lower == "help":
            response = "I'm EVA, your maintenance assistant! I can help you with:\n\n• 🔍 Search specific equipment by ID (try 'search equipment E_1')\n•  Analyze your data quality and structure\n• 🧠 Monitor AI model performance\n• ⚡ Provide real-time equipment health status\n• 🔧 Give maintenance recommendations\n• 💬 Chat about your operations\n\nJust ask me naturally, like 'check equipment E_1' or 'how is my data looking?'"
    
    # Enhanced Equipment search commands
    if not response and any(word in user_text_lower for word in ["search", "find", "equipment", "status", "health", "check", "look up", "locate"]):
        equipment_patterns = [
            r'equipment\s+([a-zA-Z0-9_]+)',
            r'machine\s+([a-zA-Z0-9_]+)', 
            r'id\s+([a-zA-Z0-9_]+)',
            r'number\s+([a-zA-Z0-9_]+)',
            r'unit\s+([a-zA-Z0-9_]+)',
            r'e_(\d+)',
            r'eq_(\d+)',
            r'([a-zA-Z]+_\d+)',
            r'([a-zA-Z0-9_]+)'
        ]
        
        equipment_id = None
        for pattern in equipment_patterns:
            match = re.search(pattern, user_text_lower)
            if match:
                equipment_id = match.group(1)
                break
        
        if not equipment_id:
            words = user_text_lower.split()
            for word in words:
                if re.match(r'^[a-zA-Z0-9_]+$', word) and len(word) > 1:
                    equipment_id = word
                    break
        
        if equipment_id and st.session_state.data_uploaded:
            df = st.session_state.current_df
            st.session_state.last_equipment_search = equipment_id
            
            result = search_equipment_status(df, equipment_id, st.session_state.id_column, st.session_state.predictions)
            
            if result and result['status'] == 'found':
                equipment_id_display = result['equipment_id']
                
                if 'health_status' in result:
                    status = result['health_status']['status']
                    risk_level = result['health_status']['risk_level']
                    risk_score = result['health_status']['risk_score']
                    confidence = result['health_status'].get('confidence', 0)
                    
                    # Convert risk_level to string and handle potential NaN values
                    risk_level_str = str(risk_level).lower() if risk_level and str(risk_level) != 'nan' else 'unknown'
                    
                    if status == 'Healthy':
                        response = f"Great news! Equipment {equipment_id_display} is showing {status} status with {risk_level_str} risk level ({risk_score:.1%} risk score). Confidence level is {confidence:.1%}."
                    else:
                        response = f"Alert! Equipment {equipment_id_display} shows {status} status with {risk_level_str} risk level ({risk_score:.1%} risk score). Confidence: {confidence:.1%}."
                    
                    if 'maintenance_recommendation' in result:
                        rec = result['maintenance_recommendation']
                        if rec['priority'] == 'HIGH':
                            response += f" URGENT: {rec['action']} - This requires immediate attention!"
                        elif rec['priority'] == 'MEDIUM':  
                            response += f" Recommendation: {rec['action']}"
                        else:
                            response += f" Status: {rec['action']}"
                else:
                    response = f"I found equipment {equipment_id_display} in the database. The equipment data is available, but I need the AI models to be trained first for detailed health analysis."
            
            elif result and result['status'] == 'not_found':
                available_ids = result.get('available_ids', [])
                response = f"I couldn't find equipment {equipment_id} in the current dataset. "
                if available_ids:
                    response += f"Available equipment IDs include: {', '.join(available_ids[:3])}... Would you like me to search for one of these instead?"
                else:
                    response += "Please check the equipment ID and try again."
            else:
                response = f"I encountered an issue searching for equipment {equipment_id}. Please try again or check if the data is properly loaded."
        
        elif equipment_id:
            response = "I'd love to help you search for equipment, but I need data to be uploaded first. Please upload your equipment CSV file and I'll be able to search for any equipment ID you need."
        else:
            response = "I can help you search for equipment! Please specify an equipment ID, like 'search equipment E_1' or 'check machine 123'. What equipment would you like me to look up?"
    
    # Data and system status queries
    elif not response and any(word in user_text_lower for word in ["data", "dataset", "info", "information", "status", "system"]):
        if st.session_state.data_uploaded:
            df = st.session_state.current_df
            healthy_count = 0
            at_risk_count = 0
            
            if st.session_state.predictions is not None and 'Health_Status' in st.session_state.predictions.columns:
                healthy_count = len(st.session_state.predictions[st.session_state.predictions['Health_Status'] == 'Healthy'])
                at_risk_count = len(st.session_state.predictions[st.session_state.predictions['Health_Status'] == 'At Risk'])
            
            response = f"Your current dataset contains {len(df)} equipment records with {len(df.columns)} parameters being monitored. "
            if st.session_state.predictions is not None:
                response += f"Current analysis shows {healthy_count} healthy units and {at_risk_count} units requiring attention."
                if at_risk_count > 0:
                    response += f" I recommend prioritizing maintenance for the {at_risk_count} at-risk units."
            else:
                response += "The data looks good and is ready for AI model training and predictive analysis."
        else:
            response = "No equipment data has been loaded yet. Please upload a CSV file with your equipment data, and I'll help you analyze equipment health, predict failures, and optimize maintenance schedules."
    
    # Model performance queries
    elif not response and any(word in user_text_lower for word in ["model", "performance", "accuracy", "training", "ai", "algorithm"]):
        if st.session_state.model_performance:
            if st.session_state.analysis_type == 'classification':
                best_model = max(st.session_state.model_performance.items(), 
                               key=lambda x: x[1]['F1-Score'])
                response = f"All AI models are trained and performing well! The best model is {best_model[0]} with {best_model[1]['F1-Score']:.1%} F1-score. The predictive maintenance system is ready and actively monitoring your equipment."
            else:
                best_model = max(st.session_state.model_performance.items(), 
                               key=lambda x: x[1]['R² Score'])
                response = f"AI models are running smoothly! Top performing model is {best_model[0]} with {best_model[1]['R² Score']:.1%} R² score. The system is ready for accurate equipment predictions."
        else:
            response = "The AI models haven't been trained yet. Once you upload your equipment data and configure the settings, I'll help you train the predictive maintenance models for accurate equipment health monitoring."
    
    # Maintenance and operational queries
    elif not response and any(phrase in user_text_lower for phrase in ["maintenance", "repair", "fix", "broken", "failure", "risk", "priority"]):
        if st.session_state.predictions is not None and 'Health_Status' in st.session_state.predictions.columns:
            at_risk_equipment = st.session_state.predictions[st.session_state.predictions['Health_Status'] == 'At Risk']
            high_risk_equipment = st.session_state.predictions[st.session_state.predictions.get('Risk_Level', '') == 'High']
            
            if len(high_risk_equipment) > 0:
                sample_high_risk = high_risk_equipment.head(3)['Equipment_ID'].tolist()
                response = f"I've identified {len(high_risk_equipment)} high-risk equipment units requiring immediate attention: {', '.join(sample_high_risk)}. "
                response += f"Additionally, {len(at_risk_equipment)} total units are flagged for maintenance. Would you like me to search specific equipment for detailed recommendations?"
            elif len(at_risk_equipment) > 0:
                sample_at_risk = at_risk_equipment.head(3)['Equipment_ID'].tolist()
                response = f"Currently monitoring {len(at_risk_equipment)} equipment units that need maintenance attention: {', '.join(sample_at_risk)}. These units should be scheduled for maintenance soon to prevent failures."
            else:
                response = "All monitored equipment is currently showing healthy status. The predictive maintenance system indicates no immediate maintenance requirements. Continue with regular scheduled maintenance."
        else:
            response = "To provide maintenance recommendations, I need to analyze your equipment data first. Please upload your data and train the AI models, then I can identify which equipment needs attention."
    
    # Fun/personality responses
    elif not response and any(phrase in user_text_lower for phrase in ["tell me a joke", "joke", "funny", "humor"]):
        jokes = [
            "Why don't maintenance engineers ever get tired? Because they always know when to take a break... before the equipment does!",
            "What did the predictive maintenance AI say to the broken machine? 'I saw this coming!'",
            "Why do maintenance teams love AI? Because it tells them what's broken before it knows it's broken!",
        ]
        import random
        joke = random.choice(jokes)
        response = f"{joke} Speaking of maintenance, how are your machines running today?"
    
    elif not response and "who are you" in user_text_lower:
        response = "I'm EVA - Equipment Vitality Analyzer! I'm your AI assistant specialized in predictive maintenance. I help monitor equipment health, predict potential failures, analyze maintenance needs, and optimize your operations."
    
    elif not response and any(word in user_text_lower for word in ["goodbye", "bye", "see you", "later", "exit"]):
        if st.session_state.user_name:
            response = f"Goodbye, {st.session_state.user_name}! I'll keep monitoring your equipment systems. Feel free to ask me anything about equipment status anytime. Have a productive day!"
        else:
            response = "Goodbye! I'll be here whenever you need equipment monitoring assistance or have questions about your maintenance operations. Take care!"
    
    # Default helpful response
    elif not response:
        if st.session_state.data_uploaded and st.session_state.trained_models:
            response = "I'm here to help with equipment monitoring! You can ask me to search specific equipment (like 'check equipment E_1'), get system status updates, or ask about maintenance priorities. What would you like to know?"
        elif st.session_state.data_uploaded:
            response = "I can help you with your loaded data! Try asking me to analyze the data quality, or let's get the AI models trained so I can start monitoring equipment health. What would you like to do next?"
        else:
            response = "I'm ready to help with predictive maintenance! Upload your equipment data CSV and I can search equipment status, predict failures, and provide maintenance recommendations. How can I assist you?"
    
    # Add conversation to history
    st.session_state.eva_conversation_history.append({
        'user': user_text,
        'eva': response,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    })
    
    # Display EVA's response
    st.markdown(f"""
        <div class="eva-response">
            <strong>EVA:</strong> {response}
        </div>
    """, unsafe_allow_html=True)
    
    # Speak the response
    eva_speak(response, st.session_state.eva_voice_type)

# DISPLAY FUNCTIONS
def show_welcome_screen():
    st.markdown("""
        <div style='text-align: center; padding: 30px 0;'>
            <h1 style='color: #1f77b4; font-size: 2.5em;'>Comparative analysis of machine learning algorithm for predictive maintenance of industrial equipments</h1>
            <p style='font-size: 1.1em; color: #666;'>Upload your equipment data CSV to get started</p>
            <p style='font-size: 1.0em; color: #888;'>Meet EVA - Your Equipment Vitality Analyzer Assistant!</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        **Features:**
        - Equipment health status search (supports IDs like E_1, E_2, etc.)
        - Automated failure prediction
        - Maintenance recommendations  
        - EVA voice assistant integration
        - Real-time analytics dashboard
        """)

def show_column_configuration(df):
    """Enhanced column configuration"""
    st.markdown("## Data Configuration")
    
    column_info = detect_column_types(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        maintenance_targets = [col for col, info in column_info.items() 
                             if info['suggested_role'] == 'target']
        potential_targets = [col for col, info in column_info.items() 
                           if info['suggested_role'] == 'potential_target']
        
        all_targets = maintenance_targets + potential_targets
        if not all_targets:
            all_targets = list(df.columns)
        
        if maintenance_targets:
            st.success(f" Recommended targets found: {', '.join(maintenance_targets)}")
        
        target_col = st.selectbox(" Target Column (predict this):", all_targets)
        
        if target_col:
            st.session_state.target_column = target_col
            st.session_state.analysis_type = determine_analysis_type(df[target_col])
            
            target_values = df[target_col].value_counts()
            st.write("**Target Distribution:**")
            for value, count in target_values.head().items():
                st.write(f"• {value}: {count} samples")
            
            st.success(f"Analysis: **{st.session_state.analysis_type.title()}**")
    
    with col2:
        potential_ids = [col for col, info in column_info.items() 
                        if info['suggested_role'] == 'id']
        potential_ids = ['None'] + potential_ids
        
        if potential_ids and len(potential_ids) > 1:
            st.success(f"🆔 Equipment ID columns detected: {', '.join(potential_ids[1:])}")
        
        id_col = st.selectbox("🆔 Equipment ID Column:", potential_ids)
        st.session_state.id_column = id_col if id_col != 'None' else None
        
        if st.session_state.id_column:
            unique_count = df[st.session_state.id_column].nunique()
            sample_ids = df[st.session_state.id_column].head(3).tolist()
            st.info(f"Found {unique_count} unique equipment IDs")
            st.write("**Sample IDs:**")
            for sample_id in sample_ids:
                st.write(f"• {sample_id}")
    
    # Feature columns
    st.markdown("### 📊 Select Features")
    potential_features = [col for col in df.columns 
                         if col != target_col and col != st.session_state.id_column]
    
    default_features = [col for col in potential_features 
                       if column_info.get(col, {}).get('suggested_role') == 'feature'][:8]
    
    selected_features = st.multiselect(
        "Choose feature columns:",
        options=potential_features,
        default=default_features,
        help="Select sensor readings and numerical features"
    )
    
    st.session_state.feature_columns = selected_features
    
    return target_col and selected_features

def show_overview(df):
    st.markdown("## Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Records", len(df))
    with col2:
        st.metric(" Columns", len(df.columns))
    with col3:
        if st.session_state.predictions is not None:
            healthy_count = len(st.session_state.predictions[
                st.session_state.predictions.get('Health_Status', '') == 'Healthy'
            ])
            st.metric("✅ Healthy Equipment", healthy_count)
        else:
            st.metric("✅ Healthy Equipment", "N/A")
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("❌ Missing Values", missing)
    
    # Enhanced Equipment ID Analysis
    st.markdown("###  Equipment ID Analysis")
    
    column_info = detect_column_types(df)
    id_candidates = [col for col, info in column_info.items() if info['suggested_role'] == 'id']
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if id_candidates:
            primary_id = id_candidates[0]
            unique_ids = df[primary_id].nunique()
            st.metric("Equipment IDs Available", unique_ids)
        else:
            st.metric("Equipment IDs Available", "Not Configured")
    
    with col_b:
        if id_candidates:
            primary_id = id_candidates[0]
            sample_ids = df[primary_id].head(5).tolist()
            st.write("**Sample Equipment IDs:**")
            for sample_id in sample_ids:
                st.write(f"• {sample_id}")
        else:
            st.write("**Available Columns:**")
            for col in df.columns[:3]:
                st.write(f"• {col}")
    
    with col_c:
        if st.session_state.target_column:
            st.success(f"**Target:** {st.session_state.target_column}")
            st.info(f"**Type:** {st.session_state.analysis_type}")
        else:
            st.warning("**Target not configured**")
    
    # CSV Preview Table
    st.markdown("###  Data Preview")
    preview_df = df.head(10)
    st.dataframe(preview_df, use_container_width=True, height=400)

def show_equipment_search():
    st.markdown("##  Equipment Status Search")
    
    if not st.session_state.data_uploaded:
        st.warning("Please upload data first.")
        return
    
    df = st.session_state.current_df
    
    column_info = detect_column_types(df)
    id_columns = [col for col, info in column_info.items() if info['suggested_role'] == 'id']
    
    if not id_columns:
        id_columns = [df.columns[0]]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_id_col = st.selectbox("ID Column:", id_columns)
        unique_ids = df[selected_id_col].unique()
        
        st.info(f"Found {len(unique_ids)} unique IDs in column '{selected_id_col}'")
        
        search_method = st.radio("Search Method:", ["Select from dropdown", "Type ID manually"])
        
        if search_method == "Select from dropdown":
            display_ids = unique_ids[:100] if len(unique_ids) > 100 else unique_ids
            if len(unique_ids) > 100:
                st.info(f"Showing first 100 IDs out of {len(unique_ids)} total")
            
            equipment_id = st.selectbox("Select Equipment ID:", display_ids)
        else:
            equipment_id = st.text_input("Enter Equipment ID:", placeholder="e.g., E_1, E_2, etc.")
    
    with col2:
        st.markdown("**Sample Equipment IDs:**")
        sample_ids = unique_ids[:8] if len(unique_ids) >= 8 else unique_ids
        for sample_id in sample_ids:
            st.text(f"• {sample_id}")
    
    if st.button("🔍 Search Equipment", type="primary", use_container_width=True):
        if equipment_id:
            result = search_equipment_status(df, equipment_id, selected_id_col, st.session_state.predictions)
            
            if result and result['status'] == 'found':
                st.markdown(f"""
                    <div class="equipment-found">
                        <h3>📋 Equipment {equipment_id} Found</h3>
                        <p>Located in column: {result['id_column_used']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if 'health_status' in result:
                    health = result['health_status']
                    status = health['status']
                    risk_score = health.get('risk_score', 0)
                    
                    if status == 'Healthy':
                        st.success(f" Status: {status}")
                    else:
                        st.error(f" Status: {status}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Risk Score", f"{risk_score:.1%}")
                    with col_b:
                        st.metric("Confidence", f"{health.get('confidence', 0):.1%}")
                
                if 'maintenance_recommendation' in result:
                    rec = result['maintenance_recommendation']
                    priority = rec['priority']
                    
                    if priority == 'HIGH':
                        st.error(f"🚨 {priority} PRIORITY: {rec['action']}")
                    elif priority == 'MEDIUM':
                        st.warning(f"⚠️ {priority} PRIORITY: {rec['action']}")
                    else:
                        st.info(f"✅ {priority} PRIORITY: {rec['action']}")
                    
                    st.metric("Days Until Maintenance", rec.get('days_until_maintenance', 'N/A'))
                
                st.markdown("###  Equipment Sensor Data")
                equipment_data = pd.DataFrame([result['data']])
                st.dataframe(equipment_data, use_container_width=True)
            
            elif result and result['status'] == 'not_found':
                st.markdown(f"""
                    <div class="equipment-not-found">
                        <h3>❌ Equipment {equipment_id} Not Found</h3>
                        <p>{result['message']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if result.get('available_ids'):
                    st.info(f"Available IDs: {', '.join(result['available_ids'])}")
        else:
            st.warning("Please enter an equipment ID to search.")

def show_predictions():
    st.markdown("##  Equipment Predictions")
    
    if st.session_state.predictions is None:
        st.warning("Train models first to generate predictions.")
        return
    
    predictions_df = st.session_state.predictions
    
    col1, col2, col3 = st.columns(3)
    
    if 'Health_Status' in predictions_df.columns:
        with col1:
            healthy_count = len(predictions_df[predictions_df['Health_Status'] == 'Healthy'])
            st.metric("✅ Healthy Equipment", healthy_count)
        
        with col2:
            at_risk_count = len(predictions_df[predictions_df['Health_Status'] == 'At Risk'])
            st.metric("⚠️ At Risk Equipment", at_risk_count)
        
        with col3:
            if 'Risk_Level' in predictions_df.columns:
                high_risk = len(predictions_df[predictions_df['Risk_Level'] == 'High'])
                st.metric(" High Risk", high_risk)
    
    if 'Health_Status' in predictions_df.columns:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            health_counts = predictions_df['Health_Status'].value_counts()
            fig_health = px.pie(values=health_counts.values, names=health_counts.index, 
                               title="Health Status Distribution")
            st.plotly_chart(fig_health, use_container_width=True)
        
        with col_chart2:
            if 'Risk_Level' in predictions_df.columns:
                risk_counts = predictions_df['Risk_Level'].value_counts()
                fig_risk = px.bar(x=risk_counts.index, y=risk_counts.values,
                                 title="Risk Level Distribution")
                st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown("### All Equipment Status")
    
    search_term = st.text_input(" Search Equipment ID:", placeholder="Enter equipment ID...")
    
    display_df = predictions_df.copy()
    if search_term:
        display_df = display_df[
            display_df['Equipment_ID'].astype(str).str.contains(search_term, case=False, na=False)
        ]
    
    st.dataframe(display_df, use_container_width=True)

def show_model_performance():
    st.markdown("##  Model Performance")
    
    if not st.session_state.model_performance:
        st.warning("Train models first to see performance metrics.")
        return
    
    performance_df = pd.DataFrame(st.session_state.model_performance).T
    
    col1, col2 = st.columns(2)
    
    if st.session_state.analysis_type == 'classification':
        with col1:
            fig = px.bar(x=performance_df.index, y=performance_df['Accuracy'], 
                        title='Model Accuracy Comparison')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=performance_df.index, y=performance_df['F1-Score'], 
                        title='F1-Score Comparison', color=performance_df['F1-Score'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        with col1:
            fig = px.bar(x=performance_df.index, y=performance_df['R² Score'], 
                        title='Model R² Score Comparison')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=performance_df.index, y=performance_df['RMSE'], 
                        title='RMSE Comparison')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("###  Detailed Performance Metrics")
    st.dataframe(performance_df.round(4), use_container_width=True)

def show_analytics(df):
    st.markdown("##  Analytics")
    
    if df is None:
        st.warning("No data available for analytics.")
        return
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 1:
        st.markdown("###  Feature Correlation")
        
        display_cols = numeric_columns[:8] if len(numeric_columns) > 8 else numeric_columns
        corr_matrix = df[display_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", 
                       title="Feature Correlation Heatmap")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    if numeric_columns.any():
        st.markdown("###  Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature:", numeric_columns[:10])
            
        with col2:
            chart_type = st.selectbox("Chart Type:", ["Histogram", "Box Plot"])
        
        target_col_for_viz = None
        if (st.session_state.target_column and 
            st.session_state.target_column in df.columns and
            df[st.session_state.target_column].nunique() <= 20):
            target_col_for_viz = st.session_state.target_column
        
        try:
            if chart_type == "Histogram":
                if target_col_for_viz:
                    fig = px.histogram(df, x=feature, color=target_col_for_viz, 
                                     title=f'{feature} Distribution by {target_col_for_viz}')
                else:
                    fig = px.histogram(df, x=feature, 
                                     title=f'{feature} Distribution')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                if target_col_for_viz:
                    fig = px.box(df, y=feature, color=target_col_for_viz,
                                title=f'{feature} Box Plot by {target_col_for_viz}')
                else:
                    fig = px.box(df, y=feature,
                                title=f'{feature} Box Plot')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            fig = px.histogram(df, x=feature, title=f'{feature} Distribution')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_feature_importance():
    st.markdown("##  Model Insights")
    
    if not st.session_state.feature_importance:
        st.warning("Train models first to see feature importance.")
        return
    
    model_name = st.selectbox("Select Model:", list(st.session_state.feature_importance.keys()))
    
    if model_name in st.session_state.feature_importance:
        importance_data = st.session_state.feature_importance[model_name]
        
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10]
        features, importances = zip(*sorted_features)
        
        fig = px.bar(x=list(importances), y=list(features), orientation='h',
                    title=f'{model_name} - Top Feature Importance')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# EVA VOICE ASSISTANT WIDGET
def eva_voice_widget():
    st.markdown("""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:15px;border-radius:12px;color:white;margin-bottom:20px;box-shadow: 0 4px 8px rgba(0,0,0,0.1)'>
            <h4>🤖 EVA - Equipment Voice Assitant</h4>
            <p style='margin: 5px 0; opacity: 0.9; font-size: 0.9em;'>Your intelligent maintenance assistant - Try: "Hello EVA" or "Search equipment E_1"</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎤 Voice Command", use_container_width=True, help="Click to speak with EVA"):
            user_text = recognize_speech_from_mic()
            if user_text:
                process_eva_command(user_text)
    
    with col2:
        text_input = st.text_input("💬 Chat with EVA:", placeholder="Try: 'Hello EVA' or 'search equipment E_1'")
        if st.button("Send", use_container_width=True) and text_input:
            process_eva_command(text_input)

    # Show recent conversation history
    if st.session_state.eva_conversation_history:
        with st.expander("📝 Recent Conversations"):
            for conv in st.session_state.eva_conversation_history[-3:]:
                st.markdown(f"""
                    <div style='margin: 10px 0; padding: 8px; border-left: 3px solid #667eea; background: #f8f9fa;'>
                        <strong>You ({conv['timestamp']}):</strong> {conv['user']}<br>
                        <strong>EVA:</strong> {conv['eva']}
                    </div>
                """, unsafe_allow_html=True)

# EXPORT FUNCTION
def export_results():
    if st.session_state.predictions is not None:
        csv = st.session_state.predictions.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="equipment_analysis.csv">📥 Download Results</a>'
    return None

# SIDEBAR
st.sidebar.markdown("## 🎛️ Control Panel")

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload Equipment Data CSV", type=['csv'])

# Quick settings
st.sidebar.markdown("### ⚙️ EVA Settings")
enable_voice = st.sidebar.checkbox("Enable EVA Voice Assistant", value=True)
if enable_voice:
    st.session_state.eva_voice_type = st.sidebar.radio("EVA's Voice:", ["female", "male"])

# User preferences
st.sidebar.markdown("### 👤 User Preferences")
if st.session_state.user_name:
    st.sidebar.success(f"Welcome, {st.session_state.user_name}!")
else:
    st.sidebar.info("Say 'My name is [Name]' to EVA")

# System status
st.sidebar.markdown("### 📊 System Status")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.session_state.data_uploaded:
        st.success("Data ✅")
    else:
        st.error("No Data")

with col2:
    if st.session_state.trained_models:
        st.success("Models ✅")
    else:
        st.error("No Models")

# EVA quick stats
if st.session_state.eva_conversation_history:
    st.sidebar.markdown("### 🤖 EVA Stats")
    conv_count = len(st.session_state.eva_conversation_history)
    st.sidebar.metric("Conversations", conv_count)

# Export
if st.session_state.predictions is not None:
    st.sidebar.markdown("### 📥 Export")
    download_link = export_results()
    if download_link:
        st.sidebar.markdown(download_link, unsafe_allow_html=True)

# Clear conversation history
if st.sidebar.button("🗑️ Clear EVA History"):
    st.session_state.eva_conversation_history = []
    st.sidebar.success("Conversation history cleared!")

# MAIN APPLICATION
def main():
    st.markdown('<h1 class="main-header">Comparative Analysis of ML Algorithm for Predictive maintenance of Industrial Equipments</h1>', 
                unsafe_allow_html=True)

    # EVA Voice Assistant
    if enable_voice:
        eva_voice_widget()
        st.markdown("---")

   
    df = None
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data_uploaded = True
            st.session_state.current_df = df
            st.success(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # EVA greeting for new data
            if not st.session_state.eva_conversation_history:
                greeting = get_current_time_greeting()
                if st.session_state.user_name:
                    welcome_msg = f"{greeting}, {st.session_state.user_name}! I see you've loaded new equipment data with {len(df)} records. I'm ready to help analyze equipment health and search for specific units like E_1, E_2, etc."
                else:
                    welcome_msg = f"{greeting}! I'm EVA, your equipment maintenance assistant. I've detected {len(df)} equipment records. I can search specific equipment (try 'search equipment E_1') and help with predictive maintenance analysis."
                
                st.markdown(f"""
                    <div class="eva-response">
                        <strong> EVA:</strong> {welcome_msg}
                    </div>
                """, unsafe_allow_html=True)
                
                eva_speak(welcome_msg, st.session_state.eva_voice_type)
                
                # Add to conversation historyyy
                st.session_state.eva_conversation_history.append({
                    'user': 'Data uploaded',
                    'eva': welcome_msg,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return

    # Main Application Logiccc
    if df is not None and st.session_state.data_uploaded:
        
        # Configuration
        if not st.session_state.target_column or not st.session_state.feature_columns:
            config_complete = show_column_configuration(df)
        else:
            config_complete = True
            st.success(f"✅ Config: Target={st.session_state.target_column}, Features={len(st.session_state.feature_columns)}")

        # Model Trainingggg
        if config_complete:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(" Train AI Models", type="primary", use_container_width=True):
                    with st.spinner("Training models..."):
                        try:
                            # Validate configuration
                            if not st.session_state.target_column:
                                st.error("Please select a target column first.")
                                return
                            
                            if not st.session_state.feature_columns:
                                st.error("Please select at least one feature column.")
                                return
                            
                            # Preprocess data
                            processed_df, feature_cols, target_col, encoders, target_encoder = preprocess_data_optimized(
                                df, st.session_state.target_column, st.session_state.feature_columns, 
                                st.session_state.id_column
                            )
                            
                            # Store target encoder for prediction mapping
                            st.session_state.target_encoder = target_encoder
                            
                            # Train models
                            model_results, trained_models, feature_importance, roc_data = train_fast_models(
                                processed_df, target_col, feature_cols, st.session_state.analysis_type
                            )
                            
                            if not trained_models:
                                st.error("No models were successfully trained. Please check your data and configuration.")
                                return
                            
                            # Store results
                            st.session_state.model_performance = model_results
                            st.session_state.trained_models = trained_models
                            st.session_state.feature_importance = feature_importance
                            st.session_state.roc_data = roc_data
                            
                            # Generate predictions
                            predictions = generate_predictions_fast(
                                processed_df, trained_models, feature_cols, 
                                st.session_state.analysis_type, st.session_state.id_column
                            )
                            st.session_state.predictions = predictions
                            
                            # Success message
                            st.success("✅ Models trained successfully!")
                            
                            # Show prediction summary
                            if 'Health_Status' in predictions.columns:
                                health_summary = predictions['Health_Status'].value_counts()
                                st.write("**Health Status Summary:**")
                                for status, count in health_summary.items():
                                    st.write(f"• {status}: {count}")
                            
                            # EVA success message
                            healthy_count = len(predictions[predictions['Health_Status'] == 'Healthy']) if 'Health_Status' in predictions.columns else 0
                            at_risk_count = len(predictions[predictions['Health_Status'] == 'At Risk']) if 'Health_Status' in predictions.columns else 0
                            
                            eva_success_msg = f"Excellent! I've successfully trained {len(trained_models)} AI models. Analysis complete: {healthy_count} healthy units, {at_risk_count} units need attention. I'm ready to search any equipment ID like E_1, E_2, etc!"
                            
                            st.markdown(f"""
                                <div class="eva-response">
                                    <strong>🤖 EVA:</strong> {eva_success_msg}
                                </div>
                            """, unsafe_allow_html=True)
                            
                            eva_speak(eva_success_msg, st.session_state.eva_voice_type)
                            
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
            
            with col2:
                if st.session_state.trained_models:
                    st.success(" Ready")

        # Dashboard Tabssss
        if st.session_state.data_uploaded:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", "🔍 Equipment Search", "🔮 Predictions", 
                "🧠 Models", "📈 Analytics", "📚 Insights"
            ])
            
            with tab1:
                show_overview(df)
                
            with tab2:
                show_equipment_search()
                
            with tab3:
                show_predictions()
                
            with tab4:
                show_model_performance()
                
            with tab5:
                show_analytics(df)
                
            with tab6:
                show_feature_importance()
    else:
        show_welcome_screen()

if __name__ == "__main__":
    main()
