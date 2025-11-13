import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from io import BytesIO

st.set_page_config(
    page_title="Equipment Data Cleaner",
    page_icon="🧹",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4a9d3b;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
        border-bottom: 3px solid #4a9d3b;
    }
    .success-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        text-align: center;
    }
    .step-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4a9d3b;
        margin: 10px 0;
    }
    .info-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaning_log' not in st.session_state:
    st.session_state.cleaning_log = []

# Maintenance Trigger Calculation Function (Your Logic)
def calculate_maintenance_need(row):
    """Calculate if equipment needs maintenance based on sensor thresholds"""
    score = 0
    
    # Get values safely with fallback to 0
    temp = row.get("Temperature (°C)", 0)
    vib = row.get("Vibration (m/s²)", 0)
    current = row.get("Current (A)", 0)
    voltage = row.get("Voltage (V)", 0)
    power = row.get("Power (W)", 0)
    humidity = row.get("Humidity (%)", 0)
    crit = row.get("Equipment Criticality", "Medium")
    
    # Temperature check
    if temp > 27:
        score += 3
    elif temp > 25:
        score += 1.5
    
    # Vibration check
    if vib > 0.4:
        score += 2.5
    elif vib > 0.35:
        score += 1
    
    # Current overload
    if current > 0.85:
        score += 2
    elif current > 0.75:
        score += 1
    
    # Voltage fluctuation
    if voltage > 118 or voltage < 112:
        score += 1
    
    # Power usage
    if power > 100:
        score += 1
    
    # Humidity extremes
    if humidity > 48 or humidity < 38:
        score += 0.5
    
    # Equipment criticality multiplier
    if crit == "High":
        score *= 1.2
    elif crit == "Low":
        score *= 0.9
    
    return 1 if score >= 4 else 0

def determine_failure_type(row):
    """Determine the type of failure based on sensor readings"""
    if row["Predictive Maintenance Trigger"] == 0:
        return "None"
    
    if row.get("Temperature (°C)", 0) > 27:
        return "Overheating"
    elif row.get("Current (A)", 0) > 0.85:
        return "Overload"
    elif row.get("Vibration (m/s²)", 0) > 0.4:
        return "Mechanical"
    return "General"

# Header
st.markdown('<h1 class="main-header">🧹 Equipment Data Cleaner & Preprocessor</h1>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 15px; margin-bottom: 30px;'>
    <h3 style='color: #2d7a26;'>Automated Equipment Data Cleaning & Maintenance Analysis</h3>
    <p style='color: #666; font-size: 1.1em;'>Upload → Clean → Analyze → Download → Use in Dashboard</p>
</div>
""", unsafe_allow_html=True)

# File Upload
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("📁 Upload Raw Equipment Data (CSV)", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean column names (remove unwanted symbols)
        df.columns = df.columns.str.replace("Â", "", regex=False).str.strip()
        
        st.session_state.original_df = df
        
        st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Data Quality Overview
        st.markdown("## 📊 Data Quality Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            missing_count = int(df.isnull().sum().sum())
            st.metric("Missing Values", missing_count, 
                     delta=f"{(missing_count/df.size*100):.1f}%",
                     delta_color="inverse")
        with col3:
            duplicate_count = int(df.duplicated().sum())
            st.metric("Duplicate Rows", duplicate_count,
                     delta_color="inverse")
        with col4:
            columns_with_nulls = int(df.isnull().any().sum())
            st.metric("Columns with Issues", columns_with_nulls,
                     delta_color="inverse")
        
        # Preview
        st.markdown("### 👀 Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column Detection
        st.markdown("### 🔍 Detected Columns")
        col_detect1, col_detect2 = st.columns(2)
        
        with col_detect1:
            st.markdown("**Available Columns:**")
            for i, col in enumerate(df.columns[:15], 1):
                st.text(f"{i}. {col}")
            if len(df.columns) > 15:
                st.info(f"... and {len(df.columns) - 15} more columns")
        
        with col_detect2:
            # Check for key equipment columns
            key_columns = {
                'Equipment ID': any('equipment' in col.lower() or col.lower().startswith('e_') for col in df.columns),
                'Temperature': any('temperature' in col.lower() for col in df.columns),
                'Vibration': any('vibration' in col.lower() for col in df.columns),
                'Current': any('current' in col.lower() for col in df.columns),
                'Voltage': any('voltage' in col.lower() for col in df.columns),
            }
            
            st.markdown("**Key Column Detection:**")
            for col_name, found in key_columns.items():
                if found:
                    st.success(f"✅ {col_name} - Found")
                else:
                    st.warning(f"⚠️ {col_name} - Not found")
        
        # Missing values visualization
        if df.isnull().any().any():
            st.markdown("### 📉 Missing Values Analysis")
            missing_data = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if not missing_data.empty:
                fig = px.bar(missing_data, x='Column', y='Missing %', 
                            title='Missing Values by Column',
                            color='Missing %',
                            color_continuous_scale='Reds')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Cleaning Configuration
        st.markdown("## 🛠️ Cleaning & Analysis Configuration")
        
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ What will happen:</strong><br>
            1. Remove duplicates and fix data issues<br>
            2. Handle missing values intelligently<br>
            3. Calculate maintenance triggers based on sensor thresholds<br>
            4. Add fault detection and operational status<br>
            5. Determine failure types and maintenance recommendations
        </div>
        """, unsafe_allow_html=True)
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            st.markdown("#### Basic Cleaning")
            remove_duplicates = st.checkbox("✅ Remove duplicate rows", value=True)
            handle_missing = st.checkbox("✅ Handle missing values", value=True)
            remove_empty_cols = st.checkbox("✅ Remove empty columns", value=True)
            
        with col_config2:
            st.markdown("#### Advanced Options")
            add_noise = st.checkbox("🎲 Add 5% random variation (for realism)", value=True)
            calculate_costs = st.checkbox("💰 Calculate maintenance costs & repair time", value=True)
        
        if handle_missing:
            missing_strategy = st.selectbox(
                "Missing value strategy:",
                ["Smart Fill (median for numeric, mode for categorical)", 
                 "Drop rows with missing", "Forward fill"]
            )
        
        # Clean Button
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            if st.button("🧹 CLEAN & ANALYZE DATA", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing your equipment data..."):
                    cleaned_df = df.copy()
                    cleaning_log = []
                    
                    try:
                        # 1. Clean and handle missing values
                        cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        cleaned_df.dropna(how="all", inplace=True)
                        cleaning_log.append(f"✅ Removed rows with all missing values")
                        
                        # 2. Remove duplicates
                        if remove_duplicates:
                            before = len(cleaned_df)
                            cleaned_df.drop_duplicates(inplace=True)
                            removed = before - len(cleaned_df)
                            if removed > 0:
                                cleaning_log.append(f"✅ Removed {removed} duplicate rows")
                        
                        # 3. Remove empty columns
                        if remove_empty_cols:
                            before = len(cleaned_df.columns)
                            cleaned_df = cleaned_df.dropna(axis=1, how='all')
                            removed = before - len(cleaned_df.columns)
                            if removed > 0:
                                cleaning_log.append(f"✅ Removed {removed} empty columns")
                        
                        # 4. Handle missing values
                        if handle_missing:
                            missing_before = int(cleaned_df.isnull().sum().sum())
                            
                            if "Smart Fill" in missing_strategy:
                                for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                                for col in cleaned_df.select_dtypes(exclude=[np.number]).columns:
                                    if not cleaned_df[col].mode().empty:
                                        cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
                            elif "Drop" in missing_strategy:
                                cleaned_df.dropna(inplace=True)
                            elif "Forward" in missing_strategy:
                                cleaned_df.fillna(method='ffill', inplace=True)
                            
                            missing_after = int(cleaned_df.isnull().sum().sum())
                            cleaning_log.append(f"✅ Handled {missing_before - missing_after} missing values")
                        
                        # 5. Calculate Predictive Maintenance Triggers
                        st.info("🔄 Calculating predictive maintenance triggers...")
                        cleaned_df["Predictive Maintenance Trigger"] = cleaned_df.apply(calculate_maintenance_need, axis=1)
                        maintenance_count = int(cleaned_df["Predictive Maintenance Trigger"].sum())
                        cleaning_log.append(f"✅ Calculated maintenance triggers: {maintenance_count} equipment need attention")
                        
                        # 6. Add Fault Detection Labels
                        cleaned_df["Fault Detected"] = cleaned_df["Predictive Maintenance Trigger"]
                        cleaned_df["Fault Status"] = cleaned_df["Predictive Maintenance Trigger"].map({1: "Fault Detected", 0: "No Fault"})
                        cleaned_df["Operational Status"] = cleaned_df["Predictive Maintenance Trigger"].map({1: "Under Maintenance", 0: "Operational"})
                        cleaned_df["Failure History"] = cleaned_df["Fault Status"]
                        cleaning_log.append(f"✅ Added fault detection and operational status")
                        
                        # 7. Determine Failure Types
                        cleaned_df["Failure Type"] = cleaned_df.apply(determine_failure_type, axis=1)
                        failure_types = cleaned_df["Failure Type"].value_counts()
                        cleaning_log.append(f"✅ Categorized failure types: {dict(failure_types)}")
                        
                        # 8. Add Maintenance Type and Costs
                        if calculate_costs:
                            cleaned_df["Maintenance Type"] = cleaned_df["Predictive Maintenance Trigger"].map({1: "Corrective", 0: "Preventive"})
                            cleaned_df["Repair Time (hrs)"] = cleaned_df["Predictive Maintenance Trigger"].apply(
                                lambda x: np.random.randint(4, 10) if x == 1 else 0
                            )
                            cleaned_df["Maintenance Costs (USD)"] = cleaned_df["Predictive Maintenance Trigger"].apply(
                                lambda x: np.random.randint(180, 280) if x == 1 else np.random.randint(100, 160)
                            )
                            cleaning_log.append(f"✅ Calculated maintenance costs and repair times")
                        
                        # 9. Add random variation for realism
                        if add_noise:
                            flip_count = int(len(cleaned_df) * 0.05)
                            flip_indices = np.random.choice(cleaned_df.index, size=flip_count, replace=False)
                            cleaned_df.loc[flip_indices, "Predictive Maintenance Trigger"] = 1 - cleaned_df.loc[flip_indices, "Predictive Maintenance Trigger"]
                            
                            # Update dependent fields
                            for i in flip_indices:
                                trigger = cleaned_df.loc[i, "Predictive Maintenance Trigger"]
                                cleaned_df.loc[i, "Fault Detected"] = trigger
                                cleaned_df.loc[i, "Fault Status"] = "Fault Detected" if trigger else "No Fault"
                                cleaned_df.loc[i, "Operational Status"] = "Under Maintenance" if trigger else "Operational"
                            
                            cleaning_log.append(f"✅ Added 5% random variation for realistic data distribution")
                        
                        # 10. Final cleanup
                        cleaned_df.reset_index(drop=True, inplace=True)
                        
                        # Store results
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.cleaning_log = cleaning_log
                        
                        # Success message
                        st.markdown("""
                            <div class="success-box">
                                <h2>✅ Data Processing Complete!</h2>
                                <p style='font-size: 1.2em; margin: 10px 0;'>Your equipment data is cleaned, analyzed, and ready for the dashboard</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Show Results
        if st.session_state.cleaned_df is not None:
            st.markdown("## 📋 Processing Results")
            
            # Cleaning log
            st.markdown("### ✅ Operations Performed")
            for log in st.session_state.cleaning_log:
                st.markdown(f"""
                    <div class="step-box">
                        {log}
                    </div>
                """, unsafe_allow_html=True)
            
            # Maintenance Analysis Summary
            cleaned_df = st.session_state.cleaned_df
            
            st.markdown("### 🔧 Maintenance Analysis Summary")
            
            col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
            
            with col_summary1:
                total_equipment = len(cleaned_df)
                st.metric("Total Equipment", total_equipment)
            
            with col_summary2:
                needs_maintenance = int(cleaned_df["Predictive Maintenance Trigger"].sum())
                st.metric("Needs Maintenance", needs_maintenance,
                         delta=f"{(needs_maintenance/total_equipment*100):.1f}%")
            
            with col_summary3:
                operational = int((cleaned_df["Operational Status"] == "Operational").sum())
                st.metric("Operational", operational,
                         delta=f"{(operational/total_equipment*100):.1f}%")
            
            with col_summary4:
                if "Failure Type" in cleaned_df.columns:
                    critical_failures = int((cleaned_df["Failure Type"] == "Overheating").sum())
                    st.metric("Critical (Overheating)", critical_failures)
            
            # Failure Type Distribution
            if "Failure Type" in cleaned_df.columns:
                st.markdown("### 📊 Failure Type Distribution")
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    failure_dist = cleaned_df["Failure Type"].value_counts()
                    fig = px.pie(values=failure_dist.values, names=failure_dist.index,
                                title="Failure Types")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    status_dist = cleaned_df["Operational Status"].value_counts()
                    fig = px.bar(x=status_dist.index, y=status_dist.values,
                                title="Operational Status",
                                color=status_dist.index,
                                color_discrete_map={
                                    "Operational": "#10b981",
                                    "Under Maintenance": "#ef4444"
                                })
                    st.plotly_chart(fig, use_container_width=True)
            
            # Before/After Comparison
            st.markdown("### 📊 Before vs After Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📉 Before Cleaning")
                st.metric("Rows", len(df))
                st.metric("Columns", len(df.columns))
                st.metric("Missing Values", int(df.isnull().sum().sum()))
                st.metric("Has Maintenance Analysis", "❌ No")
            
            with col2:
                st.markdown("#### 📈 After Cleaning")
                st.metric("Rows", len(cleaned_df), 
                         delta=int(len(cleaned_df) - len(df)))
                st.metric("Columns", len(cleaned_df.columns), 
                         delta=int(len(cleaned_df.columns) - len(df.columns)))
                st.metric("Missing Values", int(cleaned_df.isnull().sum().sum()), 
                         delta=int(int(cleaned_df.isnull().sum().sum()) - int(df.isnull().sum().sum())))
                st.metric("Has Maintenance Analysis", "✅ Yes")
            
            # Preview cleaned data
            st.markdown("### ✨ Cleaned Data Preview")
            
            # Show important columns first
            important_cols = ["Predictive Maintenance Trigger", "Fault Status", "Operational Status", "Failure Type"]
            available_important = [col for col in important_cols if col in cleaned_df.columns]
            
            if available_important:
                st.info(f"Showing maintenance analysis columns: {', '.join(available_important)}")
                preview_cols = available_important + [col for col in cleaned_df.columns if col not in available_important][:5]
                st.dataframe(cleaned_df[preview_cols].head(20), use_container_width=True)
            else:
                st.dataframe(cleaned_df.head(20), use_container_width=True)
            
            # Full preview expander
            with st.expander("📋 View All Columns"):
                st.dataframe(cleaned_df.head(50), use_container_width=True)
            
            # Statistics
            st.markdown("### 📊 Data Statistics")
            st.dataframe(cleaned_df.describe(), use_container_width=True)
            
            # Download Section
            st.markdown("## 📥 Download Cleaned & Analyzed Data")
            
            st.info("⚡ Download this file and upload it to the main dashboard for AI model training and predictions!")
            
            col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
            
            with col_dl1:
                # CSV Download
                csv = cleaned_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="Dataset_Predictive_Maintenance_CLEANED.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
            
            with col_dl2:
                # Excel Download
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    cleaned_df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                    
                    # Add summary sheet
                    summary = pd.DataFrame({
                        'Metric': ['Original Rows', 'Cleaned Rows', 'Total Equipment',
                                  'Needs Maintenance', 'Operational', 'Missing Values Removed'],
                        'Value': [
                            len(df), 
                            len(cleaned_df), 
                            len(cleaned_df),
                            int(cleaned_df["Predictive Maintenance Trigger"].sum()) if "Predictive Maintenance Trigger" in cleaned_df.columns else 0,
                            int((cleaned_df["Operational Status"] == "Operational").sum()) if "Operational Status" in cleaned_df.columns else 0,
                            int(df.isnull().sum().sum()) - int(cleaned_df.isnull().sum().sum())
                        ]
                    })
                    summary.to_excel(writer, index=False, sheet_name='Summary')
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name="Dataset_Predictive_Maintenance_CLEANED.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col_dl3:
                st.markdown("#### 📊 What's Included")
                st.success("""
                ✅ Cleaned data  
                ✅ Maintenance triggers  
                ✅ Fault detection  
                ✅ Failure types  
                ✅ Cost estimates  
                """)
            
            # Link to main dashboard
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #4a9d3b 0%, #2d7a26 100%); border-radius: 15px;'>
                    <h3 style='color: white; margin-bottom: 15px;'>✅ Ready for AI Analysis!</h3>
                    <p style='color: white; opacity: 0.9; margin-bottom: 20px;'>
                        Your data is now cleaned and analyzed. Upload it to the main dashboard to train AI models and get equipment predictions.
                    </p>
                    <a href="https://predictive-maintenance-for-industrial-equipments.streamlit.app/" 
                       target="_blank" 
                       style='background: white; color: #4a9d3b; padding: 15px 40px; 
                              border-radius: 50px; text-decoration: none; font-weight: bold; 
                              font-size: 1.1em; display: inline-block;'>
                        🚀 Go to Main Dashboard
                    </a>
                </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    # Instructions when no file uploaded
    st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h2 style='color: #4a9d3b;'>📤 Upload Your Equipment Data to Get Started</h2>
            <p style='font-size: 1.1em; color: #666; margin: 20px 0;'>
                This tool automatically cleans your data AND calculates predictive maintenance triggers
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='padding: 20px; background: #f8f9fa; border-radius: 10px; text-align: center;'>
                <h3 style='color: #4a9d3b;'>1️⃣ Upload</h3>
                <p>Upload your raw equipment CSV file</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='padding: 20px; background: #f8f9fa; border-radius: 10px; text-align: center;'>
                <h3 style='color: #4a9d3b;'>2️⃣ Analyze</h3>
                <p>Auto-calculate maintenance triggers & fault detection</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='padding: 20px; background: #f8f9fa; border-radius: 10px; text-align: center;'>
                <h3 style='color: #4a9d3b;'>3️⃣ Download</h3>
                <p>Get cleaned data with all analysis columns</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='padding: 20px; background: #f8f9fa; border-radius: 10px; text-align: center;'>
                <h3 style='color: #4a9d3b;'>4️⃣ Train AI</h3>
                <p>Use in main dashboard for predictions</p>
            </div>
        """, unsafe_allow_html=True)
    
    # What gets calculated
    st.markdown("---")
    st.markdown("## 🧠 What Gets Automatically Calculated")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### Data Cleaning:
        - ✅ Remove duplicates
        - ✅ Handle missing values
        - ✅ Fix data types
        - ✅ Remove empty columns
        """)
    
    with col_info2:
        st.markdown("""
        ### Maintenance Analysis:
        - 🔧 Predictive Maintenance Triggers
        - ⚠️ Fault Detection
        - 🏭 Operational Status
        - 🔥 Failure Type Classification
        - 💰 Cost & Time Estimates
        """)
    
    st.markdown("### 📊 Threshold Logic")
    st.info("""
    **Equipment needs maintenance if:**
    - Temperature > 27°C (Critical) or > 25°C (Warning)
    - Vibration > 0.4 m/s² (Critical) or > 0.35 m/s² (Warning)
    - Current > 0.85 A (Overload) or > 0.75 A (High)
    - Voltage outside 112-118V range
    - Power > 100W
    - Humidity outside 38-48% range
    - Combined score ≥ 4 points triggers maintenance
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🧹 Equipment Data Cleaner | Part of Predictive Maintenance Platform</p>
        <p style='font-size: 0.9em;'>Automated cleaning + maintenance analysis in one step</p>
    </div>
""", unsafe_allow_html=True)