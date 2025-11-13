"""
✅ Fixed Dataset Cleaner for Predictive Maintenance Project
Generates a fully corrected dataset ready for Streamlit AI Dashboard
"""

import pandas as pd
import numpy as np

print("🔧 Loading dataset...")

# Load dataset
df = pd.read_csv("Dataset_Predective_Maintanance.csv")

# Clean column names (remove unwanted symbols like Â)
df.columns = df.columns.str.replace("Â", "", regex=False).str.strip()

print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")

# Clean and handle missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(how="all", inplace=True)

# --- Step 1: Define key thresholds for sensors ---
def calculate_maintenance_need(row):
    score = 0

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

    # Equipment criticality adds multiplier
    if crit == "High":
        score *= 1.2
    elif crit == "Low":
        score *= 0.9

    return 1 if score >= 4 else 0


print("\n🔄 Calculating predictive maintenance triggers...")
df["Predictive Maintenance Trigger"] = df.apply(calculate_maintenance_need, axis=1)

# --- Step 2: Label Faults and Operational Status ---
df["Fault Detected"] = df["Predictive Maintenance Trigger"]
df["Fault Status"] = df["Predictive Maintenance Trigger"].map({1: "Fault Detected", 0: "No Fault"})
df["Operational Status"] = df["Predictive Maintenance Trigger"].map({1: "Under Maintenance", 0: "Operational"})
df["Failure History"] = df["Fault Status"]

# --- Step 3: Determine failure types ---
def determine_failure_type(row):
    if row["Predictive Maintenance Trigger"] == 0:
        return "None"

    if row["Temperature (°C)"] > 27:
        return "Overheating"
    elif row["Current (A)"] > 0.85:
        return "Overload"
    elif row["Vibration (m/s²)"] > 0.4:
        return "Mechanical"
    return "General"

df["Failure Type"] = df.apply(determine_failure_type, axis=1)

# --- Step 4: Maintenance type and cost/time estimation ---
df["Maintenance Type"] = df["Predictive Maintenance Trigger"].map({1: "Corrective", 0: "Preventive"})
df["Repair Time (hrs)"] = df["Predictive Maintenance Trigger"].apply(lambda x: np.random.randint(4, 10) if x == 1 else 0)
df["Maintenance Costs (USD)"] = df["Predictive Maintenance Trigger"].apply(
    lambda x: np.random.randint(180, 280) if x == 1 else np.random.randint(100, 160)
)

# --- Step 5: Add small variation (noise for realism) ---
print("🎲 Adding 5% random variation to triggers...")
flip_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
df.loc[flip_indices, "Predictive Maintenance Trigger"] = 1 - df.loc[flip_indices, "Predictive Maintenance Trigger"]

# Update dependent fields
for i in flip_indices:
    trigger = df.loc[i, "Predictive Maintenance Trigger"]
    df.loc[i, "Fault Detected"] = trigger
    df.loc[i, "Fault Status"] = "Fault Detected" if trigger else "No Fault"
    df.loc[i, "Operational Status"] = "Under Maintenance" if trigger else "Operational"

# --- Step 6: Balance summary ---
print("\n📊 New Maintenance Trigger Distribution:")
print(df["Predictive Maintenance Trigger"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# --- Step 7: Final clean-up ---
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Step 8: Save final dataset ---
output_path = "Dataset_Predictive_Maintenance_FIXED.csv"
df.to_csv(output_path, index=False)

print("\n💾 Saved cleaned dataset as:", output_path)
print("✅ Ready to use in your Streamlit dashboard!")

# --- Preview ---
print("\n🔍 Sample:")
print(df.head(5))
