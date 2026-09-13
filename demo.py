import pandas as pd
import numpy as np

# Set seed for reproducible data generation
np.random.seed(42)
n_samples = 1000

# 1. Base Firmographics
account_ids = [f"ACC-{i:04d}" for i in range(1, n_samples + 1)]
hotel_types = np.random.choice(['Boutique', 'Chain/Enterprise', 'Resort', 'Independent'], size=n_samples, p=[0.25, 0.20, 0.15, 0.40])

# Scale property size dynamically based on hotel tier
room_counts = np.where(hotel_types == 'Chain/Enterprise', np.random.randint(150, 500, n_samples),
              np.where(hotel_types == 'Resort', np.random.randint(100, 300, n_samples),
              np.where(hotel_types == 'Boutique', np.random.randint(30, 100, n_samples), 
                       np.random.randint(10, 50, n_samples))))

tenure_months = np.random.randint(3, 48, size=n_samples)
# SaaS pricing tiers map loosely to room counts
monthly_revenue = room_counts * np.random.uniform(3, 6, size=n_samples) 

# 2. Hospitality-Specific Operational Risks
# Feature adoption rate (e.g., frontend + POS + housekeeping tracker vs just front desk)
feature_adoption = np.random.uniform(0.1, 0.95, size=n_samples)
# Positive value means staff usage has dropped compared to historical average
login_drop = np.random.uniform(-0.2, 0.8, size=n_samples) 
open_tickets = np.random.randint(0, 10, size=n_samples)
# Failure rates in local hardware integrations (keycards, receipt printers)
hardware_errors = np.random.uniform(0.0, 0.15, size=n_samples)

# 3. Churn Logic Formula (Mathematical representation of risk)
# High usage drops, poor adoption, hardware issues, and unresolved tickets push risk up
churn_logit = (
    0.5 * (monthly_revenue / 1000) 
    - 0.05 * tenure_months 
    - 3.5 * feature_adoption 
    + 4.5 * login_drop 
    + 0.4 * open_tickets 
    + 9.0 * hardware_errors
    - 1.2  # Intercept balancing out the raw churn rate
)

# Convert logit equations to an actual 0-1 probability curve
churn_prob = 1 / (1 + np.exp(-churn_logit))
# Assign binary churn outcome (1 = Cancelled Subscription, 0 = Active Renewal)
churn_outcome = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

# 4. Compile into clean DataFrame
hospitality_churn_df = pd.DataFrame({
    'Account_ID': account_ids,
    'Hotel_Type': hotel_types,
    'Room_Count': room_counts,
    'Tenure_Months': tenure_months,
    'Monthly_Revenue_USD': np.round(monthly_revenue, 2),
    'Feature_Adoption_Rate': np.round(feature_adoption, 2),
    'Login_Frequency_Drop': np.round(login_drop, 2),
    'Open_Support_Tickets': open_tickets,
    'Hardware_Error_Rate': np.round(hardware_errors, 2),
    'Churn_Label': churn_outcome
})

# Save to CSV for presentation software or dashboard inputs
hospitality_churn_df.to_csv('hospitality_saas_churn_prototype.csv', index=False)
print("Dataset generated successfully! Rows:", len(hospitality_churn_df))
