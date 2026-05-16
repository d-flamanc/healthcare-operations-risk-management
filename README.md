<img width="546" height="304" alt="1-pager_Visual" src="https://github.com/user-attachments/assets/f4231920-b04f-45c1-9565-c860430bd554" />
<img width="542" height="305" alt="Model_Visual" src="https://github.com/user-attachments/assets/f6e7d5c6-7382-4cb9-abea-8842963c1048" />


# Healthcare Operations Risk Management
### Staffed Capacity Risk Forecasting System (GBM + SVM Ensemble)
---

## Executive Summary

This project delivers a machine learning–driven decision-support system designed to forecast near-term hospital capacity risk. The model predicts whether staffed capacity utilization is likely to exceed a defined high-risk threshold, enabling earlier and more structured operational intervention.

The system is intended to improve the timing, consistency, and quality of capacity management decisions. It supports—but does not replace—clinical and operational judgment.

---

## Business Problem

Healthcare systems face capacity risk when patient demand exceeds available staffing resources. This can lead to:

- Delayed care and throughput bottlenecks  
- Patient safety risk  
- Staff burnout and inefficiencies  
- Financial and operational strain  

Traditional monitoring methods are reactive. This system introduces **forward-looking risk detection**.

---

## Solution Overview

The system transforms raw patient flow data into a structured, predictive operational signal and translates it into actionable decision categories.

### Decision Framework

- **NO ACTION**: No immediate risk detected  
- **REVIEW**: Elevated risk requiring operational assessment  
- **AUTO ESCALATE**: High-confidence risk requiring immediate escalation  
- **CRISIS (override)**: Manual escalation for external constraints  

---

### Pipeline Summary

1. **Input Layer**
   - Admissions and discharges  
   - Dates and room identifiers  

2. **Data Processing**
   - Data cleaning and normalization  
   - Canonical entity mapping  

3. **Operational Reconstruction**
   - Patient stays and event stream  
   - Daily census and flow metrics  

4. **Capacity Signal Generation**
   - Staffed capacity estimation  
   - Utilization rate calculation  
   - High-risk labeling (>85%)  

5. **Feature Engineering**
   - Lag variables (short- and long-term)  
   - Rolling averages and volatility  
   - Seasonal and cyclical patterns  

6. **Model Layer**
   - Gradient Boosting Machine (GBM)  
   - Support Vector Machine (SVM)  
   - Ensemble probability estimation  

7. **Decision Layer**
   - Two-tier threshold system  
   - Safety-weighted classification logic  

---

## Key Capabilities

- Daily capacity risk forecasting (1–7 day horizon)  
- Early detection of high-utilization periods (>85%)  
- Ensemble modeling for improved robustness  
- Configurable safety vs workload balance  
- Operationally interpretable outputs  

---

## Governance and Controls

The system is designed for controlled operational use:

- Thresholds are configurable and subject to governance review  
- Alert volume is managed using workload caps  
- Model performance is monitored using recall, precision, and flag rate  
- Outputs are used to support structured decision-making  

---

## Performance Philosophy

This system prioritizes **patient safety and operational awareness**:

- **Recall (Sensitivity)** is emphasized to detect high-risk days  
- **Precision** is managed to limit unnecessary workload  
- **Flag Rate** ensures the system remains usable in practice  

Forecast accuracy decreases at longer horizons; short-term predictions are most actionable.
