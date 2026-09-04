# 🚀 Data Monitoring Tool

The **Data Monitoring Tool** is an **interactive Streamlit-based dashboard** that provides **comprehensive insights and quality checks** for datasets. It supports multiple formats including **CSV, Excel, JSON, Parquet, TSV, TXT, and Pickle**, making it ideal for **data analysts, data scientists, and ML engineers** who want to quickly profile, visualize, and validate their data before analysis or modeling.

---

## Key Features

- **Dataset Overview**: Instantly view the number of rows, columns, memory usage, and an overall data quality score.  
- **Data Preview**: Explore the first few rows of the dataset with an interactive table.  
- **Data Quality Analysis**: Detect null values, duplicates, and generate automated recommendations for cleaning the dataset.  
- **Data Quality API Integration**: Expose ready-to-use FastAPI endpoints for dataset quality checks from external apps and services.  
- **AI-Powered Features**: Generate AI-style summaries, risk assessments, and recommended actions from the observed data quality issues.  
- **Statistical Insights**: Generate descriptive statistics for numeric and categorical columns.  
- **Correlation & Relationships**: Visualize correlations between columns using heatmaps and tables.  
- **Outlier Detection**: Identify outliers using IQR-based methods and visualize them per column.  
- **Anomaly Detection**: Detect anomalies using Isolation Forest and display total anomalies with interactive plots.  
- **Cardinality Analysis**: Analyze unique values in each column for better feature understanding.  
- **Memory Profiling**: Monitor memory usage of each column and optimize dataset performance.  
- **Model Evaluation**: Train ML models (Random Forest) and evaluate metrics like precision, recall, F1-score, and accuracy with interactive tables and charts.

---

## Supported File Formats

- **CSV** (`.csv`)  
- **Excel** (`.xlsx`, `.xls`)  
- **JSON** (`.json`)  
- **Parquet** (`.parquet`)  
- **TSV** (`.tsv`)  
- **TXT** (`.txt`)  
- **Pickle** (`.pkl`)  

---

## Technology Stack

- **Python** – Core programming language  
- **Streamlit** – Interactive dashboard framework  
- **Pandas & NumPy** – Data manipulation and analysis  
- **Scikit-learn** – Machine learning modeling and anomaly detection  
- **Plotly** – Interactive visualizations

---

## Use Cases

- Rapid **data profiling** for new datasets  
- **Data preprocessing and cleaning** before ML tasks  
- **Monitoring dataset quality** in research or production environments  
- **Identifying anomalies and outliers** to ensure data integrity

---

## Data Drift Monitoring

The tool now includes a baseline-vs-current drift workflow for the high-priority monitoring steps:

1. **Baseline Dataset**: keep a reference sample for comparison.  
2. **Current Dataset**: compare the latest incoming data against the baseline.  
3. **Distribution Comparison**: compare numeric and categorical feature distributions.  
4. **Drift Detection**: use KS test and PSI-based drift detection.  
5. **Drift Score**: compute a per-feature score from 0 to 100.  
6. **Severity**: classify each feature as low, medium, high, or critical.  
7. **Visualization**: generate distribution charts for drift inspection.  

API usage:

- `POST /api/data-quality/analyze` with a dataset payload to get null counts, duplicates, score, and AI guidance.  
- `POST /api/data-quality/analyze-csv` with `csv` content for quality analysis.  
- `POST /api/drift/analyze` with `baseline` and `current` JSON arrays or dictionaries.  
- `POST /api/drift/analyze-csv` with `baseline_csv` and `current_csv` content.

## AI-Powered Features

The project now includes AI-style quality assessment capabilities that summarize dataset health, highlight top issues, and recommend follow-up actions based on missing values, duplicates, and risk level.


<img width="1920" height="965" alt="Screenshot 2026-03-26 192137" src="https://github.com/user-attachments/assets/88d54c28-8602-45f2-9e7c-251667701fca" />
