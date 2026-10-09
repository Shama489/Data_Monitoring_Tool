# Data Monitoring Tool

The **Data Monitoring Tool** provides dataset profiling, quality checks, drift analysis, forecasting, model explanations, and notification delivery through a FastAPI service. The profiling functions can also support a separate dashboard client.

## Run the API

Activate the project virtual environment and start the service:

```powershell
.\venv\Scripts\Activate.ps1
$env:AUTH_SECRET_KEY = (python -c "import secrets; print(secrets.token_urlsafe(48))")
$env:ADMIN_USERNAME = "admin"
$env:ADMIN_PASSWORD = "replace-with-a-unique-password-of-12-or-more-characters"
$env:DATASET_ALLOWED_DIRS = "C:\data\incoming;C:\data\approved"
uvicorn main:app --reload
```

On first login, the configured bootstrap administrator is created if the user
database is empty. The API is available at `http://127.0.0.1:8000`, with
interactive documentation at `/docs`. Keep `AUTH_SECRET_KEY` stable and secret:
it signs bearer tokens and derives the encryption key for stored datasets and
monitoring results. Losing or changing it makes existing encrypted data
unreadable. Do not use the example bootstrap password in a deployed environment.

All API routes except `/`, `/api/health`, and `/api/auth/login` require a
`Bearer` access token. Login with `POST /api/auth/login`; the returned access
token expires after one hour. Administrators create and manage accounts with
`POST`/`GET /api/auth/users` and `PATCH /api/auth/users/{id}`. Passwords are
stored as salted scrypt hashes. Five failed login attempts from one client IP
within 15 minutes trigger a temporary block with a `Retry-After` response.
Roles are:

- **admin**: user, configuration, alert-rule, and audit administration; can
  access all users' stored datasets and monitoring results.
- **analyst**: run monitoring and analysis, upload/manage their own datasets,
  and use assistant/notification actions.
- **viewer**: read their own datasets and monitoring results, and ask the
  assistant; data-changing actions are denied.

Datasets and monitoring results are encrypted at rest with authenticated
encryption. Dataset and result records are scoped to their owner; non-admin
users receive not-found responses for records they do not own. Requests larger
than 10 MiB are rejected. The XAI Streamlit dashboard signs in through the API,
stores its upload under the signed-in account, and disables analysis controls
for viewers. CSV dataset uploads are limited to 100,000 rows and 1,000 columns.
Local-file sources are disabled unless `DATASET_ALLOWED_DIRS` lists the
directories the API is permitted to read. Paths are canonicalized (including
symlinks), constrained to those roots, and restricted to supported tabular file
extensions before opening.
Set `API_BASE_URL` if the API is not at the default local address.
Use HTTPS for API traffic outside localhost and protect the secret environment
variables and database backups.

The optional PostgreSQL helper in `backend.py` no longer supplies a development
password. Configure `DB_PASSWORD` when required by the database; credentials are
assembled with SQLAlchemy's URL builder so reserved characters are handled safely.

## Persistent monitoring data

The API stores datasets, monitoring results, alert rules and history, configuration
values, and audit records in SQLite. By default, the database is placed in the
user's local application-data directory. Set `MONITORING_DB_PATH` before starting
the API to select another database file:

```powershell
$env:MONITORING_DB_PATH = "C:\data\monitoring.db"
uvicorn main:app --reload
```

The existing `alerts.db` alert history is migrated to the new database on first
use. Available management endpoints include:

- `POST /api/datasets`, `GET /api/datasets`, and `GET`/`DELETE /api/datasets/{id}`
- `POST /api/datasets/upload?name=events.csv` for bounded raw CSV uploads
- `GET /api/monitoring/results` and `GET /api/monitoring/results/{id}`
- `GET`/`POST`/`DELETE /api/alerts/rules` and `GET /api/alerts/history`
- `GET`/`PUT /api/configurations` and `DELETE /api/configurations/{key}`
- `GET /api/audit`
- `POST /api/auth/login`, `GET /api/auth/me`, and admin-only user management at
  `/api/auth/users`

Monitoring workflows automatically persist their input datasets and result.
Saved datasets can be monitored again by passing their `dataset_id` to
`POST /api/monitoring/analyze`.

## Dataset lineage, versions, and comparison

Datasets are immutable after creation. Create a new version with
`POST /api/datasets/{id}/versions`, providing replacement `data` and an optional
`change_summary`; each version links to its parent and is encrypted like any
other saved dataset. List the full version chain with
`GET /api/datasets/{id}/versions`.

Compare two same-owner datasets or versions with
`POST /api/datasets/compare` and `baseline_dataset_id` /
`current_dataset_id`. The persisted comparison includes added, removed, and
type-changed columns; row counts, missing values, duplicate rows, quality
scores, and per-column distribution drift where shared columns are available.
Comparisons appear in monitoring history.

Dataset ingestion, version creation, successful monitoring and comparisons,
and failed monitoring attempts are recorded in the lineage event store. Use
`GET /api/datasets/{id}/lineage` to inspect the dataset's version family,
provenance, pipeline stages, inputs, results, and failures. Events and versions
are scoped to the dataset owner; analysts can create versions of their own
datasets, while viewers have read-only access. The dashboard's **Lineage &
versions** tab supports history inspection, side-by-side comparisons, and
version creation.

## Real-time and scheduled monitoring

The API runs a persistent scheduler worker during application startup. Create a
recurring check with `POST /api/schedules` using an owned `dataset_id`,
`interval_seconds` (60 seconds to 31 days), selected checks, and optional
freshness settings (`timestamp_column` plus `max_age_hours`). A freshness
schedule always includes the quality check. Drift schedules also require an
owned `baseline_dataset_id`; forecasts and explanations require the date and
target columns. Add `alert_channels` (for example
`[{"channel":"email","recipient":"ops@example.com"}]`) to deliver notifications
when a scheduled run finds an issue. Every run is persisted in monitoring
history; `GET /api/schedules` reports the next run and most recent status. Pause
or resume with `PATCH /api/schedules/{id}` and remove a job with
`DELETE /api/schedules/{id}`. Administrators can manage all schedules; analysts
can manage the schedules they own. Due jobs are claimed transactionally in
SQLite so multiple API workers do not normally launch the same run at once.
The dashboard's **Schedules** tab provides the same controls.

## Run the Monitoring Dashboard

Start the interactive dashboard in a second terminal:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run xai_dashboard.py
```

Sign in with the bootstrap administrator or an account created by an
administrator. The default workspace combines quality scores, drift trends,
anomaly counts, configurable monitoring runs, forecasts, XAI results, alert
history, saved historical reports, and dataset lineage/version comparisons.
Users can upload or select datasets permitted by their role. The **Explainable AI** workspace retains the
standalone classification/regression workflow, target selection, holdout
evaluation, feature importance, per-row SHAP explanations, and JSON download.

## Advanced ML analytics

Authenticated analysts can submit specialized workflows to
`POST /api/analytics/advanced-ml`. Every successful analysis persists its input
dataset and report. Set `analysis` to one of:

- `clustering`: K-means segments rows; supports `n_clusters` and optional
  `feature_columns`, and reports cluster profiles, row assignments, and a
  sampled silhouette score.
- `fraud_detection`: Isolation Forest ranks unusual records for review; supports
  `contamination` and optional `feature_columns`. Flags indicate anomalies, not
  confirmed fraud.
- `predictive_maintenance`: compares supervised models for a failure/event
  target (`task: "classification"`) or remaining-useful-life/numeric target
  (`task: "regression"`); accepts `target_column`, `metric`, `test_size`, and
  optional `model_names`. Validate time-based splits before operational use.
- `recommendation`: returns item-similarity recommendations for a user, with a
  popularity fallback for new users or large interaction matrices. Provide
  `user_column`, `item_column`, optional `rating_column`, `user_id`, and
  `top_n`. When `user_id` is omitted, it returns globally popular items.

For example, cluster a dataset with
`{"analysis":"clustering","n_clusters":2,"data":[{"temperature":18},{"temperature":19},{"temperature":41},{"temperature":42}]}`.

---

## Key Features

- **Dataset Overview**: Instantly view the number of rows, columns, memory usage, and an overall data quality score.  
- **Data Preview**: Explore the first few rows of the dataset with an interactive table.  
- **Data Quality Analysis**: Detect null values, duplicates, and generate automated recommendations for cleaning the dataset.  
- **Data Quality API Integration**: Expose ready-to-use FastAPI endpoints for dataset quality checks from external apps and services.  
- **Enterprise Source Monitoring**: Analyze multiple local files, SQL databases, MongoDB collections, and cloud objects in one request.
- **Persistent Monitoring State**: Store datasets, analysis results, alert rules and history, configuration, and audit records in SQLite.
- **Advanced Monitoring Dashboard**: Unified authenticated workspace for quality scores, drift and anomaly trends, configurable monitoring runs, forecasts, XAI reports, alerts, and historical results.
- **Data Lineage & Pipeline Monitoring**: Owner-scoped provenance, immutable version links, monitoring/comparison pipeline stages, and recorded failures.
- **Dataset Comparison & Versioning**: Create linked dataset versions and compare schema, types, quality, missing values, duplicates, and statistical drift.
- **Authentication, RBAC & Security**: Multi-user bearer-token login, admin/analyst/viewer permissions, account administration, user-scoped datasets and results, password hashing, encrypted stored uploads, and request-size limits.
- **AI-Powered Features**: Generate AI-style summaries, risk assessments, and recommended actions from the observed data quality issues.  
- **Statistical Insights**: Generate descriptive statistics for numeric and categorical columns.  
- **Correlation & Relationships**: Visualize correlations between columns using heatmaps and tables.  
- **Outlier Detection**: Identify outliers using IQR-based methods and visualize them per column.  
- **Anomaly Detection**: Detect anomalies using Isolation Forest and display total anomalies with interactive plots.  
- **Cardinality Analysis**: Analyze unique values in each column for better feature understanding.  
- **Memory Profiling**: Monitor memory usage of each column and optimize dataset performance.  
- **Model Evaluation**: Train ML models (Random Forest) and evaluate metrics like precision, recall, F1-score, and accuracy with interactive tables and charts.
- **Advanced ML Analytics**: Run K-means clustering, anomaly-based fraud review, predictive-maintenance model selection, and item recommendations through an authenticated API.
- **Explainable AI Dashboard**: Explore Random Forest feature importance and optional SHAP explanations through an interactive Streamlit workflow.

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
- **SQLAlchemy** – PostgreSQL and MySQL database connectivity
- **SQLite** – Durable application state and alert management

---

## Use Cases

- Rapid **data profiling** for new datasets  
- **Data preprocessing and cleaning** before ML tasks  
- **Monitoring dataset quality** in research or production environments  
- **Identifying anomalies and outliers** to ensure data integrity

---

# Dashboard Features

### 15. Custom Dashboards

Users can:

- Create widgets
- Save layouts
- Pin important metrics

### 16. Dark Mode

- Provide a low-glare interface for extended monitoring sessions
- Improve readability across charts, tables, and metric panels

### 17. Multi-language Support

- Localize dashboard labels, tooltips, and summaries
- Support global users in different regional contexts

### 18. Mobile Responsive Dashboard

- Adapt layouts for phones and tablets
- Keep key KPIs, charts, and filters accessible on smaller screens
- Ensure a smooth monitoring experience across devices

# 📑 Reporting Features

### 27. Scheduled Reports

Generate reports:

- Daily
- Weekly
- Monthly

### 28. Download Reports

- PDF
- Excel
- PowerPoint

### 29. Executive Dashboard

Display:

- KPIs
- Trends
- Data health score

# Advanced Machine Learning

### 30. Fraud Detection

- Detect suspicious transaction patterns and anomalous behaviors using supervised and unsupervised models.
- Flag high-risk records for review before they affect downstream reporting or financial decisions.

### 31. Data Classification

- Categorize records into business-defined classes using probabilistic and tree-based classifiers.
- Measure precision, recall, and confidence to support operational decision-making.

### 32. Clustering Analysis

- Group similar records into clusters for segmentation, pattern discovery, and customer or event profiling.
- Identify hidden structure in large datasets without requiring labeled training targets.

### 33. Recommendation Engine

- Recommend relevant products, actions, or next steps based on patterns in historical behavior.
- Combine similarity-based and ranking-based techniques to improve personalization and efficiency.

### 34. Predictive Maintenance for Data Pipelines

- Monitor pipeline health, data freshness, and failure signals to anticipate maintenance needs.
- Reduce downtime by forecasting regression risk and recommending proactive intervention.

# � Data Quality Features

### 1. Schema Validation

- Detect added, removed, or renamed columns.
- Validate data types automatically.

### 2. Near-duplicate Detection

- Identify exact duplicates and near-duplicate records using similarity matching.
- Highlight repeated or highly similar entries that may distort downstream insights.

### 3. Data Freshness Monitoring

- Detect when data has not been updated within an expected time window.
- Surface stale datasets before they affect reporting or decision-making.

### 4. Business Rule Validation

- Validate custom rules such as age not being negative, salary not being less than zero, or email addresses containing a valid domain.
- Flag records that fail business requirements before analytics or ML workflows proceed.

# �🚀 Cutting-Edge Features (Very Impressive)

### 35. Chatbot Assistant

- Provide an in-dashboard assistant that answers user questions about data quality, drift, alerts, and trends.
- Help analysts get instant explanations without digging through raw reports.

### 36. Voice Commands

- Allow users to trigger actions like "Show anomaly report" or "Open drift summary" using voice input.
- Improve accessibility and hands-free monitoring for operational teams.

### 37. Generative AI Report Summaries

- Automatically generate concise insights, actionable recommendations, and executive summaries from monitored data.
- Surface the key story behind metrics in plain language for non-technical stakeholders.

### 38. Data Lineage Visualization

- Show where data originated, how it moved through pipelines, and where transformations occurred.
- Make audits, debugging, and trust analysis much easier for data teams.

### 39. Pipeline Monitoring

- Track ETL jobs, failed executions, latency, throughput, and job health in one place.
- Detect regressions before they impact downstream analytics or production data quality.

### 40. Dataset Comparison Tool

- Compare two datasets side-by-side and highlight schema, value, and drift differences.
- Help teams identify changes in data shape, quality, and business meaning over time.

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

Quality endpoints accept optional `use_llm: true` to request an OpenAI quality summary. Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` to enable it; without configuration, or if the request fails, the built-in summary is returned.
- `POST /api/drift/analyze` with `baseline` and `current` JSON arrays or dictionaries.  
- `POST /api/drift/analyze-csv` with `baseline_csv` and `current_csv` content.

## Agentic Monitoring Workflow

`POST /api/monitoring/analyze` orchestrates the existing deterministic monitoring functions through a tool registry. It accepts inline records (`data`, `dataset`, or `records`) or a single configured `source`; a `baseline` enables drift analysis automatically. Use `checks` to select `quality`, `drift`, `forecast`, and/or `explain`. Drift requires a baseline, while forecast and explain require `date_column` and `target_column`, respectively. The current dataset may be supplied as `current`.

```json
{
	"data": [{"age": 28, "income": 42000}, {"age": null, "income": 51000}],
	"baseline": [{"age": 31, "income": 39000}, {"age": 34, "income": 47000}],
	"expected_columns": ["age", "income"],
	"notify": false
}
```

The response includes the selected checks, executed tool trace, quality and drift results, and root-cause findings with evidence taken from those results. Notifications are sent only when `notify` is `true` and findings exist; provide `channels` to choose recipients. `GET /api/monitoring/tools` lists the registered deterministic tools. The workflow planner uses explicit request fields and rules; it does not require an LLM, and `use_llm` only opts into the existing quality-summary feature.

## Enterprise Source Monitoring

`POST /api/sources/analyze` accepts a list of sources and returns a compact quality summary for each one. A request can mix local files and database or cloud sources:

```json
{
	"sources": [
		{"type": "csv", "path": "data/events.csv"},
		{"type": "excel", "path": "data/events.xlsx", "sheet_name": "Events"},
		{"type": "json", "path": "data/events.json"},
		{"type": "sql", "url": "postgresql+psycopg2://user:password@host/db", "query": "SELECT * FROM events LIMIT 1000"},
		{"type": "mongodb", "uri": "mongodb://localhost:27017", "database": "monitoring", "collection": "events"},
		{"type": "s3", "bucket": "monitoring-data", "key": "events.csv"},
		{"type": "dropbox", "path": "/monitoring/events.csv"},
		{"type": "google_drive", "file_id": "drive-file-id", "file_type": "csv"}
	]
}
```

Supported source types are `csv`, `excel`, `json`, `parquet`, `tsv`, `txt`, `sql`, `postgresql`, `mysql`, `mongodb`, `s3`, `dropbox`, and `google_drive`. SQL sources only accept `SELECT` queries. Configure cloud credentials through the provider SDK environment variables, including `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `DROPBOX_ACCESS_TOKEN`, and `GOOGLE_APPLICATION_CREDENTIALS`.

Use `GET /api/sources/capabilities` to check supported providers, missing SDKs, and required credential configuration before submitting a multi-source analysis. Provider SDK imports remain optional at startup; install the relevant entries from `requirements.txt` when needed. AWS credentials are resolved through the normal SDK chain, including environment variables, profiles, and execution roles.

## Explainability and Model Evaluation

The XAI dashboard and `POST /api/analytics/explain` report evaluate a held-out test split in addition to global feature importance. Classification reports weighted precision, recall, F1, accuracy, and a confusion matrix; regression reports MAE, RMSE, and R². The dashboard supports selecting a holdout share from 10% to 50%, reviewing actual-versus-predicted rows, and inspecting per-row SHAP contributions when the optional SHAP calculation is available.

Forecasting supports linear and ARIMA models, with Prophet and TensorFlow-backed LSTM options. LSTM forecasts require at least 10 time periods; if its runtime is unavailable or training fails, the response identifies the reason and uses the linear fallback. Near-duplicate checks use exact candidate comparisons for smaller datasets and MinHash/LSH candidate generation with exact similarity verification for larger datasets; large-input results include candidate-generation and truncation metadata.

## AI-Powered Features

The project now includes AI-style quality assessment capabilities that summarize dataset health, highlight top issues, and recommend follow-up actions based on missing values, duplicates, and risk level.

## Monitoring Notifications

Send a monitoring alert through email, SMS, WhatsApp, Slack, or Microsoft Teams:

```json
POST /api/notifications/send
{
	"subject": "Data drift alert",
	"message": "Customer age drift exceeded the threshold.",
	"dry_run": false,
	"channels": [
		{"channel": "email", "recipient": "alerts@example.com"},
		{"channel": "sms", "recipient": "+15550000000"},
		{"channel": "whatsapp", "recipient": "+15550000000"},
		{"channel": "slack", "recipient": "https://hooks.slack.com/services/..."},
		{"channel": "teams", "recipient": "https://outlook.office.com/webhook/..."}
	]
}
```

Use `dry_run: true` to validate the payload without contacting providers. Configure secrets with environment variables: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `ALERT_FROM_EMAIL`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_SMS_FROM`, `TWILIO_WHATSAPP_FROM`, `SLACK_WEBHOOK_URL`, and `TEAMS_WEBHOOK_URL`. Slack and Teams recipients may be omitted when their webhook URL is configured.


<img width="1920" height="965" alt="Screenshot 2026-03-26 192137" src="https://github.com/user-attachments/assets/88d54c28-8602-45f2-9e7c-251667701fca" />
