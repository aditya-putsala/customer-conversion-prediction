# Customer Conversion Prediction

Predicting which customers will subscribe to a term insurance plan after a telephonic marketing call, using six classification models: KNN, Decision Tree, Random Forest, XGBoost, Naive Bayes and Logistic Regression.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Results at a Glance](#results-at-a-glance)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modelling](#modelling)
- [Model Results](#model-results)
- [Key Takeaways](#key-takeaways)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)

---

## Problem Statement

A new-age insurance company uses several outreach plans to sell term insurance. Telephonic marketing campaigns remain one of the most effective ways to reach people, but they cost a lot. The company therefore wants to identify the customers most likely to convert, so that calls can be targeted at them. Using the historical marketing data, the task is to build a machine learning model that predicts whether a client will subscribe to the insurance.

This is a **supervised, binary classification** problem. The target column is `y` (`yes` = subscribed, `no` = did not subscribe).

---

## Results at a Glance

| Item | Value |
|------|-------|
| Customers / columns | 45,211 / 11 |
| Converted (`yes`) | 5,289 (11.7%) |
| Did not convert (`no`) | 39,922 (88.3%) |
| Missing values | None |
| Highest AUROC | **Logistic Regression: 0.855** (calculated from predicted probabilities) |
| Highest accuracy | Random Forest and XGBoost: 0.899 |
| Highest conversion rates | Students (28.7%) and retired customers (22.8%) |

---

## Dataset

`Customer_Conversion_Prediction.csv` has **45,211 rows and 11 columns**, one row per customer contact.

| Column | Description | Values in the data |
|--------|-------------|--------------------|
| `age` | Age of the customer | 18 to 95 (mean 40.9) |
| `job` | Customer's occupation | blue-collar 9,732; management 9,458; technician 7,597; admin. 5,171; services 4,154; retired 2,264; self-employed 1,579; entrepreneur 1,487; unemployed 1,303; housemaid 1,240; student 938; unknown 288 |
| `marital` | Marital status | married 27,214; single 12,790; divorced 5,207 |
| `education_qual` | Education level | secondary 23,202; tertiary 13,301; primary 6,851; unknown 1,857 |
| `call_type` | How the customer was contacted | cellular 29,285; unknown 13,020; telephone 2,906 |
| `day` | Day of the month | 1 to 31 |
| `mon` | Month | may 13,766; jul 6,895; aug 6,247; jun 5,341; nov 3,970; apr 2,932; feb 2,649; jan 1,403; oct 738; sep 579; mar 477; dec 214 |
| `dur` | Call duration in seconds | 0 to 4,918 (mean 258) |
| `num_calls` | Number of calls made | 1 to 63 (mean 2.8) |
| `prev_outcome` | Outcome of the previous contact | unknown 36,959; failure 4,901; other 1,840; success 1,511 |
| `y` | **Target:** did the customer subscribe? | no 39,922; yes 5,289 |

---

## Data Preparation

- **Missing values:** none in any column.
- **`prev_outcome`:** the `other` category was merged into `unknown`, leaving three values (success, failure, unknown).
- **Encoding:**

| Column | Encoding |
|--------|----------|
| `job` | One-hot encoded into 12 columns |
| `marital` | single = 0, married = 1, divorced = 2 |
| `education_qual` | unknown = 0, primary = 1, secondary = 2, tertiary = 3 |
| `call_type` | telephone = 0, unknown = 1, cellular = 2 |
| `prev_outcome` | success = 0, failure = 1, unknown = 2 |
| `mon` | jan = 0 through dec = 11 |
| `y` | no = 0, yes = 1 |

After encoding, the models use **21 features**: `age`, `marital`, `education_qual`, `call_type`, `day`, `mon`, `dur`, `num_calls`, `prev_outcome` and the 12 `job` columns.

---

## Exploratory Data Analysis

Only **11.7%** of customers converted, so the target is heavily unbalanced. The age boxplot also shows many outliers.

**Conversion rate by group** (share of customers in each group who said `yes`)

| Feature | Conversion rate by group |
|---------|--------------------------|
| `age` | 20 and under 34.0% (97 customers), 21 to 30 16.0%, 31 to 40 10.2%, 41 to 50 9.1%, 51 to 60 10.1%, over 60 42.3% (1,188 customers) |
| `job` | student 28.7%, retired 22.8%, unemployed 15.5%, management 13.8%, admin. 12.2%, self-employed 11.8%, unknown 11.8%, technician 11.1%, services 8.9%, housemaid 8.8%, entrepreneur 8.3%, blue-collar 7.3% |
| `marital` | single 14.9%, divorced 11.9%, married 10.1% |
| `education_qual` | primary 8.6%, secondary 10.6%, tertiary 15.0%, unknown 13.6% |
| `call_type` | cellular 14.9%, telephone 13.4%, unknown 4.1% |
| `day` (week of the month) | days 1 to 7 12.6%, days 8 to 14 13.2%, days 15 to 21 9.7%, days 22 to 31 12.0% |
| `mon` (quarter) | January to March 18.3%, April to June 9.3%, July to September 11.5%, October to December 16.8% |
| `num_calls` | 1 to 3 calls 12.9%, 4 to 6 calls 8.3%, 7 to 10 calls 6.1%, more than 10 calls 3.9% |
| `dur` (seconds) | up to 250 4.7%, 251 to 500 15.8%, 501 to 750 33.6%, 751 to 1,000 49.9%, over 1,000 59.7% |

**Key observations**
- **Age:** conversion drops from the 20s to the 50s and then rises sharply. Customers over 60 and customers aged 20 or under are the two groups most likely to buy.
- **Job:** students and retired customers convert most, while blue-collar workers and entrepreneurs convert least.
- **Education:** conversion rises with education level, from primary (8.6%) to tertiary (15.0%).
- **Call type:** cellular calls convert far better than calls where the type is unknown.
- **Timing:** the second week of the month is the best and the third week the weakest. The first and last quarters of the year do best, and the second quarter does worst.
- **Number of calls:** most customers (35,570) received 1 to 3 calls, and conversion falls as the number of calls grows. More than 10 calls rarely pays off.
- **Call duration:** conversion rises steadily with longer calls, from 4.7% for calls up to 250 seconds to 59.7% for calls over 1,000 seconds.

---

## Modelling

All models predict `y` from the 21 encoded features and were fitted on the unscaled data.

| Model | Settings | Test split | Test customers |
|-------|----------|------------|---------------:|
| KNN | `n_neighbors=6` | 20% (`random_state=42`) | 9,043 |
| Decision Tree | Default | 22% (`random_state=0`) | 9,947 |
| Random Forest | `n_estimators=100`, `criterion='entropy'` | 33% (`random_state=0`) | 14,920 |
| XGBoost | `learning_rate=0.5`, `n_estimators=100`, `random_state=50` | 33% (`random_state=0`) | 14,920 |
| Naive Bayes | Gaussian | 33% (`random_state=0`) | 14,920 |
| Logistic Regression | `max_iter=1000` | 33% (`random_state=0`) | 14,920 |

---

## Model Results

| Model | AUROC | Accuracy | Converters caught |
|-------|:-----:|:--------:|------------------:|
| KNN | 0.7719 | 0.8817* | 175 of 1,091 |
| Decision Tree | 0.6892 | 0.8670 | 534 of 1,169 |
| Random Forest | 0.6707 | 0.8989 | 654 of 1,757 |
| XGBoost | 0.7148 | 0.8989 | 833 of 1,757 |
| Naive Bayes | 0.6852 | 0.8287 | 874 of 1,757 |
| **Logistic Regression** | **0.8548** | 0.8933 | 446 of 1,757 |

*The KNN accuracy is calculated from its confusion matrix. Every other value is as printed in the notebook.

**Confusion matrices** (number of test customers)

| Model | Non-converters correctly predicted | Non-converters wrongly flagged as converters | Converters missed | Converters caught |
|-------|-----------------------------------:|---------------------------------------------:|------------------:|------------------:|
| KNN | 7,798 | 154 | 916 | 175 |
| Decision Tree | 8,090 | 688 | 635 | 534 |
| Random Forest | 12,757 | 406 | 1,103 | 654 |
| XGBoost | 12,578 | 585 | 924 | 833 |
| Naive Bayes | 11,490 | 1,673 | 883 | 874 |
| Logistic Regression | 12,882 | 281 | 1,311 | 446 |

**How to read these numbers**
- AUROC for KNN and Logistic Regression is calculated from predicted probabilities. For Decision Tree, Random Forest, XGBoost and Naive Bayes it is calculated from the predicted yes/no labels.
- The models were tested on different splits (20%, 22% and 33% of the data), so the scores are not like-for-like comparisons.
- About 88.3% of customers did not convert, so accuracy is best read together with the number of converters each model caught.

**Logistic Regression coefficients.** The largest positive coefficients are `student` (0.922), `call_type` (0.507), `retired` (0.387), `education_qual` (0.130) and `unemployed` (0.081). The largest negative coefficients are `prev_outcome` (-1.128), `blue-collar` (-0.754), `entrepreneur` (-0.724), `housemaid` (-0.663) and `services` (-0.577). The model was fitted on unscaled features, so coefficient sizes are not directly comparable across features measured on different scales.

---

## Key Takeaways

- The task is a binary classification problem, and Logistic Regression had the highest AUROC (0.855).
- Students and retired customers are the groups most likely to convert, followed by unemployed customers.
- Customers over 60 and customers aged 20 or under convert far more often than the 30 to 60 age range.
- Higher education, cellular calls and calls in the first and last quarters of the year go with higher conversion.
- Three calls or fewer give the best conversion, and conversion falls steadily as the number of calls rises.
- Longer calls are linked to much higher conversion.

---

## Getting Started

**1. Install the libraries**
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost jupyter
```

**2. Place the dataset next to the notebook.** The notebook loads the data with `pd.read_csv('Customer Conversion Prediction.csv')`, so make sure the file name in that cell matches your CSV.

**3. Open the notebook**
```bash
jupyter notebook Customer_Conversion_Prediction.ipynb
```

---

## Repository Structure

```
.
├── Customer_Conversion_Prediction.ipynb   # Cleaning, EDA, encoding and model comparison
├── Customer_Conversion_Prediction.csv     # 45,211 customers x 11 columns
└── README.md
```

---

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn (KNN, Decision Tree, Random Forest, Naive Bayes, Logistic Regression, metrics), XGBoost, Jupyter Notebook.
