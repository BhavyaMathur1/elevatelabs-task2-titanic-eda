# 🚢 Task 2 — Exploratory Data Analysis (EDA) | Titanic Dataset

> **AI & ML Internship — Elevate Labs**
> Tools: `Pandas` · `Matplotlib` · `Seaborn`

---

## 📌 Objective

Understand the Titanic dataset through descriptive statistics and rich visualizations to uncover patterns, trends, and anomalies that influence passenger survival.

---

## 📁 Project Structure

```
titanic-eda/
├── titanic.csv          # Dataset (891 rows × 8 columns)
├── eda_titanic.py       # Main EDA script
├── plots/               # All generated visualizations
│   ├── 01_survival_overview.png
│   ├── 02_histograms.png
│   ├── 03_boxplots.png
│   ├── 04_correlation_matrix.png
│   ├── 05_pairplot.png
│   ├── 06_age_distribution.png
│   ├── 07_fare_skewness.png
│   ├── 08_embarkation.png
│   └── 09_missing_values.png
└── README.md
```

---

## 📊 Dataset Overview

| Feature    | Type        | Description                         |
|------------|-------------|-------------------------------------|
| `survived` | int (0/1)   | Target: 0 = No, 1 = Yes             |
| `pclass`   | int (1–3)   | Passenger class (1=First, 3=Third)  |
| `sex`      | categorical | male / female                       |
| `age`      | float       | Age in years (~20% missing)         |
| `sibsp`    | int         | # of siblings/spouses aboard        |
| `parch`    | int         | # of parents/children aboard        |
| `fare`     | float       | Ticket fare paid                    |
| `embarked` | categorical | Port: S=Southampton, C=Cherbourg, Q=Queenstown |

---

## 🧮 Summary Statistics

```
Shape : 891 rows × 8 columns
Missing values: age (181 = ~20%), embarked (2)

        mean     std    min    25%    50%    75%    max
age    29.59   13.85   0.42  19.57  29.99  39.42  74.10
fare   31.83   43.69   1.11   9.93  18.73  36.99 445.10
sibsp   0.52    1.04   0.00   0.00   0.00   1.00   8.00
parch   0.38    0.80   0.00   0.00   0.00   0.00   6.00
```

---

## 📈 Visualizations & Insights

### 1. Survival Overview
![Survival Overview](plots/01_survival_overview.png)
- Overall survival rate: **~38%**
- Females had a much higher survival rate than males ("women and children first")
- 1st class passengers had the highest survival rate

### 2. Histograms of Numeric Features
![Histograms](plots/02_histograms.png)
- **Age** is roughly bell-shaped, centered around ~30
- **Fare** is heavily right-skewed (most paid low fares, a few paid extremely high)
- **SibSp** and **Parch** are zero-inflated (most passengers traveled alone)

### 3. Boxplots — Age & Fare by Survival
![Boxplots](plots/03_boxplots.png)
- Survivors tended to pay **higher fares** (wealthier = better access to lifeboats)
- Age distribution is similar across groups, but **younger children** show slightly higher survival

### 4. Correlation Matrix
![Correlation](plots/04_correlation_matrix.png)
- `pclass` negatively correlates with survival — higher class → better survival chances
- `fare` positively correlates with survival — expensive tickets linked to 1st class
- Low multicollinearity among most features (good for modeling)

### 5. Pairplot
![Pairplot](plots/05_pairplot.png)
- Clear separation between survivors and non-survivors in fare vs pclass space
- Age alone is not a strong predictor, but combined with class/fare it adds value

### 6. Age Distribution by Class & Sex
![Age Dist](plots/06_age_distribution.png)
- 1st class passengers are generally **older** (wealthier adults)
- 3rd class has more **young adults and children**
- Male passengers slightly older on average than female passengers

### 7. Fare Skewness & Log Transformation
![Fare Skewness](plots/07_fare_skewness.png)
- Raw fare skewness: **4.53** — extreme right skew
- After log(Fare+1): skewness drops to **~0.27** — near-normal
- Log transformation is recommended before using fare in ML models

### 8. Embarkation Analysis
![Embarkation](plots/08_embarkation.png)
- ~72% of passengers boarded at **Southampton (S)**
- Cherbourg (C) passengers had the highest survival rate (more 1st class travelers)

### 9. Missing Values Heatmap
![Missing](plots/09_missing_values.png)
- **Age** has ~20% missing — recommend median imputation or model-based filling
- **Embarked** has 2 missing values — fill with mode ('S')

---

## 🔑 Key Insights

| # | Insight |
|---|---------|
| 1 | **Sex** is the strongest survival predictor — females survived at ~2× the male rate |
| 2 | **Pclass** matters — 1st class survival >> 3rd class |
| 3 | **Fare** is highly right-skewed and should be log-transformed for ML |
| 4 | **Age** has ~20% missing data — needs careful imputation |
| 5 | **Traveling alone** is common (SibSp=0, Parch=0 for most passengers) |
| 6 | Embarkation port acts as a proxy for socioeconomic status |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/titanic-eda.git
cd titanic-eda

# Install dependencies
pip install pandas matplotlib seaborn

# Run EDA
python eda_titanic.py
```

All plots will be saved in the `plots/` directory.

---

## 📚 Interview Q&A (from task sheet)

**1. What is the purpose of EDA?**
EDA helps understand data structure, detect missing values/outliers, uncover relationships between variables, and form hypotheses before building ML models.

**2. How do boxplots help in understanding a dataset?**
Boxplots show the median, quartiles, and outliers compactly. They make it easy to compare distributions across groups (e.g., fare by survival status).

**3. What is correlation and why is it useful?**
Correlation measures linear relationships between variables (−1 to +1). It helps identify which features are related to the target and which features are redundant (multicollinearity).

**4. How do you detect skewness in data?**
Compare mean vs median (if mean > median → right skew), compute `.skew()` in pandas, or plot a histogram. Fare had a skewness of 4.53 — heavily right-skewed.

**5. What is multicollinearity?**
When two or more features are highly correlated with each other (not just the target). It inflates variance in linear models. Detected via correlation matrix or VIF.

**6. What tools do you use for EDA?**
Pandas (statistics), Matplotlib & Seaborn (visualization), Plotly (interactive plots), NumPy (computation).

**7. Can you explain a time when EDA helped you find a problem?**
During this analysis, EDA revealed that `age` had ~20% missing values — a critical issue that would silently corrupt model predictions if ignored. EDA also revealed the extreme skewness of `fare`, prompting a log transformation before modeling.

**8. What is the role of visualization in ML?**
Visualization validates assumptions, reveals patterns invisible in raw numbers, communicates findings to non-technical stakeholders, and guides feature engineering decisions.

---

## 👨‍💻 Author

Made as part of the **Elevate Labs AI & ML Internship — Task 2**
