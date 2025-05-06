# Maximizing-performance-and-data-analytics-of-f1-drivers
# 🏎️ F1 Driver Performance Prediction Using Machine Learning

This project uses data analytics and machine learning to model, analyze, and predict the performance of Formula 1 (F1) drivers based on historical race data. It focuses on identifying key performance indicators and building regression models to forecast race wins and classify driver success.

## 📊 Project Description

The goal of this project is to:
- Analyze the relationship between key driver statistics (pole positions, years active, etc.)
- Visualize trends and correlations using plots and heatmaps
- Build predictive models (Linear Regression, Random Forest, SGD Classifier)
- Identify performance patterns among drivers

The project uses Python, Pandas, Matplotlib, Seaborn, and Scikit-learn.

---

## 🚀 Features

- 📈 Correlation heatmap to explore relationships between variables
- 📊 Pie chart for driver nationalities
- 📉 Linear regression for `Race Wins` vs `Pole Positions` and `Years Active`
- 🌲 Random Forest Regressor and SGD Classifier for performance modeling
- 📦 Data preprocessing and feature selection
- 🧠 Overfitting and underfitting checks

---

## 📂 Dataset

The dataset includes driver information such as:
- Name
- Nationality
- Pole Positions
- Race Wins
- Years Active
- Championships

(Ensure your dataset is named `df` and cleaned to contain only numeric data for modeling.)

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/f1-driver-performance.git
   cd f1-driver-performance
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 How It Works

### Data Exploration
```python
df.info()
df.describe()
df.dtypes
```

### Correlation Heatmap
```python
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap="YlGnBu")
```

### Regression Modeling
```python
X = df[['Pole_Positions']].values
y = df['Race_Wins'].values
model = LinearRegression()
model.fit(X, y)
```

### Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, max_depth=5)
reg.fit(X_train, y_train)
```

### Classification with SGD
```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log_loss')
clf.fit(X_train, y_train_class)
```

---

## 📉 Evaluation Metrics

- **R² Score**
- **MAE / RMSE**
- **Correlation Coefficient**

Used to determine how well the regression/classification models perform.

---

## 📌 Future Improvements

- Add hyperparameter tuning (GridSearchCV)
- Build a web dashboard using Streamlit or Flask
- Extend model to include team performance, circuits, and weather data


## 🙋‍♂️ Author

**Your Name**  
LinkedIn: [Vidyansh Sinha](www.linkedin.com/in/vidyansh-sinha-515240348)  
GitHub: [@ishuaa](https://github.com/ishuaaa)
