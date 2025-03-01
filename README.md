# Customer Churn Analysis - Python project

## Project Overview
This project aims to analyze customer churn using a dataset from a telecommunications company. The analysis was performed using a Jupyter Notebook environment using Python programming language. The primary goal is to identify patterns and factors that contribute to customer churn, enabling the company to develop strategies to retain customers.

## Libraries Used

The following libraries were used in this project:

- NumPy: For performing linear algebra operations, which are essential for numerical computations.
- Pandas: For data processing, including reading CSV files, manipulating data frames, and performing exploratory data analysis.
- Seaborn: For creating informative and attractive statistical graphics.
- Matplotlib: For generating a wide variety of static, animated, and interactive plots.
- Missingno: For visualizing missing data to understand the completeness of the dataset.
- Plotly: For creating interactive visualizations that allow for dynamic data exploration.
- StandardScaler from sklearn.preprocessing: standardizes features by removing the mean and scaling to unit variance. This ensures that each feature contributes equally to the model, which is crucial for many machine learning algorithms sensitive to the scale of the input data.

## Steps Performed

### Data Loading and Initial Exploration:

- The dataset was loaded into a Pandas DataFrame.
- The structure of the dataset was examined, including its shape, column names, and basic statistics, to understand the data at a high level.

### Data Cleaning and Checking:

- The dataset was loaded into a Pandas DataFrame.
- The structure of the dataset was examined, including its shape, column names, and basic statistics, to understand the data at a high level.

### Data Analysis:

##### Churn Distribution:

- The proportion of churn and non-churn customers was analyzed to understand the overall churn rate within the dataset.

##### Gender Distribution:

- The distribution of gender among customers was examined, revealing that gender does not significantly impact churn, as the proportions are nearly equal.

##### Senior Citizens:

- The percentage of senior citizens was visualized, showing that 83.8% of the customers are non-senior citizens.
- The distribution of senior citizens among churned and non-churned customers was also analyzed.

##### Tenure Distribution:

- The distribution of customer tenure was analyzed to understand how long customers typically stay with the company before churning.

##### Internet Service:

- The distribution of different internet services (Fiber optic, DSL, No internet service) among churned and non-churned customers was examined.
- It was found that customers with DSL are more likely to remain, whereas those with Fiber optic are more prone to churn.

##### Contract Types:

- The distribution of different contract types (Month-to-month, One year, Two years) among churned and non-churned customers was analyzed to understand the impact of contract length on churn.

##### Phone Service:

- The distribution of phone service among customers was examined, revealing that a higher number of customers who churned had an active phone service.

### Visualizations

- Bar Plots: Used to visualize the distribution of churn, gender, internet service, contract types, and phone service, providing clear comparisons between different categories.
- Pie Chart: Used to show the percentage of senior citizens, offering a quick visual representation of the age demographics.
- Histograms: Used to analyze the distribution of senior citizens and tenure, helping to identify patterns and trends in the data.
- Scatter Plot: Used to visualize the tenure distribution, highlighting the relationship between tenure and churn.
- Interactive Plots: Created using Plotly for a more dynamic exploration of the data, allowing for interactive analysis and deeper insights.

### Data Sources
- Kaggle Dataset: https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn
