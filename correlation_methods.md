# Tutorials on Correlation Methods

## Table of Contents
  - [Tutorial 1: Pearson Correlation Coefficient](#tutorial-1-pearson-correlation-coefficient)
  - [Tutorial 2: Spearman's Rank Correlation](#tutorial-2-spearmans-rank-correlation)
  - [Tutorial 3: Kendall's Tau](#tutorial-3-kendalls-tau)
  - [Tutorial 4: Point-Biserial Correlation](#tutorial-4-point-biserial-correlation)
  - [Tutorial 5: Cramer's V](#tutorial-5-cramers-v)
  - [Tutorial 6: Phi Coefficient](#tutorial-6-phi-coefficient)
  - [Tutorial 7: Distance Correlation](#tutorial-7-distance-correlation)
  - [Tutorial 8: Cross-correlation](#tutorial-8-cross-correlation)

## Tutorial 1: Pearson Correlation Coefficient

### Introduction:
The Pearson correlation coefficient measures the linear relationship between two continuous variables. It ranges from -1 to +1, where -1 indicates a perfect negative correlation, +1 indicates a perfect positive correlation, and 0 indicates no linear correlation.
### Data:
For this tutorial, we will use the famous "Iris" dataset, which can be obtained from the `seaborn` library. The "Iris" dataset contains measurements of different features of iris flowers.
### Python Code Example:
```python
import seaborn as sns
import plotly.graph_objects as go

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Numerical encoding for the 'species' column
species_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
iris['species_code'] = iris['species'].map(species_mapping)

# Calculate Pearson correlation coefficient
correlation_coefficient = iris[['sepal_length', 'sepal_width']].corr().iloc[0, 1]

# Scatter plot to visualize the relationship
fig = go.Figure()
fig.add_trace(go.Scatter(x=iris['sepal_length'], y=iris['sepal_width'], mode='markers', 
                         marker=dict(color=iris['species_code'], colorscale='Viridis'),
                         hovertext=iris['species']))

fig.update_layout(title=f"Pearson Correlation: {correlation_coefficient:.2f}",
                  xaxis_title='Sepal Length',
                  yaxis_title='Sepal Width')

fig.show()
```
### Use Cases:
- The Pearson correlation coefficient is suitable for determining the strength and direction of the linear relationship between two continuous variables.
- It is commonly used in fields like biology, finance, and social sciences to understand how variables change together.
### When Not to Use:
- Pearson correlation assumes that the relationship between variables is linear. If the relationship is non-linear, it may not provide an accurate measure of correlation.
- It requires normally distributed data, so it may not be appropriate for skewed or non-normally distributed data.
## Tutorial 2: Spearman's Rank Correlation

### Introduction:
Spearman's rank correlation is a non-parametric method that measures the strength and direction of the monotonic relationship between two variables. It is based on ranking the data instead of using the actual values.
### Data:
For this tutorial, we will use a synthetic dataset that exhibits a non-linear relationship.
### Python Code Example:
```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Create synthetic dataset
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = x + np.sin(x) + np.random.normal(0, 1, 100)

# Convert to DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Calculate Spearman's rank correlation
correlation_coefficient = df.corr(method='spearman').loc['x', 'y']

# Scatter plot to visualize the relationship
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color='blue'),
                         hovertext=df['x']))

fig.update_layout(title=f"Spearman's Rank Correlation: {correlation_coefficient:.2f}",
                  xaxis_title='X',
                  yaxis_title='Y')

fig.show()
```
### Use Cases:
- Use Spearman's rank correlation when dealing with non-normally distributed data or when the relationship between variables is not strictly linear.
- It is commonly used in the analysis of rankings or ordinal data.
### When Not to Use:
- Spearman's rank correlation may not be the best choice for data with many tied values, as it may result in an underestimation of the correlation coefficient.
- It may not capture certain complex relationships, as it only considers the monotonicity of the relationship.
## Tutorial 3: Kendall's Tau

### Introduction:
Kendall's Tau is a non-parametric measure of correlation that assesses the strength and direction of the ordinal association between two variables. It is based on comparing the number of concordant and discordant pairs of data to calculate the correlation coefficient.
### Data:
For this tutorial, we will use a synthetic dataset that exhibits a non-linear relationship.
### Python Code Example:
```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Create synthetic dataset
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 3 * np.sin(x) + np.random.normal(0, 1, 100)

# Convert to DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Calculate Kendall's Tau correlation
correlation_coefficient = df.corr(method='kendall').loc['x', 'y']

# Scatter plot to visualize the relationship
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color='red'),
                         hovertext=df['x']))

fig.update_layout(title=f"Kendall's Tau Correlation: {correlation_coefficient:.2f}",
                  xaxis_title='X',
                  yaxis_title='Y')

fig.show()
```
### Use Cases:
- Kendall's Tau is appropriate for ordinal data or when the relationship between variables is non-linear.
- It is commonly used in ranking and preference analysis.
### When Not to Use:
- Kendall's Tau may not be suitable for small sample sizes, as it requires a sufficient number of tied values to provide accurate results.
- It may not be the best choice for continuous data with many tied values, as it can result in the loss of information.
## Tutorial 4: Point-Biserial Correlation

### Introduction:
Point-Biserial Correlation determines the correlation between a continuous variable and a dichotomous (binary) variable. It is used when one variable is continuous, and the other variable has two categories.
### Data:
For this tutorial, we will use the "Titanic" dataset, which contains information about passengers on the Titanic, including whether they survived (binary) and their age (continuous).
### Python Code Example:
```python
import seaborn as sns
import plotly.graph_objects as go

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Drop missing age values
titanic.dropna(subset=['age'], inplace=True)

# Calculate Point-Biserial Correlation
correlation_coefficient = titanic['age'].corr(titanic['survived'], method='pearson')

# Bar plot to visualize the relationship
fig = go.Figure()
fig.add_trace(go.Bar(x=titanic['survived'], y=titanic['age'], 
                     marker=dict(color=titanic['survived'], colorscale='Viridis'),
                     hovertext=titanic['survived']))

fig.update_layout(title=f"Point-Biserial Correlation: {correlation_coefficient:.2f}",
                  xaxis_title='Survived',
                  yaxis_title='Age')

fig.show()
```
### Use Cases:
- Point-Biserial Correlation is useful when comparing how a continuous variable varies between two distinct groups.
- It is commonly used in medical research to determine the correlation between a binary outcome (e.g., disease status) and a continuous predictor (e.g., age).
### When Not to Use:
- Point-Biserial Correlation should not be used when both variables are continuous or when the binary variable has more than two categories.
- It assumes that the continuous variable follows a normal distribution, so it may not be suitable for skewed or non-normally distributed data.
## Tutorial 5: Cramer's V

### Introduction:
Cramer's V is a measure of correlation between two categorical variables. It is an extension of the chi-square test and is used for cross-tabulated data with more

 than two categories per variable.
### Data:
For this tutorial, we will use the "Titanic" dataset, which contains information about passengers on the Titanic, including their class (Pclass) and survival status (Survived).
### Python Code Example:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Create a cross-tabulation
cross_tab = pd.crosstab(titanic['pclass'], titanic['survived'])

# Calculate Cramer's V correlation
def cramers_v(correlation_matrix):
    chi2 = chi2_contingency(correlation_matrix)[0]
    n = correlation_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = correlation_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

correlation_coefficient = cramers_v(cross_tab)

# Heatmap to visualize the relationship between the categorical variables
fig = go.Figure(data=go.Heatmap(z=cross_tab.values, x=cross_tab.columns, y=cross_tab.index,
                                colorscale='Viridis'))

fig.update_layout(title=f"Cramer's V Correlation: {correlation_coefficient:.2f}",
                  xaxis_title='Survived',
                  yaxis_title='Pclass')

fig.show()
```
### Use Cases:
- Cramer's V is useful for determining the strength of association between two categorical variables with more than two categories each.
- It is commonly used in social sciences and market research to analyze survey responses and assess the relationship between multiple categorical variables.
### When Not to Use:
- Cramer's V should not be used for two binary variables (2x2 contingency table) as it reduces to the Phi Coefficient in such cases.
- It may not be appropriate when dealing with very small sample sizes or sparse data, as it can lead to unstable results.
## Tutorial 6: Phi Coefficient

### Introduction:
The Phi coefficient is used to determine the correlation between two dichotomous variables (two categories each). It is based on the concept of the chi-square test.
### Data:
For this tutorial, we will use a synthetic dataset that consists of two binary variables.
### Python Code Example:
```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Create synthetic binary dataset
np.random.seed(42)
data_size = 100
x = np.random.choice([0, 1], size=data_size)
y = np.random.choice([0, 1], size=data_size)

# Convert to DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Calculate Phi Coefficient correlation
def phi_coefficient(x, y):
    confusion_matrix = pd.crosstab(x, y)
    n = confusion_matrix.sum().sum()
    a = confusion_matrix.iloc[0, 0]
    d = confusion_matrix.iloc[1, 1]
    b = confusion_matrix.iloc[0, 1]
    c = confusion_matrix.iloc[1, 0]
    return (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

correlation_coefficient = phi_coefficient(df['x'], df['y'])

# Heatmap to visualize the relationship between the two binary variables
confusion_matrix = pd.crosstab(df['x'], df['y'])
fig = go.Figure(data=go.Heatmap(z=confusion_matrix.values, x=confusion_matrix.columns, y=confusion_matrix.index,
                                colorscale='Viridis'))

fig.update_layout(title=f"Phi Coefficient Correlation: {correlation_coefficient:.2f}",
                  xaxis_title='Y',
                  yaxis_title='X')

fig.show()

```
### Use Cases:
- The Phi Coefficient is useful for determining the strength of association between two binary variables (yes/no, true/false).
- It is commonly used in contingency table analysis and in studying the relationship between two binary outcomes.
### When Not to Use:
- Phi Coefficient should not be used for categorical variables with more than two categories, as it is specifically designed for dichotomous variables.
- It may not be appropriate when dealing with very small sample sizes, as it can lead to unreliable estimates.
## Tutorial 7: Distance Correlation

### Introduction:
Distance correlation is a measure of correlation that is not restricted to linear relationships. It is based on the idea of comparing the distances between points in high-dimensional spaces.
### Data:
For this tutorial, we will use a synthetic dataset that exhibits a non-linear relationship.
### Python Code Example:
```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform

# Create synthetic dataset
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 3 * np.sin(x) + np.random.normal(0, 1, 100)

# Convert to DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Calculate Distance Correlation
def distance_correlation(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    dcov = np.sqrt(np.sum(np.outer(x, x) * np.outer(y, y))) / len(x)
    x_dist = squareform(pdist(x[:, np.newaxis]))
    y_dist = squareform(pdist(y[:, np.newaxis]))
    dcor = np.sum(squareform(pdist(x_dist * y_dist))) / (len(x) ** 2)
    return dcor / dcov

correlation_coefficient = distance_correlation(df['x'].to_numpy(), df['y'].to_numpy())

# Scatter plot to visualize the relationship
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', marker=dict(color='green'),
                         hovertext=df['x']))

fig.update_layout(title=f"Distance Correlation: {correlation_coefficient:.2f}",
                  xaxis_title='X',
                  yaxis_title='Y')

fig.show()
```
### Use Cases:
- Distance correlation is useful when looking for non-linear relationships between variables.
- It is commonly used in bioinformatics, image analysis, and nonlinear dimensionality reduction.
### When Not to Use:
- Distance correlation may not be appropriate for very large datasets, as it requires calculating pairwise distances between data points, which can be computationally expensive.
- It is not well-suited for small sample sizes

, as it may lead to unreliable estimates.
## Tutorial 8: Cross-correlation

### Introduction:
Cross-correlation is used to determine the similarity between two time series or signals at different time lags. It is commonly used in signal processing and time series analysis.
### Data:
For this tutorial, we will use synthetic time series data with a known time lag.
### Python Code Example:
```python
import numpy as np
import plotly.graph_objects as go

# Create synthetic time series data with a known time lag
np.random.seed(42)
time = np.arange(0, 10, 0.1)
signal_1 = np.sin(time)
signal_2 = np.sin(time + 2)

# Convert to DataFrame
df = pd.DataFrame({'signal_1': signal_1, 'signal_2': signal_2})

# Calculate Cross-correlation with a maximum lag of 5
max_lag = 5

def cross_correlation(x, y, max_lag):
    cross_corr = [np.correlate(x, np.roll(y, lag)) for lag in range(-max_lag, max_lag+1)]
    return np.array(cross_corr).flatten()

cross_corr_values = cross_correlation(df['signal_1'], df['signal_2'], max_lag)
lags = list(range(-max_lag, max_lag+1))  # Convert 'lags' range into a list

# Plot Cross-correlation values
fig = go.Figure()
fig.add_trace(go.Scatter(x=lags, y=cross_corr_values, mode='markers+lines'))

fig.update_layout(title='Cross-correlation between Signal 1 and Signal 2',
                  xaxis_title='Lag',
                  yaxis_title='Cross-correlation')

fig.show()
```
### Use Cases:
- Cross-correlation is useful in signal processing to find similarities or time delays between two signals, such as audio or sensor data.
- It is commonly used in time series analysis for lag detection and alignment of time series data.
### When Not to Use:
- Cross-correlation assumes a linear relationship between the time series at different lags. If the relationship is non-linear, other methods may be more suitable.
- It may not be appropriate for very long time series data due to computational complexity.

---