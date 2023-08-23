# Statistical Significance and Margin of Error Handbook

## Table of Contents
  - [1. Introduction](#1-introduction)
  - [2. Basics of Hypothesis Testing](#2-basics-of-hypothesis-testing)
  - [3. Statistical Significance](#3-statistical-significance)
  - [4. Confidence Intervals](#4-confidence-intervals)
  - [5. Margin of Error (MoE)](#5-margin-of-error-moe)
  - [6. Advanced Concepts](#6-advanced-concepts)
  - [7. Practical Applications and Case Studies](#7-practical-applications-and-case-studies)
  - [8. Tools and Software](#8-tools-and-software)
  - [9. Conclusion and Further Reading](#9-conclusion-and-further-reading)
  - [10. Appendices](#10-appendices)

## 1. Introduction
The language of data, statistics, governs every research decision, from medical advancements to market research. This handbook aims to comprehensively unpack two foundational statistical concepts: **statistical significance** and **margin of error**. Their grasp is indispensable in inferential statistics, guiding professionals in making informed inferences about populations based on sample data.

---
## 2. Basics of Hypothesis Testing

### **Null and Alternative Hypotheses:**
- **Null Hypothesis ($H_0$)**: Suggests no effect or difference. It forms the foundation of any test of significance.
- **Alternative Hypothesis ($H_a$ or $H_1$)**: Proposes the presence of an effect or difference.

_Example_: For testing a drug's efficacy:
- $H_0$: The drug doesn't affect the ailment.
- $H_1$: The drug has an impact on the ailment.
### **Type I and Type II Errors:**
Mistakes happen, even in statistics. These errors represent the pitfalls of hypothesis testing:
- **Type I Error (False Positive)**: Rejecting a true $H_0$.
- **Type II Error (False Negative)**: Not rejecting a false $H_0$.
### **Understanding p-value:**
- **Definition**: The p-value measures the evidence against a null hypothesis. It calculates the probability of obtaining results as extreme (or more) than observed if $H_0$ were true.

In Python, we can perform a t-test to determine p-value:
```python
import numpy as np
from scipy.stats import ttest_ind

# Generate data: Heights of two groups of people
group_a = np.random.normal(170, 10, 100)
group_b = np.random.normal(173, 10, 100)

# Two-sample t-test
t_stat, p_val = ttest_ind(group_a, group_b)

print(f"p-value = {p_val:.4f}")
```
### **Visualization with Plotly**:
Understanding distributions can be simplified using visual representations. Let's plot the generated data above:
```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Histogram(x=group_a, name='Group A', opacity=0.75))
fig.add_trace(go.Histogram(x=group_b, name='Group B', opacity=0.75))

fig.update_layout(barmode='overlay', title=f"p-value: {p_val:.4f}")
fig.show()
```

![plot1](figures/p_value.png)

This interactive histogram allows one to see the distributions of the two groups and how they overlap.

---
## 3. Statistical Significance
The golden standard in most research fields for statistical significance is a p-value of less than 0.05. However, it's essential to understand the context:

- **Statistical vs. Practical Significance**: A finding might be statistically significant but may not have practical implications.
### Methods of Determining Statistical Significance:
1. **Traditional methods**: 
   - **T-tests**: Used for comparing means.
   - **Chi-square tests**: Used for categorical data.
2. **Modern methods**:
   - **Bootstrap resampling**: Uses repeated sampling to determine significance.
   - **Permutation tests**: Randomly reallocates data to determine null distribution.

| Method            | Strengths                                     | Weaknesses                               | Use Cases                                  |
|-------------------|-----------------------------------------------|------------------------------------------|--------------------------------------------|
| T-test            | Simple, widely accepted                       | Assumes normality                        | Comparing two group means                  |
| Chi-square        | Useful for categorical data                   | Requires sufficiently large sample size | Association between two categorical variables |
| Bootstrap         | Doesn't assume underlying distribution        | Computationally intensive                | When data isn't normally distributed       |
| Permutation tests | Doesn't assume random sampling                | Computationally intensive                | Testing randomness or chance               |

Absolutely! Let's continue the handbook.

---
## 4. Confidence Intervals

### **Definition**
A confidence interval (CI) offers a range within which the true population parameter is expected to lie with a certain level of confidence.

_Example_: If we state that the average height of men in New Zealand is 173 cm ± 2 cm with a 95% confidence interval, we imply that we're 95% certain the actual average height lies between 171 and 175 cm.
### **Calculating Confidence Intervals**
- For **means**:
  
  The formula for the confidence interval for means is:

$$ \bar{X} \pm Z \left( \frac{\sigma}{\sqrt{n}} \right) $$ 

  Where:
  - $ \bar{X} $ = Sample mean
  - $ Z $ = Z-score (from Z-table based on desired confidence level)
  - $ \sigma $ = Population standard deviation
  - $ n $ = Sample size

- For **proportions**:
  
  The formula is:

  $ p \pm Z \sqrt{ \frac{p(1-p)}{n} } $

  Where:
  - $ p $ = Sample proportion
  - $ n $ = Sample size
### Confidence Intervals Visualization: Titanic and Tips Datasets

### Overview
Confidence intervals (CIs) offer a range within which a population parameter, such as the mean, is likely to fall. A 95% confidence interval means we are 95% confident that the true population mean lies within the interval.

The following visualizations demonstrate the confidence intervals for two different datasets using both histograms and kernel density estimation (KDE) plots.

---
### 1. Titanic Dataset: Fare Distribution with 95% Confidence Interval
The `titanic` dataset contains a variety of information about passengers on the Titanic, including fares they paid for their tickets. We will visualize the distribution of fares and overlay the 95% confidence interval for the mean fare.
```python
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats

# Load Titanic dataset
titanic = sns.load_dataset("titanic")
fares = titanic['fare'].dropna()

# Sample statistics
sample_mean = np.mean(fares)
sample_std = np.std(fares)
n = len(fares)

# Compute the z value for a 95% confidence interval
z_value = stats.norm.ppf(0.975)

# Calculate CI
margin_error = z_value * (sample_std / np.sqrt(n))
ci_lower = sample_mean - margin_error
ci_upper = sample_mean + margin_error

# Compute KDE
kde_x = np.linspace(fares.min(), fares.max(), 100)
kde_y = stats.gaussian_kde(fares)(kde_x)

# Visualization
fig = go.Figure()

# Add histogram for the fare data
fig.add_trace(go.Histogram(x=fares, histnorm='percent', name='Fares', nbinsx=50, opacity=0.5))

# Add KDE line
fig.add_trace(go.Scatter(x=kde_x, y=kde_y*100, mode='lines', name='Density', line=dict(color='blue', width=1.5)))

# Add vertical lines for the confidence interval
fig.add_shape(type="line", 
              x0=ci_lower, x1=ci_lower, y0=0, y1=10,  
              line=dict(color="LightBlue", width=2))
fig.add_shape(type="line", 
              x0=ci_upper, x1=ci_upper, y0=0, y1=10, 
              line=dict(color="LightBlue", width=2))

# Add a vertical line for the sample mean
fig.add_shape(type="line", 
              x0=sample_mean, x1=sample_mean, y0=0, y1=10, 
              line=dict(color="Red", width=2))

fig.update_layout(title="Distribution of Titanic Fares with 95% Confidence Interval",
                  xaxis_title="Fare",
                  yaxis_title="Percentage",
                  bargap=0.1)

fig.show()
```
![plot2](figures/confidence_interval_1.png)

#### Interpretation:
The blue bars represent the histogram of the fare values, showing the distribution of different fare amounts among the Titanic passengers. The KDE (in blue line) offers a smoothed representation of this distribution.

The red vertical line indicates the sample mean of the fares, while the light blue vertical lines show the 95% confidence interval around this mean. Given our sample data, we're 95% confident that the true average fare of all Titanic passengers falls between these two blue lines.
### 2. Tips Dataset: Total Bill Distribution with 95% Confidence Interval
The `tips` dataset provides information on tips received by a waiter over a period of time. We'll visualize the distribution of the total bills and highlight the 95% confidence interval for the mean bill.
```python
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats

# Load Tips dataset
tips = sns.load_dataset("tips")
bills = tips['total_bill']

# Sample statistics
sample_mean = np.mean(bills)
sample_std = np.std(bills)
n = len(bills)

# Compute the z value for a 95% confidence interval
z_value = stats.norm.ppf(0.975)

# Calculate CI
margin_error = z_value * (sample_std / np.sqrt(n))
ci_lower = sample_mean - margin_error
ci_upper = sample_mean + margin_error

# Compute KDE
kde_x = np.linspace(bills.min(), bills.max(), 100)
kde_y = stats.gaussian_kde(bills)(kde_x)

# Visualization
fig = go.Figure()

# Add histogram for the total bill data
fig.add_trace(go.Histogram(x=bills, histnorm='percent', name='Total Bill', nbinsx=30, opacity=0.5))

# Add KDE line
fig.add_trace(go.Scatter(x=kde_x, y=kde_y*100, mode='lines', name='Density', line=dict(color='blue', width=1.5)))

# Add vertical lines for the confidence interval
fig.add_shape(type="line", 
              x0=ci_lower, x1=ci_lower, y0=0, y1=10,  
              line=dict(color="LightBlue", width=2))
fig.add_shape(type="line", 
              x0=ci_upper, x1=ci_upper, y0=0, y1=10, 
              line=dict(color="LightBlue", width=2))

# Add a vertical line for the sample mean
fig.add_shape(type="line", 
              x0=sample_mean, x1=sample_mean, y0=0, y1=10, 
              line=dict(color="Red", width=2))

fig.update_layout(title="Distribution of Total Bills with 95% Confidence Interval",
                  xaxis_title="Total Bill ($)",
                  yaxis_title="Percentage",
                  bargap=0.1)

fig.show()
```
![plot3](figures/confidence_interval_2.png)

#### Interpretation:
The gray bars illustrate the histogram of the total bill amounts, giving us an idea of how much diners typically spent. The KDE (depicted as the blue line) gives a smoothed representation of the bill distribution.

Similar to the previous example, the red line marks the sample mean of the total bills. The two light blue lines encapsulate the 95% confidence interval for this mean. Given our sample, we can be 95% confident that the true average bill for all diners is between these two values.

---

By understanding the spread and center of data (through histograms and KDEs) and combining that with confidence intervals, we can make more informed conclusions about populations based on sample data.
---
## 5. Margin of Error (MoE)

### **Definition**
The Margin of Error represents the range within which we expect our survey results to differ from the actual population. It accounts for the inherent variability that arises from sampling.
### **Calculating MoE**
Using the previously mentioned formulas, the MoE can be extracted. For **means**, it's the term:

$$ Z \left( \frac{\sigma}{\sqrt{n}} \right) $$

And for **proportions**:

$$ Z \sqrt{ \frac{p(1-p)}{n} } $$
### **Factors Affecting MoE**
1. **Sample Size**: Larger samples generally lead to smaller MoE.
2. **Variability in Data**: More variability can increase the MoE.
3. **Confidence Level**: Higher confidence levels lead to a larger MoE.
### **Interpreting MoE**
Understanding MoE is critical to interpreting survey results. If a political candidate has a 52% support rate with a 3% MoE, their actual support could be anywhere from 49% to 55%.

---
## 6. Advanced Concepts

### **Effect Size**
While p-values indicate if an effect exists, effect size measures the magnitude. Common effect size metrics include Cohen's d for t-tests or Phi for chi-square tests.
### **Power Analysis**
Statistical power is the probability that a test correctly rejects the null hypothesis. Higher power reduces the risk of Type II errors. It depends on sample size, significance level, and effect size.
### **Multiple Comparisons Problem**
When performing multiple tests, the chance of encountering a Type I error increases. Corrections like Bonferroni or FDR can adjust significance levels.
### **Non-parametric Tests**
Used when data doesn't meet normal distribution assumptions. Examples include the Mann-Whitney U test or Kruskal-Wallis test.

---
## 7. Practical Applications and Case Studies

### **Case Study 1**: A/B Testing
Online stores often use A/B tests to compare two versions of a webpage. They'll measure metrics like conversion rate for each version and use statistical significance to determine which version is superior.
### **Case Study 2**: Clinical Trials
When new drugs are developed, they're tested against placebos. Statistical significance can determine whether a drug's effects are likely due to the drug itself rather than chance.

---
## 8. Tools and Software
For **Python**, libraries such as `statsmodels` and `scipy` offer vast resources for hypothesis testing. `plotly` and `seaborn` assist with visualizations.

For those interested in **R**, functions like `t.test()`, `chisq.test()`, or packages like `ggplot2` for plotting are invaluable.

---
## 9. Conclusion and Further Reading
This handbook offers a foundational understanding of statistical significance and margin of error. For deeper insights, resources such as [Statistical Inference](https://www.coursera.org/learn/statistical-inference) on Coursera or textbooks like "Statistics" by Robert S. Witte are recommended.

---
## 10. Appendices

### **Statistical Tables**: Accessible in most statistics textbooks or online resources.

### **Glossary**:
- **Null Hypothesis**: Claim of no effect.
- **p-value**: Probability of observing data given that the null hypothesis is true.
### **FAQs**:
1. **Why is 0.05 chosen as a significance level?** Historically established and widely accepted, though it's not a strict rule.

---

This handbook offers a thorough introduction to the topics at hand. Given the constraints, some sections are summarized, but they provide a solid foundation. Adjustments, inclusions, or deeper dives can be made based on feedback or specific requirements.