# HeartDisease_EDA-StatisticalAnalysis_R

Statistical Methods for Data Science Course

**Topic: Statistical Methods for Heart Disease & Indicators**

## Abstract
In the healthcare industry, understanding what factors or indicators affect a disease is an essential
part of the decision-making and problem-solving process. These indicators allow
decision-makers to identify any potential ways to reduce risk factors of future health and increase
the likelihood of disease prevention effectively (Santos et al., 2019). Likewise, RStudio is used
to assist with statistical analyses by extracting important insights using computational statistics,
machine learning, and visualizations. Hence, this project aims to improve the process of
analyzing patients’ heart disease in the healthcare industry to allow earlier detection and
avoidance of heart disease and morbidity. The objective is to create a statistical model that
classifies patients into those who will develop heart disease in the future and those who will not
based on the relevance of data variables. Essentially, the insights gained by analyzing the
statistical inference of each data variable to the target data will aid in establishing which factor or
indicator is critical in causing heart disease.

**Keywords:** Data Science, Statistical & Inferential Analysis, Exploratory Data Analysis (EDA), Regression Analysis, Heart Disease.

## Table of Content
- Problem Statement & Dataset
- Basic Data Exploration & Pre-processing
- Exploratory Data Analysis (EDA)
- Data Cleaning & Preparation
- Data Modeling: Binomial Logistic Regression
  - Statistical Modeling & Assessments
  - Regression Assumption Diagnostics

## Problem Statement & Dataset

<p align="center">
    <img width="850" src="()">
    <br>
    Figure (). ().
</p>


## Basic Data Exploration & Pre-processing

<p align="center">
    <img width="850" src="()">
    <br>
    Figure (). ().
</p>

<p align="center">
    <img width="850" src="()">
    <br>
    Figure (). ().
</p>


<p align="center">
    <img width="850" src="()">
    <br>
    Figure (). ().
</p>



## Exploratory Data Analysis (EDA)
### Univariate Analysis
<p align="center">
    <img width="850" src="https://user-images.githubusercontent.com/90960615/190369809-f110a7ad-de1e-4831-85a4-8ab0da19edf9.png">
    <br>
    Figure (). Descriptive Statistics and Target (Response) Variable.
</p>

- Figure () shows the descriptive statistics of the heart disease dataset variables. 
- Most of the variables appear to be appropriate, with the exception of "RestingBP" and "Cholesterol", which have minimum values of 0 and are outliers since it is impossible to have a value of 0 for cholesterol and resting blood pressure. 
- The bar chart also illustrates that the target or response variable (HeartDisease) is balanced and that there is no need for data resampling.

### Bivariate Analysis
<p align="center">
    <img width="850" src="https://user-images.githubusercontent.com/90960615/190375275-a4f43878-8a84-458b-9be9-e5328c2787f4.png">
    <br>
    Figure (). Bivariate Analysis between Binary, Continuous, and Categorical Variables.
</p>

- Figure () depicts the bivariate analysis of the dataset's most significant data. 
- A binary (Sex), continuous (Age), and categorical (ChestPainType) variables are coupled with the response variable (HeartDisease). Male patients are more likely to have heart disease, the patients’ age are normally distributed, and patients with exercise angina are more likely to have heart disease.
- Patients with exercise angina are also more likely to experience ASY chest pain.



## Data Cleaning & Preparation
<p align="center">
    <img width="850" src="()">
    <br>
    Figure (). ().
</p>



<p align="center">
    <img width="850" src="()">
    <br>
    Figure (). ().
</p>



<p align="center">
    <img width="850" src="()">
    <br>
    Figure (). ().
</p>



## Data Modeling: Binomial Logistic Regression
Based on the objective and dataset, a binomial logistic regression would be best suited for regression analysis. Binomial logistic regression is a linear model where the data to be modeled are expressed as numerical, discrete categories, for instance, refers to whether a patient has heart disease or not as the response variable. The following are the proposed statistical modelling technique methods and procedures:

- Select binomial distribution for probability distribution, as the response variable indicates whether a patient will have heart disease or not based on
demographic and health indicators.
- Fit and interpret the model estimates using the training set with a binomial and logit family link.
- Evaluate the model fit by including or excluding variables based on the corresponding p-values of the fitted model coefficients as well as the p-values, deviance, and residuals deviance from the Analysis of Variance (ANOVA) test.
- Perform hypothesis testing using Wald statistic (Chi-Square statistic) to measure the overall model significance.
- Assess and evaluate the model using confusion matrix, ROC curve, and AUC value on the testing set.
- Perform logistic regression assumption diagnostics to ensure that the assumptions are satisfied.

### Statistical Modeling & Assessments
#### Model Fitting & Interpretation
<p align="center">
    <img width="850" src="https://user-images.githubusercontent.com/90960615/190371165-125fd08e-6c21-4973-afee-342006a9ba87.png">
    <br>
    Figure (). Fitted Model with ANOVA Test.
</p>

- After data preparation, the data is split (70:30) into training and testing sets for bias-free
results.
- The training set is then fitted into a GLM model and assessed using the associated
coefficient p-values and ANOVA test. 
- According to Figure (), the fitted model contains several insignificant variables ("RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "ChestPainType TA", "RestingECG Normal", "RestingECG ST", and "ST Slope Up"). 
- Similarly, the ANOVA test result reveals that all of the variables stated, except "MaxHR," are insignificant to the fitted model, as determined by the low deviance residuals and the Pr(>Chi) of > .05, respectively.

<br>

<p align="center">
    <img width="850" src="https://user-images.githubusercontent.com/90960615/190373972-436e031e-b4a5-4abc-ba4e-dd4bfdd92b22.png">
    <br>
    Figure (). Updated Fitted Model with ANOVA Test.
</p>

- Figure () depicts the final fitted model after removing all of the previously indicated insignificant variables except "MaxHR" because the ANOVA test results indicate that it offers significance.
- However, even though "Oldpeak" is significant, it will be excluded from the model fit in this case because
  - The AIC only differs by around a value of 10 when comparing with or without the variable, which is insufficient to justify that it decreases model performance.
  - The variable violates the assumption of linearity of independent variables and log-odds due to its negative and zero data values.

<br>

**Interpretation**

<p align="center">
    <img width="850" src="https://user-images.githubusercontent.com/90960615/190377407-c6aaf957-b463-4f00-a801-838fd516c201.png">
    <img width="850" src="https://user-images.githubusercontent.com/90960615/190377019-8f99a92c-50b8-4b93-8e68-b94d881bfccf.png">
    <br>
    Figure (). Fitted Model Outcome, Results, and Interpretation.
</p>


#### Model Assessment & Evaluation






### Regression Assumption Diagnostics
#### Assumption 1: Appropriate Binary Outcome Type




#### Assumption 2: Sufficiently Large Sample Size of Dataset



#### Assumption 3: Linearity of Independent Variables and Log-odds



#### Assumption 4: Absence of Multicollinearity



#### Assumption 5: Absence of Strongly Influential Outliers



