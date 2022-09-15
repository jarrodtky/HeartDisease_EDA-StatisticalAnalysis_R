# HeartDisease_EDA-StatisticalAnalysis_R

Statistical Methods for Data Science Course

**Topic: Statistical Methods for Heart Disease & Indicators**

## Abstract
In the healthcare industry, understanding what factors or indicators affect a disease is an essential part of the decision-making and problem-solving process. 
These indicators allow decision-makers to identify any potential ways to reduce risk factors of future health and increase the likelihood of disease prevention effectively (Santos et al., 2019). 
Likewise, RStudio is used to assist with statistical analyses by extracting important insights using computational statistics, machine learning, and visualizations. Hence, this project aims to improve the process of analyzing patients’ heart disease in the healthcare industry to allow earlier detection and avoidance of heart disease and morbidity. 
The objective is to create a statistical model that classifies patients into those who will develop heart disease in the future and those who will not based on the relevance of data variables. 
Essentially, the insights gained by analyzing the statistical inference of each data variable to the target data will aid in establishing which factor or indicator is critical in causing heart disease.

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
    <img width="500" src="()">
    <br>
    Figure (). ().
</p>


## Basic Data Exploration & Pre-processing

<p align="center">
    <img width="500" src="()">
    <br>
    Figure (). ().
</p>

<p align="center">
    <img width="500" src="()">
    <br>
    Figure (). ().
</p>


<p align="center">
    <img width="500" src="()">
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
    <img width="500" src="()">
    <br>
    Figure (). ().
</p>



<p align="center">
    <img width="500" src="()">
    <br>
    Figure (). ().
</p>



<p align="center">
    <img width="500" src="()">
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
    <img width="500" src="https://user-images.githubusercontent.com/90960615/190377407-c6aaf957-b463-4f00-a801-838fd516c201.png">
    <img width="500" src="https://user-images.githubusercontent.com/90960615/190377019-8f99a92c-50b8-4b93-8e68-b94d881bfccf.png">
    <br>
    Figure (). Fitted Model Outcome, Context & Results, and Interpretation.
</p>


#### Model Assessment & Evaluation
<p align="center">
    <img width="500" src="https://user-images.githubusercontent.com/90960615/190378489-5afd5c4c-8f93-4b0c-945f-f20fdbadfad3.png">
    <br>
    Figure (). Confusion Matrix and Classification Scores.
</p>

- According to Figure (), the confusion matrix is built with a threshold of 0.5, and the fitted model for the test set projected a classification accuracy of 84% and a misclassification error rate of 16%, both of which are acceptable. 
- The sensitivity and specificity findings are similarly 82% and 86%, respectively. 
  - This indicates that a significant proportion of patients have true heart disease and a small proportion have false heart disease
  - Whereas a large proportion do not have actual heart disease and a small proportion do not have false heart disease.

<br> 

<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190378608-56b1516b-bd81-489d-87df-d665052043c6.png">
    <br>
    Figure (). ROC Curve and AUC Value.
</p>

- Figure () depicts the fitted model for the test set, which uses the ROC curve to calculate the AUC value to assess model performance. 
- The ROC curve and AUC value determined by the fitted model for the test set is 0.84.
- This suggests that the model has an 84% chance of correctly classifying patients as positive (having heart disease) and negative (not having heart disease).


### Regression Assumption Diagnostics
#### Assumption 1: Appropriate Binary Outcome Type
<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190380141-fc2dd0f3-a905-4201-91e6-4f6d8aa2de64.png">
    <br>
    Figure (). Heart Disease (Target Variable) Bar Chart.
</p>

- Based on Figure (), results show that there are only two outcomes (i.e. binary classification of have heart disease or does not have heart disease). 
- Hence, the assumption is met.


#### Assumption 2: Sufficiently Large Sample Size of Dataset
<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190380261-beb26101-7bc2-4e81-b42c-ed13ae347161.png">
    <br>
    Figure (). Dataset Size and Variables.
</p>

- The dataset should ideally contain at least 500 rows of observations, with at least 10-20 instances of the least likely result for each predictor variable in the model. 
- According to Figure (), the training set has 641 rows of observations and 16 variables, which is an acceptable dataset size to work with; hence, the assumption is satisfied.


#### Assumption 3: Linearity of Independent Variables and Log-odds
<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190380309-26f764c8-fcf3-4b79-bee3-6abc576b6f4d.png">
    <br>
    Figure (). Box-Tidwell Test.
</p>

- The "Oldpeak" variable is removed from the model, as stated in the model fitting and interpretation section; since positive and negative values of "Oldpeak" have distinct meanings, and data manipulation will result in biased findings. 
- As a result, the Box-Tidwell test result in Figure () indicates that the variables "Age" and "MaxHR" are not significant based on Pr(>|z|) > 0.05. 
- The assumption is satisfied since the "Age" and "MaxHR" log-odds features are linear.


#### Assumption 4: Absence of Multicollinearity
<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190380350-d4d97f50-9877-4cad-8cda-357d38d2694e.png">
    <br>
    Figure (). Correlation Matrix.
</p>

- As a rule of thumb, one might suspect multicollinearity when the correlation between two (predictor) variables is below -0.9 or above +0.9. 
- Based on Figure (), it shows that 2 variables; “ST_Slope_Flat” and “ST_Slope_Up” have high correlation with each other based on the encoded dataset.

<br>

<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190380448-b4db98eb-2e7a-4292-a5e6-abea2d89eaea.png">
    <br>
    Figure (). Fitted Model VIF Values.
</p>

- After removing insignificant and highly correlated variables, observing the VIF values also helps to detect multicollinearity in the fitted model. 
- As a rule of thumb, a VIF exceeding 5 requires further investigation, whereas VIFs above 10 indicate extreme multicollinearity. 
- Based on Figure (), it shows that none of the variables exceed the VIF value of 5; hence, the assumption is met.


#### Assumption 5: Absence of Strongly Influential Outliers
<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190380518-325cb79b-15ba-4b3a-af13-c250ec252c03.png">
    <br>
    Figure (). Standardized Residuals Plot.
</p>

- Standardized residuals of the fitted model can be used to determine whether a data point is an outlier or not based on values greater than 3 representing possible extreme outliers.
- Based on Figure (), it shows that none of the data points of the fitted model consist of any outliers.

<br>

<p align="center">
    <img width="650" src="https://user-images.githubusercontent.com/90960615/190380576-02f06bad-9e03-4435-96aa-1d4867c837c5.png">
    <br>
    Figure (). Cook’s D Plot.
</p>

- Cook's Distance, which is determined using the residual and leverage of a data point, can also estimate its influence. 
- One common threshold is 4/N (where N is the number of observations), which means that observations with Cook's Distance greater than 4/N are considered influential.
- According to Figure (), only 9.2% (59/641) of the data points are in the outlier zone based on the pre-defined criterion (4/N), which is a insignificant number. 
- As a result, there is no need to address the few outliers, and the assumption is satisfied.



## Conclusion
- EDA provides fascinating insights between predictor variables and response/target variable, with data processing utilized to clean and better match the data to the model. 
- A few variables are omitted from the trained fitted model owing to an insignificant relationship to the response variable as assessed by the p-value of the coefficients and the ANOVA test, as well as variables with negative and zero data points to satisfy the linearity and log(odds) assumptions.
- As a result, the test set's fitted model is deemed good due to its low misclassification rate, high accuracy, sensitivity, specificity, and AUC value. 
  - This indicates that the model correctly classified the dataset's positive and negative classes (having or not having heart disease). 
- Finally, all assumptions are met, except the assumption of observation independence due to the necessity to transform to log(odds) and verifying it is out of scope as well; hence, the assumption is discarded.

