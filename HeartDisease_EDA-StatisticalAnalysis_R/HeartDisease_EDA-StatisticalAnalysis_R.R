
# Introduction.

# In the healthcare industry, understanding what factors or indicators affect a disease is an essential part of the decision-making and problem-solving process. 
# These indicators allow decision-makers to identify any potential ways to reduce risk factors of future health and increase the likelihood of disease prevention effectively (Santos et al., 2019). 
# Likewise, RStudio would assist statistical analytics by extracting important insights using computational statistics, machine learning, and visualizations. 
# Hence, this project aims to improve the process of analyzing patients' heart disease in the healthcare industry to allow earlier detection and avoidance of heart disease and morbidity. 
# The objective is to create a statistical model that classifies patients into those who will develop heart disease in the future and those who will not based on the relevance of data variables. 
# Essentially, the insights gained by analyzing the statistical inference of each data variable to the target data will aid in establishing which factor or indicator is critical in causing heart disease.





# Import & Load Library Packages.

## Install the library packages (if needed).
install.packages("magrittr")
install.packages("dplyr")
install.packages("tidyr")
install.packages("tidyverse")
install.packages("broom")
install.packages("ggplot2")
install.packages("cowplot")
install.packages("corrplot")
install.packages("ggcorrplot")
install.packages('fastDummies')
install.packages("PerformanceAnalytics")
install.packages("InformationValue")
install.packages("caret", dependencies = TRUE)
install.packages("pROC")
install.packages("caTools")
install.packages("car")


## Import the library packages. 
library(magrittr) # Needs to be run everytime when want to use %>% function.
library(dplyr) # alternatively, this also loads %>% function.
library(tidyr) # Works with dplyr library package to tidy data.
library(tidyverse) # Used in performing sub-setting, transformations, and visualizations.
library(broom) # Summarizes key information about models in tidy library package.
library(ggplot2) # Used for data visualization.
library(cowplot) # Used to improve ggplot2 data visualization.
library(corrplot) # To build correlation matrix.
library(ggcorrplot) # Used to easily visualize a correlation matrix using ggplot2.
library(fastDummies) # Used to create dummy variables (especially for categorical variables)
library(PerformanceAnalytics) # Used for Chart Correlation
library(InformationValue) # Used for model assessment and evaluation
library(caret) # To view confusion matrix.
library(pROC) # This is to analyze and compare ROC curves.
library(caTools) # Used to calculate AUC using ROC plot
library(car) # Used to make use of the box-tidwell function and calculate VIF values 





# Import the Heart Disease Data Set.

## setwd() before to check before importing.

## Import the dat set into a data frame using the read.csv() function.
df_heart <- read.csv("heart.csv")

## Print the first 6 rows of data frame.
head(df_heart) 





# Basic Data Exploration and Wrangling.

## Display the variable's names.
names(df_heart) 

## Display the list structure.
str(df_heart) 

## Display the basic descriptive statistics.
summary(df_heart) 

## Display the number of rows.
nrow(df_heart) 

## Display the number of columns.
ncol(df_heart) 

## Display the number of missing (NULL/NA) values.
colSums(is.na(df_heart)) 



## Data Wrangling: Edit the variables to better-fit the list structure and summary 
df_heart2 <- df_heart %>%
  mutate(ChestPainType = as.factor(ChestPainType),
         RestingECG = as.factor(RestingECG),
         ST_Slope = as.factor(ST_Slope),
         Sex = factor(Sex, levels = c('F', 'M'), labels = c("Female", "Male")),
         FastingBS = factor(FastingBS, levels = c(0, 1), labels = c("Otherwise", "Diabetic")),
         ExerciseAngina = factor(ExerciseAngina, levels = c('N', 'Y'), labels = c("No", "Yes")),
         HeartDisease = factor(HeartDisease, levels = c(0, 1), labels = c("No", "Yes")))
glimpse(df_heart2)

## Inspect the descriptive statistics after data wrangling.
summary(df_heart2) 



## Verify data is not imbalanced (mainly focus categorical variables).

xtabs(~ HeartDisease + Sex, data = df_heart2)

xtabs(~ HeartDisease + ChestPainType, data = df_heart2)

xtabs(~ HeartDisease + FastingBS, data = df_heart2)

xtabs(~ HeartDisease + RestingECG, data = df_heart2)

xtabs(~ HeartDisease + ExerciseAngina, data = df_heart2)

xtabs(~ HeartDisease + ST_Slope, data = df_heart2)





# Exploratory Data Analysis (EDA).

## Univariate Analysis.

### HeartDisease (Target Variable) Bar Chart.
ggplot(df_heart2, aes(x = HeartDisease)) +
  geom_bar(fill = "steelblue")  +
  labs(
    title = "Bar Chart of Total Heart Disease Count",
    caption = "Source: Heart Disease Dataset",
    x = "HeartDisease",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### Cholesterol Histogram.
ggplot(df_heart2, aes(Cholesterol)) +
  geom_histogram(color = "#000000", fill = "#0099F8") +
  labs(
    title = "Histogram of Patients' Cholesterol",
    caption = "Source: Heart Disease Dataset",
    x = "Cholesterol",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### RestingBP Histogram.
ggplot(df_heart2, aes(RestingBP)) +
  geom_histogram(color = "#000000", fill = "#0099F8") +
  labs(
    title = "Histogram of Patients' Resting Blood Pressure",
    caption = "Source: Heart Disease Dataset",
    x = "RestingBP",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### Oldpeak Histogram.
ggplot(df_heart2, aes(Oldpeak)) +
  geom_histogram(color = "#000000", fill = "#0099F8") +
  labs(
    title = "Histogram of Patients' Oldpeak (value of depression)",
    caption = "Source: Heart Disease Dataset",
    x = "Oldpeak",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )



## Bivariate Analysis.

### HeartDisease vs Sex.
ggplot(data = df_heart2, aes(x = Sex, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  labs(
    title = "HeartDisease vs Sex",
    caption = "Source: Heart Disease Dataset",
    x = "Sex",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### HeartDisease vs ChestPainType.
ggplot(data = df_heart2, aes(x = ChestPainType, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  labs(
    title = "HeartDisease vs ChestPainType",
    caption = "Source: Heart Disease Dataset",
    x = "ChestPainType",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### HeartDisease vs FastingBS.
ggplot(data = df_heart2, aes(x = FastingBS, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  labs(
    title = "HeartDisease vs FastingBS",
    caption = "Source: Heart Disease Dataset",
    x = "FastingBS",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### HeartDisease vs RestingECG.
ggplot(data = df_heart2, aes(x = RestingECG, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  labs(
    title = "HeartDisease vs RestingECG",
    caption = "Source: Heart Disease Dataset",
    x = "RestingECG",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### HeartDisease vs ExerciseAngina.
ggplot(data = df_heart2, aes(x = ExerciseAngina, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  labs(
    title = "HeartDisease vs ExerciseAngina",
    caption = "Source: Heart Disease Dataset",
    x = "ExerciseAngina",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### ExerciseAngina vs ChestPainType.
ggplot(data = df_heart2, aes(x = ChestPainType, fill = ExerciseAngina)) +
  geom_bar(position = "dodge") +
  labs(
    title = "ExerciseAngina vs ChestPainType",
    caption = "Source: Heart Disease Dataset",
    x = "ChestPainType",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### ExerciseAngina vs FastingBS.
ggplot(data = df_heart2, aes(x = FastingBS, fill = ExerciseAngina)) +
  geom_bar(position = "dodge") +
  labs(
    title = "ExerciseAngina vs FastingBS",
    caption = "Source: Heart Disease Dataset",
    x = "FastingBS",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

### HeartDisease vs. Age Histogram.
ggplot(df_heart2, aes(Age, color = HeartDisease)) + 
  geom_density(alpha = 0.5) + 
  labs(
    title = "HeartDisease vs. Age",
    caption = "Source: Heart Disease Dataset",
    x = "Age",
    y = "Density"
  )

### HeartDisease vs. MaxHR Histogram.
ggplot(df_heart2, aes(MaxHR, color = HeartDisease)) + 
  geom_density(alpha = 0.5) + 
  labs(
    title = "HeartDisease vs. MaxHR",
    caption = "Source: Heart Disease Dataset",
    x = "MaxHR",
    y = "Density"
  )





# Data Preprocessing & Transformation.

## Create dummy variables for categorical variables.
df_heart3 <- dummy_cols(df_heart2, select_columns = c('Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope'),
                       remove_selected_columns = TRUE, 
                       remove_first_dummy = TRUE) # Remove first dummy variable to prevent collinearity.



## Change HeartDisease data type to binary.
df_heart3$HeartDisease <- ifelse(df_heart3$HeartDisease == "Yes", 1, 0)



########## Notes ##########
# Removing outliers is a bit complicated and controversial because there are data that are real but are considered as outliers and removing them would make our model overfitting. 
# But, there are also errors when putting the values, for example in cholesterol and RestingBP, it have values 0 and it is something that it is not possible, but to have high cholesterol or RestingBP, yes.
# Hence, in this case, I will transform the data that is 0 using median imputation. 
########## Notes ##########

## Change 0 to NA and replace with median for Cholesterol & RestingBP (median imputation).
df_heart3 <- df_heart3 %>% 
  mutate_at(c('Cholesterol', 'RestingBP'), ~na_if(., 0))

df_heart3 <- df_heart3 %>% 
  mutate_if(is.numeric, list(~ replace(., is.na(.), median(., na.rm = TRUE))))

### Check to see any null or missing values in the data frame.
colSums(is.na(df_heart3))

### Compare the boxplots of the raw and cleaned Cholesterol values.
boxplot(df_heart3$Cholesterol, df_heart2$Cholesterol,
        main = "Boxplot of Patients' Cholesterol Values",
        xlab = "mm/dl",
        ylab = "Cholesterol (Before vs. After)",
        col = "orange",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)

### Compare the boxplots of the raw and cleaned RestingBP values.
boxplot(df_heart3$RestingBP, df_heart2$RestingBP,
        main = "Boxplot of Patients' Resting Blood Pressure Values",
        xlab = "mm/gh",
        ylab = "Resting Blood Pressure (Before vs. After)",
        col = "pink",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)



## Multivariate Analysis (Only available after transforming variables to numerical).

### Heat Map Correlation.
corrplot(cor(df_heart3))

### Chart Correlation. (takes awhile to load)
chart.Correlation(df_heart3, histogram = TRUE, pch = "+")





# Statistical Modeling & Assessment (Multiple Logistic Regression).

## Model Fitted Equation & Interpretation.

### split dataset into training and testing set (not needed for this subject but I proceeded because I do not want bias results).
set.seed(1)
sample_split <- sample(c(TRUE, FALSE), nrow(df_heart3), replace=TRUE, prob=c(0.7,0.3))
df_heart3_train <- df_heart3[sample_split, ]
df_heart3_test <- df_heart3[!sample_split, ]

### Fit the glm() model (logistic regression) into logreg.
logreg <- glm(formula = HeartDisease ~ ., family = binomial(link="logit"), data = df_heart3_train) 

### Call the logreg using the summary() function.
summary(logreg)

### Use the anova() function to analyze the table of deviance.
anova(logreg, test="Chisq")


######################################## Results & Interpretation ########################################

# Results from the fitted logistic regression model shows that there are some insignificant variables such as ...
# "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "ChestPainType_TA", "RestingECG_Normal", "RestingECG_ST", and "ST_Slope_Up" based on its respective p-values.

# Results from the ANOVA test shows that "RestingBP", "Cholesterol", "ChestPainType_TA", "RestingECG_Normal", "RestingECG_ST", and "ST_Slope_Up" are insignificant to the fitted model.
# This is judged by the low deviance residuals as well as the Pr(>Chi) of > .05, respectively.

# Hence, fit the updated glm() model (logistic regression) without the insignificant variables (excluding "MaxHR" because it shows significance in the ANOVA test) into logreg2.
# Even though "Oldpeak" is significant based on the ANOVA test, for this case, it will be excluded from the model fit because ...
# 1. The AIC only differs with around a value of 10 when comparing with or without the variable, which is not evident enough to justify it decreases the model performance.
# 2. The variable will violate the assumption of linearity of independent variables and log-odds due to its negative and 0 data values.

######################################## Results & Interpretation ########################################


### Fit the updated glm() model (logistic regression) into logreg2.
logreg2 <- glm(formula = HeartDisease ~ Age + MaxHR + Sex_Male + 
                 ChestPainType_ATA + ChestPainType_NAP + FastingBS_Diabetic + 
                 ExerciseAngina_Yes + ST_Slope_Flat, 
               family = binomial(link="logit"), 
               data = df_heart3_train) 

summary(logreg2)

### Use the anova() function to analyze the updated table of deviance.
anova(logreg2, test="Chisq")



######################################## Results & Interpretation ########################################

# Results from the updated fitted logistic regression model shows that only "MaxHR" variable is insignificant based on its p-value.

# Additionally, results from the ANOVA test shows all variables, including "MaxHR", are significant to the fitted model.
# This is judged by the low deviance residuals as well as the Pr(>Chi) of > .05, respectively.

# Hence, the "MaxHR" variable is kept in the fitted model. 

######################################## Results & Interpretation ########################################


######################################## Model Fitting Output & Interpretation ######################################## 

# Summary of the deviance residuals.
## Results show that the values looks good since they are close to being centered on 0 and are roughly symmetrical.


# Coefficients & Wald statistic (Chi-Square statistic).
## Equation: ln(odds) = ln(p/(1-p)) = ??0 + ??1X + ??2X + ... ??NX
### ln(p/(1-p))(HeartDisease) = -3.508525 + 0.046(Age) - 0.006(MaxHR) + 1.350(Sex_Male) - 1.828(ChestPainType_ATA) 
###                                   - 1.409(ChestPainType_NAP) + 1.119(FastingBS_Diabetic) 
###                                   + 1.194(ExerciseAngina_Yes) + 2.232(ST_Slope_Flat)

### This can be interpreted as the probability of a patient having heart disease.
### The interpretation of a predictor variable means that it assumes the rest of the predictor variables in the fitted model remain constant.

### Interpretation of continuous variables: Increasing the age by 1 unit (1 year) will result in a 0.05 increase in logit(p). 
### Since exp(0.05) = 1.05, this is a 5% increase in the odds of a patient having heart disease.


### Interpretation of continuous variables:Binary: The impact of being a male patient with heart disease will result in a 1.35 increase in logit(p). 
### Since exp(1.35) = 3.86, this means the odds for a patient having heart disease are 286% higher for male.


### Interpretation of continuous variables:Categorical: The impact of being a diabetic patient with heart disease will result in a 1.12 increase in logit(p). 
### Since exp(1.12) = 3.07, this means the odds for a patient having heart disease are 207% higher if the patient is a diabetic.


## Pr(>|z|)
### The column represents the p-value associated with the value in the z value column.
### If the p-value is less than a certain significance level (e.g. ?? = .05).
### Then, this indicates that the predictor variable has a statistically significant relationship with the response variable in the model.
### If p-value below 0.05, thus, the log(odds) and the log(odds ratios) are statistically significant. As an example, ...

### The p-value for the predictor variable "MaxHR" is .24. 
### Since this value is not less than .05, it does not have a statistically significant relationship with the response variable in the model.
### However, based on the anova test, it showed that leaving this predictor in the model improves the overall model performance.

### All other predictor variables have a p-value of less than .05
### Hence, they have a statistically significant relationship with the response variable in the model.


# Null and Residual Deviance.
## Can be used to compare models, compute R-sq and overall p-value. 
## In this case, it will ...

### 1. Check for overdispersion (Residual deviance > degrees of freedom (df))/ underdispersion (Residual deviance < degrees of freedom (df)).
### Results showed that the model is underdispersion. Negative binomial dispersion is out of scope for this topic.
### Hence, this will be up for discussion and recommendation in the future.

### 2. Test the overall model fit using logistic regression hypothesis testing.

### For testing the full model against the null, one can construct the test from the values of likelihoods or deviances and degrees of freedom typically reported for the full and null models. 
### Using the summary(), take the difference between the reported null and residual deviances, and test against chi-square with degrees of freedom equal to the difference between the null and residual degrees of freedom.
### To calculate corresponding p-value of overall Chi-Square statistic (Null deviance - Residual deviance) / (Null df - Residual df).

1-pchisq(879.82-453.14, 640-632)

########## Logistic Regression Hypothesis Testing Results & Interpretation ##########

# H0: ??1 = ??2 = . = ??k = 0
# HA: ??1 = ??2 = . = ??k ??? 0

# The overall p-value below 0.05 = significant.
# Since p-value = 0, it means that it is too small.
# Since this p-value is less than .05, we reject the null hypothesis (H0).
# In other words, there is a statistically significant relationship between the response variable (Heart Disease) and predictor variables.

########## Logistic Regression Hypothesis Testing Results & Interpretation ##########


# Akaike Information Criterion (AIC).
## The residual deviance adjusted for the number of parameters in the model.
## Can be used to compare one model to another.
### In this case, used to compare logreg (original model) to logreg2 (updated model).
### Usually, a lower AIC value is better, but in this case, the values differ around 10, which is not evident enough to justify which is a better model.

######################################## Model Fitting Output & Interpretation ######################################## 


## Model Assessment & Evaluation.

# Cassification or prediction
# The table() function is used to build a confusion matrix for the fitted model. 
# In the example code below, the prediction threshold used is 0.5. A commonly used classifier threshold is 0.5.



# Predict on df_heart3
logreg.pred <- predict(logreg2, df_heart3_test, type = "response")

logreg.pred <- ifelse(logreg.pred > 0.5, 1, 0)

# Find the optimal cutoff probability to use to maximize accuracy
optimal <- optimalCutoff(df_heart3_test$HeartDisease, logreg.pred)

# Create confusion matrix
caret::confusionMatrix(as.factor(logreg.pred), as.factor(df_heart3_test$HeartDisease))

# Calculate total misclassification error rate (or Accuracy (100 - misclassification error rate))
misClassError(df_heart3_test$HeartDisease, logreg.pred, threshold = optimal)

# Calculate AUC using ROC plot (OR auc(df_heart3_test$HeartDisease, logreg.pred))
colAUC(logreg.pred, df_heart3_test$HeartDisease, plotROC = TRUE)


######################################## Results & Interpretation ########################################

# Confusion Matrix

## The fitted model for the test set predicted the classification accuracy of around 84% and the misclassification error rate is 16%, which is good.

## The results also consist of 82% and 86% for sensitivity and specificity, respectively. 
## This means that there is a large proportion of patients' with actual heart disease and a small proportion with false heart disease while there is a large proportion of patients' do not have actual heart disease and a small proportion do not have false heart disease. 

## The ROC curve and AUC value evaluated for the fitted model for the test set is 0.84. 
## This means there is a 84% chance that the model is able to classify patients between positive class (have heart disease) and negative class (do not have heart disease).

######################################## Results & Interpretation ########################################





# Regression Assumption Diagnostics (Multiple Logistic Regression).

## Assumption 1: Test for Appropriate Binary Outcome Type.

### Verify there should only be two unique outcomes in the outcome variable.
unique(df_heart3_train$HeartDisease) 

### HeartDisease (Target Variable) Bar Chart.
ggplot(df_heart3_train, aes(x = HeartDisease)) +
  geom_bar(fill = "steelblue")  +
  labs(
    title = "Bar Chart of Total Heart Disease Count",
    caption = "Source: Heart Disease Dataset",
    x = "Heart Disease",
    y = "Count"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 10, face = "bold"),
    plot.caption = element_text(face = "italic")
  )

######################################## Results & Interpretation ########################################

# Results show that there are only two outcomes (i.e. binary classification of have heart disease or does not have heart disease).
# The assumption is met. 

######################################## Results & Interpretation ########################################



## Assumption 2: Test for Sufficiently Large Sample Size of Data Set.

### Verify the sufficient number of rows of the data set.
nrow(df_heart3_train) 

### Verify the sufficient number of columns of the data set.
ncol(df_heart3_train) 

### Verify the list structure of the data set.
str(df_heart3_train) 

######################################## Results & Interpretation ########################################

# Ideally, the dataset should have at least 500 rows of observations.
# Plus, having at least 10-20 instances of the least frequent outcome for each predictor variable in the model.
# Results show that there are 641 rows of observations and 16 variables.
# The assumption is met. 

######################################## Results & Interpretation ########################################



## Assumption 3: Test for Linearity of independent variables and log-odds.

### Box-Tidwell Test with 'car' library and `boxTidwell` function.
boxTidwell(formula = HeartDisease ~ Age + MaxHR,
           other.x = ~ Sex_Male + ChestPainType_ATA + ChestPainType_NAP +  
             FastingBS_Diabetic  + ExerciseAngina_Yes + ST_Slope_Flat, 
           data = df_heart3_train)

######################################## Results & Interpretation ########################################

# One of the important assumptions of logistic regression is the linearity of the logit over the continuous covariates. 
# This assumption means that relationships between the continuous predictors and the logit (log odds) is linear.
# Furthermore, Box-Tidwell Test only works for positive values.
# As mentioned in the model fitting and interpretation section, the "Oldpeak" variable will be dropped from the model because ... 
# The positive and negative values of "Oldpeak" have different meaning and data transformation would cause bias results.

# Results show that the variables "Age" and "MaxHR" are not significant based on Pr(>|z|) > .05
# Meaning, there is linearity in the "Age" and "MaxHR" log-odds features.
# Hence, the assumption is met. 

######################################## Results & Interpretation ########################################



## Assumption 4: Test for the Absence of Multicollinearity (Multicollinearity Analysis).

######################################## Notes ########################################

# Method 1: Correlation Matrix.
## visualizes the correlation between multiple continuous variables. 
## Correlations range always between -1 and +1, where -1 represents perfect negative correlation and +1 perfect positive correlation.
## Correlations close to- 1 or +1 might indicate the existence of multicollinearity. 
## As a rule of thumb, one might suspect multicollinearity when the correlation between two (predictor) variables is below -0.9 or above +0.9.

# Method 2: Variance Inflation Factors (VIF).
## The Variance Inflation Factor (VIF) measures the inflation in the coefficient of the independent variable due to the collinearities among the other independent variables. 
## A VIF of 1 means that the regression coefficient is not inflated by the presence of the other predictors, and hence multicollinearity does not exist.
## As a rule of thumb, a VIF exceeding 5 requires further investigation, whereas VIFs above 10 indicate multicollinearity. 
## Ideally, the Variance Inflation Factors are below 5.

######################################## Notes ########################################

### Method 1: Correlation Matrix.

corrplot(cor(df_heart3_train), method = "number")

### Method 2: Variance Inflation Factors (VIF).

car::vif(logreg2)

######################################## Results & Interpretation ########################################

# Method 1: Correlation Matrix.
## Result shows that 2 variables: "ST_Slope_Flat" and "ST_Slope_Up" have high correlation with each other.
## In this case, the correlation matrix can be difficult to interpret because there are many independent variables.
## Furthermore, not all collinearity problems can be detected by inspection of the correlation matrix.
## It is possible for collinearity to exist between three or more variables even if no pair of variables has a particularly high correlation.

# Method 2: Variance Inflation Factors (VIF).
## Results show that all the variables have a VIF value of < 5.

# Since none of the variables crossed the previous mentioned threshold values, respectively, the assumption is met.

######################################## Results & Interpretation ########################################



## Assumption 5: Test for the Absence of Strongly Influential Outliers.

######################################## Notes ########################################

# Test using standardized residuals and Cook's Distance.
# Standardized residuals values > 3 = influential outlier.
# Cook's D value > Cook's D Threshold (4/N) = influential outlier.

######################################## Notes ########################################


### Place all the calculated values from the logistic regression model into a new data frame.
logreg.data <- augment(logreg2) %>%
  mutate(index = 1:n())

### Show the top 6 highest standardized residuals (if > 3 = influential outlier).
head(logreg.data$.std.resid[order(-logreg.data$.std.resid)])

### Plot of standardized residuals
plot(fitted(logreg2),
     rstandard(logreg2))


### Set Cook's D Threshold.
cook_threshold <- 4 / nrow(df_heart3_train)

### Cook's D Plot.
plot(logreg2, which = 4, id.n = 12)
abline(h = cook_threshold, col = "red")

### Put outlier data into a new data frame where > Cook's D Threshold = influential outliers.
influ_out <- logreg.data %>%
  filter(.cooksd > cook_threshold)

### Get the percentage of influential outliers.
outliers <- round(100*(nrow(influ_out) / nrow(logreg.data)),1)

### Store values in a data variable.
print_outliers <- format(round(outliers, 2), nsmall = 2)

### Print the number of percentage of observations that exceed Cook's distance threshold.
sprintf('Proportion of data points that are highly influential = %s Percent', print_outliers)

######################################## Results & Interpretation ########################################

# Standardized Residuals.
## Results show that none of the data points of the fitted model consist of any outliers.

# Cook's Distance.
## In addition, based on the pre-defined threshold (4/N), only 9.2% (59/641) of the data points are in the outlier zone, which is small as well.

# Since none of the variables crossed the previous mentioned threshold values, respectively, the assumption is met.
# The management of outliers is outside the scope of this subject.
 
######################################## Results & Interpretation ########################################



## Assumption 6: Test for Independence of Observations (Residuals Analysis).

######################################## Notes ########################################

# Due to needs to transform to log(odds), the need for testing independence of observation is out of scope of this assessment.
# The assumption is ignored. 

######################################## Notes ########################################





# Conclusion 

# Create data frame that contains the probabilities of having HD with the actual HD status.
predicted.data <- data.frame(
  probability.of.hd = logreg2$fitted.values,
  hd = df_heart3_train$HeartDisease
)

# Sort the data frame from low to high probabilities
predicted.data <- predicted.data[
  order(predicted.data$probability.of.hd, decreasing=FALSE),]

# Add new column to the data frame that has the rank of each sample, from low to high probability.
predicted.data$rank <- 1:nrow(predicted.data)

# Plot the graph
ggplot(data = predicted.data, aes(x = rank, y=probability.of.hd)) + 
  geom_point(aes(color = hd), alpha =1 , shape = 4, stroke = 2) + 
  xlab("Index") + 
  ylab("Predicted probaility of getting heart disease")
  

######################################## Notes ########################################

# The graph shows the predicted probabilities that each patient has heart disease along with their actual heart disease status.
# Most of the patients with heart disease (1), are predicted to have a high probability of having heart disease.
# Moreover, most of the patients without heart disease (0), are predicted to have a low probability of having heart disease. 
# Thus, the fitted logistic regression model has done a good job at classifying patients' heart disease status.

######################################## Notes ########################################
