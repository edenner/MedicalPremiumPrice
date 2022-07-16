Prediction of Medical Premium Price
================
Elena Denner

### Background and Motivation

Health insurance first began in the United States in 1929 by a group of
Texas Hospitals that banded together to help patients pay for medical
expenses. The first medical insurance company, Blue Cross, extended
benefits to Dallas teachers for only a 50 cents monthly premium
(California Health Institute, 2019). Now, almost 100 years later, health
insurance prices for an individual have increased to an average of $456
per month (Porretta, 2021). Health care coverage makes it possible for
people to afford medical treatment in the face of health related
complications. It is extremely important to be medically insured in case
of an emergency, accident, or disease onset. Insurance companies assess
an individual’s lifestyle, medical history, and other physical
attributes to determine their premium price for medical coverage.

### Goals and Methodology

The goal of this project is to build a regression model that accurately
predicts a person’s yearly medical coverage costs and to determine the
most influential variables in doing so. The modeling techniques utilized
are: step-wise linear regression (forward and backward selection),
penalized linear regression (ridge and lasso), principal components
regression, and decision trees (standard, bagging, random forest, and
boosting).

### Libraries

The following libraries are used throughout the project for tidying the
data, randomly splitting the data, creating data visualizations,
formulating summary tables, and building a variety of machine learning
models.

``` r
library(tidyverse)
library(dplyr)
library(rsample)
library(randomForest)
library(janitor)
library(rpart)
library(rpart.plot)
library(gbm)
library(glmnet)
library(tree)
library(RColorBrewer)
library(gridExtra)
library(ggplot2)
library(pls)
library(jtools)
```

### Dataset

The dataset, “Medical Insurance Premium Prediction,” comes from
kaggle.com and was originally published on August 4, 2021 (Tejashvi,
2021). The dataset contains simulated yearly medical coverage costs for
986 customers. The premium price for each customer is given in Indian
Rupees, which has the following currency conversion: 1 INR = 0.013 US
dollar. Data was voluntarily given by all customers and contains 10
health-related variables: Age (years), Diabetes (0 or 1), Blood Pressure
Problems (0 or 1), Any Major Organ Transplants (0 or 1), Any Chronic
Diseases (0 or 1), Height (cm), Weight (kg), Any Known Allergies (0 or
1), Family History of Cancer (0 or 1), and Number of Major Surgeries
(0,1,2 or 3).

``` r
setwd('/Users/elena/Desktop/medpremium/')
medpremium <- read_csv("Medicalpremium.csv")
glimpse(medpremium)
```

    Rows: 986
    Columns: 11
    $ Age                     <dbl> 45, 60, 36, 52, 38, 30, 33, 23, 48, 38, 60, 66…
    $ Diabetes                <dbl> 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0…
    $ BloodPressureProblems   <dbl> 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0…
    $ AnyTransplants          <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0…
    $ AnyChronicDiseases      <dbl> 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
    $ Height                  <dbl> 155, 180, 158, 183, 166, 160, 150, 181, 169, 1…
    $ Weight                  <dbl> 57, 73, 59, 93, 88, 69, 54, 79, 74, 93, 74, 67…
    $ KnownAllergies          <dbl> 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1…
    $ HistoryOfCancerInFamily <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0…
    $ NumberOfMajorSurgeries  <dbl> 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 2, 0, 1, 0, 1, 1…
    $ PremiumPrice            <dbl> 25000, 29000, 23000, 28000, 23000, 23000, 2100…

### Data Cleaning

Before beginning analysis, the dataset must be cleaned. This entails
dropping any missing values and converting our binary variables to
factors. We will also recode ‘0’ as ‘No’ and ‘1’ as ‘Yes’ for the
relevant predictors. In addition, the column names of the data set are
renamed such that they do not contain spaces or capital letters. This
naming convention of the variables is necessary when using R modeling
functions later and is good coding practice.

``` r
medpremium <- clean_names(medpremium)

medpremium <- medpremium %>%
  drop_na() %>%
  mutate(diabetes = as.factor(case_when(diabetes == 0 ~ "No",
                              diabetes == 1 ~ "Yes"))) %>%
  mutate(blood_pressure_problems = as.factor(case_when(blood_pressure_problems == 0 ~ "No",
                              blood_pressure_problems == 1 ~ "Yes"))) %>%
  mutate(any_transplants = as.factor(case_when(any_transplants == 0 ~ "No",
                              any_transplants == 1 ~ "Yes"))) %>%
  mutate(any_chronic_diseases = as.factor(case_when(any_chronic_diseases == 0 ~ "No",
                              any_chronic_diseases == 1 ~ "Yes"))) %>%
  mutate(known_allergies = as.factor(case_when(known_allergies == 0 ~ "No",
                              known_allergies == 1 ~ "Yes"))) %>%
  mutate(history_of_cancer_in_family = as.factor(case_when(history_of_cancer_in_family == 0 ~ "No",
                              history_of_cancer_in_family == 1 ~ "Yes")))
```

### Data Exploration

Next we want to familiarize ourselves with the predictor variables by
creating visualizations of each one. A box plot nicely shows the
distribution of all six binary variables and their relationship with
premium price. Scatter plots of age, weight, height and a violin plot of
number of major surgeries versus premium price are shown for the
numerical predictors.

``` r
v1 <- ggplot(medpremium) +
  geom_boxplot(aes(y=premium_price, x=diabetes, fill=diabetes), show.legend=FALSE) +
  xlab("Diabetes") +
  ylab("Premium Price")

v2 <- ggplot(medpremium) +
  geom_boxplot(aes(y=premium_price, x=any_transplants, fill=any_transplants), show.legend=FALSE) +
  xlab("Any Transplants") +
  ylab("Premium Price")

v3 <- ggplot(medpremium) +
  geom_boxplot(aes(y=premium_price, x=any_chronic_diseases, fill=any_chronic_diseases), show.legend = FALSE) +
   xlab("Chronic Diseases") +
  ylab("Premium Price")

v4 <- ggplot(medpremium) +
  geom_boxplot(aes(y=premium_price, x=blood_pressure_problems, fill=blood_pressure_problems), show.legend = FALSE) +
   xlab("Blood Pressure Problems") +
  ylab("Premium Price")

v5 <- ggplot(medpremium) +
  geom_boxplot(aes(y=premium_price, x=known_allergies, fill=known_allergies), show.legend = FALSE) +
   xlab("Known Allergies") +
  ylab("Premium Price")

v6 <- ggplot(medpremium) +
  geom_boxplot(aes(y=premium_price, x=history_of_cancer_in_family, fill=history_of_cancer_in_family), show.legend = FALSE) +
  xlab("Cancer in Family") +
  ylab("Premium Price")

grid.arrange(v1, v2, v3, v4, v5, v6, nrow=2)
```

![](medicalpremium_files/figure-gfm/exp1-1.png)<!-- -->

There appears to be a substantial difference in premium price for
individuals that have had transplants compared to individuals that have
not had any transplants, due to the middle 50% of observations not
overlapping. Chronic diseases and blood pressure problems also appear to
have a substantial trend upwards in premium price for individuals that
have these conditions compared to individuals that do not. The average
premium price for subjects with these conditions tends to be greater
than that of those without the condition. The distributions of price for
Diabetes and Cancer in the Family appear to be more similar for people
with and without these conditions, which suggests these two variables
may not be significant predictors of premium price. The boxplot for
Known Allergies shows almost an identical distribution of premium price
for individuals with and without this condition, indicating known
allergies likely does not affect insurance companies’ rates for medical
coverage costs.

``` r
v7 <- ggplot(medpremium) +
  geom_point(aes(x=age,y=premium_price)) +
  geom_smooth(aes(x=age,y=premium_price)) +
  xlab("Age (years)") +
  ylab("Premium Price")

v8 <- ggplot(medpremium) +
  geom_point(aes(x=weight,y=premium_price)) +
  geom_smooth(aes(x=weight,y=premium_price), colour="green") +
  xlab("Weight (kg)") +
  ylab("Premium Price")

v9 <- ggplot(medpremium) +
  geom_point(aes(x=height,y=premium_price)) +
  geom_smooth(aes(x=height,y=premium_price), colour="red") +
  xlab("Height (cm)") +
  ylab("Premium Price")

v10 <- ggplot(medpremium, mapping=aes(x=premium_price, y=factor(number_of_major_surgeries), fill=factor(number_of_major_surgeries))) +
  geom_violin(color="red", fill="orange", alpha=0.2, show.legend = FALSE) +
  labs(fill="Number of Major Surgeries") +
  ylab("Number of Major Surgeries") +
  xlab("Premium Price")

grid.arrange(v7, v8, v9, v10, nrow=2)
```

![](medicalpremium_files/figure-gfm/exp2-1.png)<!-- -->

There seems to be a positive correlation between age and premium price,
for an increase in age coincides with an in increase in premium price.
However, no linear relationship is clear for height or weight with
premium price, as the distribution of price appears not to change very
much when weight or height increases. Number of major surgeries may
affect premium price, for there does appear to be a more right skewed
distribution of price for someone with 2 surgeries compared to someone
with only 1 major surgery or someone without any surgeries. However, the
distribution of price is nearly identical for people who have had 0
major surgeries or 1 major surgery. Very few people (only 16
observations) have had 3 surgeries, hence the singular dot in the graph.

### Split data into training and testing sets

Now that we are familiar with the trends in our data, we can begin the
machine learning process. The dataset is randomly split into the
training set and the testing set, with 75% in the training set (740
observations) and 25% in the test set (246 observations). We will build
our models on the training set and evaluate their performance on the
test set.

``` r
set.seed(11)
med.split <- initial_split(medpremium, prop = 3/4)
med.train <- training(med.split)
med.test <- testing(med.split)
```

The goal across supervised machine learning techniques is to build a
model using the data available (training set) to accurately predict
unforeseen data (test set). Two measures for each model will be computed
and compared: r-squared and mean squared error. The square of the
correlation coefficient (r-squared) measures the percent of variance in
premium price explained by the regression model, and so a higher
r-squared indicates a better fitting model. Mean squared error (MSE)
measures the loss of each model and a lower MSE indicates a more
accurate model. The functions used for computing r squared and mean
squared error are printed below:

``` r
rsquared <- function(pred){
  if (length(pred)==length(med.test$premium_price)){
    r2 = 1 - (sum((med.test$premium_price-pred)^2)/sum((med.test$premium_price-mean(med.test$premium_price))^2))
  }
  if (length(pred)==length(med.train$premium_price)){
    r2 = 1 - (sum((med.train$premium_price-pred)^2)/sum((med.train$premium_price-mean(med.train$premium_price))^2))
  }
  return (r2)
}


MSE <- function(pred){
  if (length(pred)==length(med.test$premium_price)){
    mse = sum((med.test$premium_price-pred)^2)/length(med.test$premium_price)
  }
  if (length(pred)==length(med.train$premium_price)){
    mse = sum((med.train$premium_price-pred)^2)/length(med.train$premium_price)
  }
  return (mse)
}
```

When building complex machine learning models, over fitting may occur if
the random “noise” in the training set is given too much emphasis. This
random noise in the training set is not significant for prediction of
the test set and fitting parameters too closely to this random noise may
lower the test set accuracy. However, too simple of a model will cover
very little variance in the test set and produce inaccurate results.
Finding the right balance of model complexity is key to building an
accurate machine learning model, and there are many techniques to do so.
Evaluating these techniques is the next step in our analysis.

### Linear Stepwise Regression

Linear regression is the simplest technique for prediction of a
continuous numerical response. An estimated slope coefficient is
assigned to each independent variable and an intercept term may be
included as well. Two variable selection methods for linear regression
will be evaluated: forward selection and backward selection. Forward
selection starts with an empty model and adds variables in, beginning
with the most significant variable and successively adding contributing
variables until a stopping criterion is met. Backward selection starts
with a full model consisting of all predictor variables and removes the
least contributing variables in order until a stopping criterion is met.

#### Forward Selection

``` r
linear.fwd <- step(lm(premium_price ~., data=med.train), direction = c("forward"))

fwd.pred.train = predict(linear.fwd, med.train)
mse.fwd.train = MSE(fwd.pred.train)
r2.fwd.train = rsquared(fwd.pred.train)

fwd.pred.test = predict(linear.fwd, med.test)
mse.fwd.test = MSE(fwd.pred.test)
r2.fwd.test = rsquared(fwd.pred.test)
```

``` r
summary(linear.fwd)
```


    Call:
    lm(formula = premium_price ~ age + diabetes + blood_pressure_problems + 
        any_transplants + any_chronic_diseases + height + weight + 
        known_allergies + history_of_cancer_in_family + number_of_major_surgeries, 
        data = med.train)

    Residuals:
       Min     1Q Median     3Q    Max 
    -12277  -2189   -420   1750  24127 

    Coefficients:
                                   Estimate Std. Error t value Pr(>|t|)    
    (Intercept)                     6312.38    2379.55    2.65   0.0082 ** 
    age                              325.43      11.13   29.25  < 2e-16 ***
    diabetesYes                     -504.56     287.77   -1.75   0.0800 .  
    blood_pressure_problemsYes       159.69     289.05    0.55   0.5808    
    any_transplantsYes              8478.74     643.59   13.17  < 2e-16 ***
    any_chronic_diseasesYes         2398.05     361.45    6.63  6.4e-11 ***
    height                            -9.87      13.61   -0.73   0.4685    
    weight                            70.28       9.67    7.27  9.2e-13 ***
    known_allergiesYes               350.39     338.49    1.04   0.3009    
    history_of_cancer_in_familyYes  2289.09     447.32    5.12  4.0e-07 ***
    number_of_major_surgeries       -642.98     211.62   -3.04   0.0025 ** 
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 3700 on 729 degrees of freedom
    Multiple R-squared:  0.648, Adjusted R-squared:  0.643 
    F-statistic:  134 on 10 and 729 DF,  p-value: <2e-16

The variables found to be significant by forward stepwise selection in
prediction of premium price are: age, any transplants, any chronic
diseases, weight, history of cancer in the family, and number of major
surgeries. The slope coefficient for age found by forward selection is
325.43, which is interpreted as such: for every one year increase in
age, an individual’s yearly medical coverage cost is expected to
increase by 325.43 rupees on average. The r squared on the training set
is 0.6479 which indicates 64.79% of the variance in premium price is
explained by its linear relationship with these six variables. The
training MSE for the forwards selection model is 13478542.

#### Backward Selection

``` r
linear.bwd = step(lm(premium_price ~., data=med.train), direction = c("backward"))

bwd.pred.train = predict(linear.bwd, med.train)
mse.bwd.train = MSE(bwd.pred.train)
r2.bwd.train = rsquared(bwd.pred.train)

bwd.pred.test = predict(linear.bwd, med.test)
mse.bwd.test = MSE(bwd.pred.test)
r2.bwd.test = rsquared(bwd.pred.test)
```

``` r
summary(linear.bwd)
```


    Call:
    lm(formula = premium_price ~ age + diabetes + any_transplants + 
        any_chronic_diseases + weight + history_of_cancer_in_family + 
        number_of_major_surgeries, data = med.train)

    Residuals:
       Min     1Q Median     3Q    Max 
    -12221  -2152   -467   1745  23941 

    Coefficients:
                                   Estimate Std. Error t value Pr(>|t|)    
    (Intercept)                     4804.86     865.23    5.55  3.9e-08 ***
    age                              325.84      11.01   29.59  < 2e-16 ***
    diabetesYes                     -515.76     285.96   -1.80   0.0717 .  
    any_transplantsYes              8488.40     641.30   13.24  < 2e-16 ***
    any_chronic_diseasesYes         2376.39     360.17    6.60  8.0e-11 ***
    weight                            69.69       9.57    7.28  8.6e-13 ***
    history_of_cancer_in_familyYes  2329.98     445.58    5.23  2.2e-07 ***
    number_of_major_surgeries       -605.35     207.38   -2.92   0.0036 ** 
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 3700 on 732 degrees of freedom
    Multiple R-squared:  0.647, Adjusted R-squared:  0.644 
    F-statistic:  192 on 7 and 732 DF,  p-value: <2e-16

The variables found to be significant by backward stepwise selection are
the same as in forward selection. The slope coefficient for any
transplants found by backward selection is 8488.4, which is interpreted
as follows: the yearly medical coverage cost for an individual that has
any transplants is expected to be 8488.4 rupees higher on average
compared to individuals that have not had any transplants. The r squared
on the training set is 0.6469 which indicates 64.69% of the variance in
premium price is explained by its linear relationship with the six
variables found by backward selection. The training MSE for the
backwards selection model is 13515399 which is higher than the MSE of
the forwards selection model, indicating the forwards selection model is
a better fit for the training data.

### Penalized Linear Regression

Penalized linear regression methods keep all variables in the model but
shrink some coefficients towards zero that are not significant. Two
shrinkage methods are used: Ridge Regression (shrinks coefficients
proportional to the sum of their square, called the L2 norm) and Lasso
Regression (shrink coefficients relative to the sum of their absolute
value, called the L1 norm).

#### Ridge Regression

``` r
grid = 10^seq(10, -2, length=100)
xtrain = model.matrix(premium_price ~.,med.train)[,-1]
ytrain = med.train$premium_price
xtest = model.matrix(premium_price ~.,med.test)[,-1]
ytest = med.test$premium_price

cv.ridge.out = cv.glmnet(xtrain, ytrain, alpha=0)
bestlam.ridge = cv.ridge.out$lambda.min
i <- which(cv.ridge.out$lambda == cv.ridge.out$lambda.min)
mse.min.ridge <- cv.ridge.out$cvm[i]

ridge.model = glmnet(xtrain, ytrain, alpha=0, lambda=bestlam.ridge)

ridge.pred.train = predict(ridge.model, newx=xtrain)
mse.ridge.train = MSE(ridge.pred.train)
r2.ridge.train = rsquared(ridge.pred.train)

ridge.pred.test = predict(ridge.model, newx=xtest)
mse.ridge.test = MSE(ridge.pred.test)
r2.ridge.test = rsquared(ridge.pred.test)
```

#### Lasso

``` r
cv.lasso.out = cv.glmnet(xtrain, ytrain, alpha=1)
bestlam.lasso = cv.lasso.out$lambda.min
i <- which(cv.lasso.out$lambda == cv.lasso.out$lambda.min)
mse.min.lasso <- cv.lasso.out$cvm[i]

lasso.model = glmnet(xtrain, ytrain, alpha=1, lambda=bestlam.lasso)

lasso.pred.train = predict(lasso.model, newx=xtrain)
mse.lasso.train = MSE(lasso.pred.train)
r2.lasso.train = rsquared(lasso.pred.train)

lasso.pred.test = predict(lasso.model, newx=xtest)
mse.lasso.test = MSE(lasso.pred.test)
r2.lasso.test = rsquared(lasso.pred.test)
```

``` r
s1 <- ggplot(mapping = aes(x=log(cv.ridge.out$lambda), y=cv.ridge.out$cvm)) +
  geom_point() +
  geom_point(aes(x=log(bestlam.ridge), y=mse.min.ridge, color="blue", size = 2), show.legend = FALSE, color="blue") +
  geom_errorbar(aes(ymin=cv.ridge.out$cvm-cv.ridge.out$cvsd, ymax=cv.ridge.out$cvm+cv.ridge.out$cvsd), color="gray") +
  xlab("Log(lambda)") +
  ylab("Mean Squared Error") +
  labs(title = "Optimal Lambda for Ridge Regression", subtitle = paste("Best Lambda: ", bestlam.ridge)) +
  theme_classic()

s2 <- ggplot(mapping = aes(x=log(cv.lasso.out$lambda), y=cv.lasso.out$cvm)) +
  geom_point() +
  geom_point(aes(x=log(bestlam.lasso), y=mse.min.lasso, size = 2), show.legend = FALSE, color="red") +
  geom_errorbar(aes(ymin=cv.lasso.out$cvm-cv.lasso.out$cvsd, ymax=cv.lasso.out$cvm+cv.lasso.out$cvsd), color="gray") +
  xlab("Log(lambda)") +
  ylab("Mean Squared Error") +
  labs(title = "Optimal Lambda for Lasso Regression", subtitle = paste("Best Lambda: ", bestlam.lasso)) +
  theme_classic()

grid.arrange(s1, s2, nrow=2)
```

![](medicalpremium_files/figure-gfm/lambda-1.png)<!-- -->

The amount of shrinkage for each method is determined by lambda, a
tuning parameter. Lambda was chosen for each method by 10 fold cross
validation using the mean squared error on the training set to measure
performance. The optimal lambda for ridge regression is 437.2077 and
optimal lambda for lasso is 26.2099 as shown by the plots. 64.42% of the
variance in premium price can be explained by the ridge regression
model, while 64.77% of this variability can be explained by the lasso
model. The training MSE for the ridge regression model is 13622112 and
the training MSE for the lasso model is 13487396. Lasso is a more
accurate model compared to ridge regression, and has a lower MSE
compared to backward selection but a higher MSE compared to forward
selection. Ridge Regression has a higher MSE than both forward and
backward selection.

``` r
ridge.coef <- rownames_to_column(data.frame(coef(ridge.model)[,1]), var = "Variable") %>%
  rename(Coefficient = coef.ridge.model....1.)

ridge.coef <- ridge.coef %>%
  filter(Variable != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))
lasso.coef <- rownames_to_column(data.frame(coef(lasso.model)[,1]), var = "Variable") %>%
  rename(Coefficient = coef.lasso.model....1.)

lasso.coef <- lasso.coef %>%
  filter(Variable != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))

coef.compare <- lasso.coef %>%
  left_join(ridge.coef, by = "Variable") %>%
  rename("Lasso Coefficient" = Coefficient.x) %>%
  rename("Ridge Coefficient" = Coefficient.y)

coef.compare
```

                             Variable Lasso Coefficient Ridge Coefficient
    1              any_transplantsYes          8355.437          7955.626
    2         any_chronic_diseasesYes          2330.251          2275.360
    3  history_of_cancer_in_familyYes          2174.751          1989.369
    4       number_of_major_surgeries          -560.368          -386.252
    5                     diabetesYes          -454.892          -391.878
    6                             age           321.866           297.476
    7              known_allergiesYes           287.669           318.048
    8      blood_pressure_problemsYes            95.251           214.682
    9                          weight            68.252            65.860
    10                         height            -7.179            -8.211

The coefficients chosen by ridge and lasso give insights into which
variables are not significant. Coefficients close to 0 for ridge and
exactly 0 for lasso are not significant predictors of premium price,
while coefficients far from 0 are significant. None of the coefficients
are shrunk to 0 by lasso and none are shrunk close to 0 by ridge,
indicating that both methods find some significance in all the
variables.

### Principal Components Regression

Principal Components Regression builds linear combinations of variables
into directions that essentially serve as new features. PCR works well
with highly correlated variables and harnesses the most variation in the
first principal component. The second principal component is
perpendicular to the first and has the second most variation, and so on.
The optimal number of principal components has the lowest MSE and
explains the highest percentage of variability in the training data.
However, we must be cautious not to over fit the training data by using
too many principal components. The graph shows the training MSE for each
number of principal components and the summary shows the actual
percentages explained by each component.

``` r
pcr.model = pcr(premium_price ~., data=med.train, scale=TRUE, validation ="CV")
validationplot(pcr.model, val.type="MSEP", main = "Premium Price", ylab = "Mean Squared Error")
```

![](medicalpremium_files/figure-gfm/pc-1.png)<!-- -->

``` r
summary(pcr.model)
```

    Data:   X dimension: 740 10 
        Y dimension: 740 1
    Fit method: svdpc
    Number of components considered: 10

    VALIDATION: RMSEP
    Cross-validated using 10 random segments.
           (Intercept)  1 comps  2 comps  3 comps  4 comps  5 comps  6 comps
    CV            6196     5461     5470     5286     5205     4618     4627
    adjCV         6196     5461     5476     5252     5260     4609     4623
           7 comps  8 comps  9 comps  10 comps
    CV        4583     4546     4505      3736
    adjCV     4577     4546     4503      3733

    TRAINING: % variance explained
                   1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps
    X                17.75    29.85    40.96    51.81    62.28    71.39    79.91
    premium_price    22.46    22.47    29.05    29.11    45.30    45.37    47.11
                   8 comps  9 comps  10 comps
    X                87.63    94.83    100.00
    premium_price    47.79    48.96     64.79

The plot shows a drop in MSE at 5 components and another drop in MSE at
10 components. The percent variance of premium price explained by the
model increases from 29.11% to 45.30% when 5 versus 4 components are
used, and so we will build a model utilizing 5 principal component
directions. There is a flattening of the MSE graph from 5 to 9 principal
components and only about a 3% increase in variability explained by
additional components. This stagnancy suggests we do not get any
additional predictive power by adding directions. The percent variance
in premium price explained by 10 principal components increases from
48.96% (at 9 components) to 64.79% (at 10 components). Since we have 10
variables, using 10 principal components defeats the purpose of PCR
(decreasing dimensionality) so we will instead build a model with 9
components for performance comparison.

``` r
pcr.pred5.train = predict(pcr.model, med.train, ncomp = 5)
mse.pcr5.train = MSE(pcr.pred5.train)
r2.pcr5.train = rsquared(pcr.pred5.train)

pcr.pred5.test = predict(pcr.model, med.test, ncomp = 5)
mse.pcr5.test = MSE(pcr.pred5.test)
r2.pcr5.test = rsquared(pcr.pred5.test)

pcr.pred9.train = predict(pcr.model, med.train, ncomp = 9)
mse.pcr9.train = MSE(pcr.pred9.train)
r2.pcr9.train = rsquared(pcr.pred9.train)

pcr.pred9.test = predict(pcr.model, med.test, ncomp = 9)
mse.pcr9.test = MSE(pcr.pred9.test)
r2.pcr9.test = rsquared(pcr.pred9.test)
```

The PCR model with 5 components explains 45.3% of the variability of
premium price in the training set and the PCR model with 9 components
explains 48.96% of the corresponding quantity. The lower percent
variances explained by the two PCR models compared to the stepwise and
penalized linear regression models is likely due to low dimensionality
of the dataset and low correlation between the predictors. The MSE on
the training set for 5 components is 20939317 and for 9 components is
19539433, which are both higher than the previous models evaluated.

### Regression Trees

Regression decision trees are the final modeling method evaluated, and
four tree models will be built. Decision trees split observations into
partitions using an algorithm called binary recursive partitioning.
Binary recursive partitioning selects the variables at each split point
that maximizes the closeness of the response variable within partitions
and minimizes the similarity of the response between the two partitions.
This splitting process is continued until each partition reaches a
minimum size or the sum of the squared deviance from the mean in a
partition becomes zero (which indicates all observations in that node
have the same annual premium price). When the splitting process is
complete, these partitions become terminal nodes. For regression trees,
the average of the response variable is computed for each terminal node
and this average is assigned to each observation predicted to be in that
node (“Regression Trees”). Three decision tree techniques are evaluated
beyond a standard regression tree, and these techniques are: bagging,
random forest, and boosting. These additional methods combine weak
regression models into one strong regression tree by taking a multitude
of samples from the training set, building a separate model for each
sample, and combining their outputs.

#### Standard (Single) Regression Tree

``` r
medcost.model <- rpart(premium_price ~., data = med.train, method = "anova")
rpart.plot(medcost.model, main = "Prediction of Yearly Medical Coverage Costs", extra = 101, digits = -1, yesno = 2, type = 5)
```

![](medicalpremium_files/figure-gfm/treeplot-1.png)<!-- -->

``` r
pred.tree.train <- predict(medcost.model, med.train)
mse.tree.train <- MSE(pred.tree.train)
r2.tree.train <- rsquared(pred.tree.train)

pred.tree.test <- predict(medcost.model, med.test)
mse.tree.test <- MSE(pred.tree.test)
r2.tree.test <- rsquared(pred.tree.test)
```

This single regression tree was built considering all 10 predictor
variables at possible split points, but only 6 were actually included in
the model. Variables included are: age, chronic diseases, transplants,
family history of cancer, weight, and number of major surgeries. Age and
weight are split upon twice, indicating these two variables are
significant predictors of premium price. There are 9 terminal nodes, and
the tree diagram shows the mean annual premium price of each node, the
number of observations in each node, and the percent of the training set
that is predicted to be in that node. The regression tree output is
interpreted as follows: for an individual less than 30 years of age and
no chronic diseases, their predicted annual medical coverage cost is
16,000 rupees. There are 167 individuals with this classification (23%
of the train set). For an individual greater than or equal to 47 years
of age, no transplants, weight greater than or equal to 95 kilograms and
less than 2 major surgeries, their predicted yearly medical coverage
cost is 35,043 rupees. There are 23 individuals in the training set with
this classification (3%). This single regression tree explains 76.36% of
the variation in premium price of the training set, and the training MSE
is 9050352, which is the lowest MSE of all models evaluated thus far.

#### Bagging

Bagging randomly selects multiple samples with replacement from the
training set, builds a regression tree model for each sample, and
computes the average of the models to generate predictions. Bagging
considers all predictor variables at split points when building each
tree. 500 random samples are taken and a regression tree is built using
each sample, considering all 10 variables as possible split points. The
500 trees are separate models and will produce different price
predictions for each observation. Outputs are generated by computing the
mean premium price across all 500 weak models for each observation.

``` r
set.seed(11)
medcost.bag.model <- randomForest(premium_price ~., data = med.train, mtry = 10, importance = TRUE)

pred.bag.train <- predict(medcost.bag.model, med.train)
mse.bag.train <- MSE(pred.bag.train)
r2.bag.train <- rsquared(pred.bag.train)

pred.bag.test <- predict(medcost.bag.model, med.test)
mse.bag.test <- MSE(pred.bag.test)
r2.bag.test <- rsquared(pred.bag.test)
```

The Bagging model explains 79.71% of the variation in premium price and
the training MSE for this model is 1957190. The R squared for this model
is about 3% higher than that of the single regression tree and the MSE
for the bagging model is 7093163 lower than that of the standard
regression tree.

#### Random Forest

Random Forest is a variation of Bagging except only a subset of
predictor variables are considered at split points. The number of
variables randomly selected is determined by the total number of
predictor variables divided by 3. In this model, there are 10
independent variables and so 3 will be randomly selected for each weak
tree. 500 weak regression trees are generated by randomly selecting and
considering 3 variables for each tree and the average across the models
is computed for prediction.

``` r
set.seed(11)
medcost.rf.model <- randomForest(premium_price ~., data = med.train, mtry = 3, importance = TRUE)

pred.rf.train <- predict(medcost.rf.model, med.train)
mse.rf.train <- MSE(pred.rf.train)
r2.rf.train <- rsquared(pred.rf.train)

pred.rf.test <- predict(medcost.rf.model, med.test)
mse.rf.test <- MSE(pred.rf.test)
r2.rf.test <- rsquared(pred.rf.test)
```

77.73% of the variance of premium price was explained by the random
forest model, which is slightly lower than the bagging model. The MSE on
the training set is 2824946, which is higher compared to bagging but
still an improvement from the single regression tree.

``` r
imp <- data.frame(importance(medcost.rf.model, type =1))
imp <- rownames_to_column(imp, var = "variable")
ggplot(imp, aes(x=reorder(variable, X.IncMSE), y=X.IncMSE, color=reorder(variable, X.IncMSE))) +
  geom_point(show.legend=FALSE, size=3) +
  geom_segment(aes(x=variable, xend=variable, y=0, yend=X.IncMSE), size=3, show.legend=FALSE) +
  xlab("") +
  ylab("% Increase in MSE") +
  labs(title = "Variable Importance for Prediction of Premium Price") +
  coord_flip() +
  scale_color_manual(values = colorRampPalette(brewer.pal(1,"Purples"))(10)) +
  theme_classic()
```

![](medicalpremium_files/figure-gfm/varimp-1.png)<!-- -->

``` r
imp %>%
  arrange(desc(X.IncMSE)) %>%
  rename(`% Increase in MSE` = X.IncMSE)
```

                          variable % Increase in MSE
    1                          age          114.0435
    2              any_transplants           47.2265
    3         any_chronic_diseases           24.7475
    4    number_of_major_surgeries           24.6255
    5                       weight           23.3641
    6  history_of_cancer_in_family           19.9419
    7      blood_pressure_problems           13.5772
    8                     diabetes            2.1337
    9                       height            0.8325
    10             known_allergies           -0.1025

The plot and table show the importance of each variable in predicting
premium price across the 500 trees. The variable importance is
calculated by the average percent increase in mean squared error across
all 500 trees when each variable is not included in the model. Age is by
far the most important variable in predicting medical coverage costs,
followed by transplants, chronic diseases, number of major surgeries,
weight, family history of cancer, and blood pressure problems. There is
over a 100% increase in MSE when premium price is predicted without age
in the model, which highlights the influence of a person’s age in their
medical coverage costs. There is about a zero percent increase in MSE
when known allergies and height are excluded from the model, which
suggests these variables are not significant in predicting health care
coverage.

#### Boosting

Boosting sequentially builds tree models with a concentration on
observations with inaccurate predictions in previous models by assigning
increased weights to inaccurate predictions and decreased weights to
accurate predictions. As a result, successive regression models are
forced to put more learning emphasis on observations with inaccurate
predictions. Boosting then calculates the weighted average among all
models and this weighted average is the generated prediction for that
particular observation.

``` r
set.seed(11)
medcost.boost.model <- gbm(premium_price ~., data = med.train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)

pred.boost.train <- predict(medcost.boost.model, med.train)
```

    Using 5000 trees...

``` r
mse.boost.train <- MSE(pred.boost.train)
r2.boost.train <- rsquared(pred.boost.train)

pred.boost.test <- predict(medcost.boost.model, med.test)
```

    Using 5000 trees...

``` r
mse.boost.test <- MSE(pred.boost.test)
r2.boost.test <- rsquared(pred.boost.test)
```

5000 successive tree models were built using boosting and the training
r-squared is 0.996, which is the highest of all models thus far. The
training MSE is 153033, which is by far the lowest train MSE of all ten
models. However, boosting may be over fitting the training data and the
accuracy of the boosting model may not be generalizable to the test set.

``` r
df <- data.frame(summary(medcost.boost.model, plotit = FALSE), row.names = NULL)
ggplot(df, aes(x=reorder(var,rel.inf), y=rel.inf, fill=reorder(var,rel.inf))) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Relative Influence of Variables in Predicting Premium Price") +
  ylab("Relative Influence") +
  xlab("") +
  scale_fill_manual(values = colorRampPalette(brewer.pal(1,"Oranges"))(10)) +
  theme_classic()
```

![](medicalpremium_files/figure-gfm/relinf-1.png)<!-- -->

``` r
df %>%
  rename(Variable = var) %>%
  rename(`Relative Influence` = rel.inf)
```

                          Variable Relative Influence
    1                          age             48.919
    2                       weight             19.421
    3                       height             12.626
    4              any_transplants              7.876
    5      blood_pressure_problems              2.541
    6    number_of_major_surgeries              2.230
    7         any_chronic_diseases              2.223
    8  history_of_cancer_in_family              1.718
    9                     diabetes              1.358
    10             known_allergies              1.088

This method gives insight to the relative influence of each variable on
predicted premium price. The relative influence is calculated by whether
or not a variable was selected to split upon, and how much the mean
squared error decreased for prediction of premium price by splitting on
this particular variable. The results for each variable’s calculated
importance across the 5000 tree models are shown by the plot. Given the
adjusted weights for inaccurate predictions that boosting implements,
the most important predictors are similar but not the same as random
forest. The variables with the highest relative influence are: age,
weight, height, and any transplants. The table gives the normalized
increase in MSE when each variable is excluded from the model, and can
be interpreted as such: the squared error of predicted premium price and
actual premium price increases by 48.9% on average when age is not
included in the model. If weight is excluded from the model, there is a
19.4% average increase in squared error and if height is excluded from
the model there is a 12.6% increase.

### Results on Test Set

All calculations up to this point have been done using the training set.
Now we will measure and compare each model’s performance on the test
set. The scatter plots below show the actual premium price versus the
predicted premium price for each model. The identity line y=x provides
reference for perfect predictions; points on this line indicate an exact
prediction of premium price for an individual using that model.
Therefore, models with points closest to this line have the lowest mean
squared error for the test set. The table provides each models’ MSE and
is organized in ascending order, with the best performing models on top
and the worst performing models on the bottom.

``` r
p1 <- ggplot(mapping = aes(x = med.test$premium_price, y = pred.bag.test)) +
  geom_point(color = "#FFB5C5") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Bagging")

p2 <- ggplot(mapping = aes(x = med.test$premium_price, y = pred.rf.test)) +
  geom_point(color = "#BF87B3") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Random Forest")

p3 <- ggplot(mapping = aes(x = med.test$premium_price, y = pred.boost.test)) +
  geom_point(color = "#7F5AA2") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Boosting")

p4 <- ggplot(mapping = aes(x = med.test$premium_price, y = pred.tree.test)) +
  geom_point(color = "#000080") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Standard Regression Tree")

p5 <- ggplot(mapping = aes(x = med.test$premium_price, y = fwd.pred.test)) +
  geom_point(color = "turquoise") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Forward Selection")

p6 <- ggplot(mapping = aes(x = med.test$premium_price, y = bwd.pred.test)) +
  geom_point(color = "purple") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Backward Selection")

p7 <- ggplot(mapping = aes(x = med.test$premium_price, y = ridge.pred.test)) +
  geom_point(color = "pink") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Ridge Regression")

p8 <- ggplot(mapping = aes(x = med.test$premium_price, y = lasso.pred.test)) +
  geom_point(color = "steelblue4") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Lasso")

p9 <- ggplot(mapping = aes(x = med.test$premium_price, y = pcr.pred5.test)) +
  geom_point(color = "salmon1") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "PCR (5 directions)")

p10 <- ggplot(mapping = aes(x = med.test$premium_price, y = pcr.pred9.test)) +
  geom_point(color = "palegreen3") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "PCR (9 directions)")

grid.arrange(p1, p2, p3, p4, nrow=2)
```

![](medicalpremium_files/figure-gfm/complinear-1.png)<!-- -->

``` r
grid.arrange(p5, p6, p7, p8, p9, p10, nrow=3)
```

![](medicalpremium_files/figure-gfm/complinear-2.png)<!-- -->

``` r
mse.df <- data.frame(rbind(c("Forward Selection", mse.fwd.test), c("Backward Selection", mse.bwd.test), c("Ridge Regression", mse.ridge.test), c("Lasso Regression", mse.lasso.test), c("PCR (5 directions)", mse.pcr5.test), c("PCR (9 directions)", mse.pcr9.test), c("Single Regression Tree", mse.tree.test), c("Bagging (10 variables)", mse.bag.test), c("Random Forest", mse.rf.test), c("Boosting", mse.boost.test)))

mse.df <- mse.df %>%
  mutate(X2 = as.numeric(X2)) %>%
  arrange(-desc(X2)) %>%
  rename("Regression Method" = X1, "Mean Squared Error" = X2)

mse.df
```

            Regression Method Mean Squared Error
    1  Bagging (10 variables)            8221854
    2  Single Regression Tree            9621461
    3           Random Forest            9643261
    4                Boosting           13057381
    5      Backward Selection           15403000
    6        Lasso Regression           15414378
    7       Forward Selection           15418784
    8        Ridge Regression           15600128
    9      PCR (9 directions)           21542633
    10     PCR (5 directions)           23129094

The four regression trees all performed better than the stepwise,
penalized, and PCR models. Bagging was the best model for predicting
premium price, with an MSE of 8221854, while PCR with 5 components was
the worst model with an MSE of 23129094. Boosting performed the worst of
the four regression trees but still better than the other 6 regression
models. The test MSE for boosting is 13057381 which is significantly
worse than training MSE for this model, indicating that boosting overfit
the training data. The best regression model outside the tree models was
backward selection, with a test MSE of 15403000.

``` r
mse.compare <- ggplot(mse.df, aes(x=reorder(`Regression Method`, `Mean Squared Error`), y=`Mean Squared Error`, fill=`Regression Method`)) +
  geom_col() +
  xlab("Regression Method") +
  theme(axis.text.x = element_blank()) +
  scale_fill_discrete(limits = mse.df$`Regression Method`) +
  labs(title = "Comparison of MSE Across All 10 Models")

mse.compare
```

![](medicalpremium_files/figure-gfm/msegraph-1.png)<!-- -->

The bar chart above provides a visual comparison of the MSE across the
ten models. We see a similar MSE between bagging, the single regression
tree, and random forest but a higher MSE for boosting. The test MSE is
similar for backward selection, forward selection, lasso regression, and
ridge regression. There is a big jump in MSE for the PCR models with 5
and 9 components.

``` r
rsquared.df <- data.frame(rbind(c("Forward Selection", r2.fwd.test), c("Backward Selection", r2.bwd.test), c("Ridge Regression", r2.ridge.test), c("Lasso Regression", r2.lasso.test), c("PCR (5 directions)", r2.pcr5.test), c("PCR (9 directions)", r2.pcr9.test), c("Single Regression Tree", r2.tree.test), c("Bagging (10 variables)", r2.bag.test), c("Random Forest", r2.rf.test), c("Boosting", r2.boost.test)))

rsquared.df <- rsquared.df %>%
  mutate(X2 = as.numeric(X2)) %>%
  arrange(desc(X2)) %>%
  rename("Regression Method" = X1, "R Squared" = X2)

rsquared.df
```

            Regression Method R Squared
    1  Bagging (10 variables)    0.7994
    2  Single Regression Tree    0.7652
    3           Random Forest    0.7647
    4                Boosting    0.6814
    5      Backward Selection    0.6241
    6        Lasso Regression    0.6238
    7       Forward Selection    0.6237
    8        Ridge Regression    0.6193
    9      PCR (9 directions)    0.4743
    10     PCR (5 directions)    0.4356

The table above provides the test set r-squared for each method,
arranged in descending order. As expected, the bagging model explains
the highest percentage of premium price variability (79.94%) and the PCR
model with 5 components explains the lowest variability (43.56%).

### Conclusion

In conclusion, a variety of regression techniques can be useful in
prediction of individuals’ yearly medical coverage costs. The best
performing models are generalizable to the test set and provide insights
to the most important predictors. Regression trees proved to be the best
modeling technique for this dataset and bagging in particular had the
highest prediction accuracy for the test set. Further analysis could be
done by tuning the bagging model parameters, such as the number of trees
built. The most significant predictors of premium price ended up being
age, having transplants, number of major surgeries, weight, family
history of cancer, and chronic diseases.

### References

California Health Insurance. “History of Health Insurance and 2019 &
Beyond Projections.” Health for California Insurance Center. Mar 5
2019.https://www.healthforcalifornia.com/blog/history-of-health-insurance

Porretta, Anna. “How Much Does Individual Health Insurance Cost?”
eHealth. Nov 5 2021.
<https://www.ehealthinsurance.com/resources/individual-and-family/how>
much-does-individual-health-insurance-cost

“Regression Trees.” Frontline Solvers. 2021.
<https://www.solver.com/regression> trees

Tejashvi. “Medical Insurance Premium Prediction.” Kaggle. Aug 2021.
<https://www.kaggle.com/tejashvi14/medical-insurance-premium-prediction>?
select=Medicalpremium.csv
