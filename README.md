# Medical Premium Price Prediction in R 
### Goals: 
- Train and compare mutiple machine learning regression models that predict individuals' medical insurance premium prices. 
- Evaluate model parameters and accuracy measurements across the machine learning techniques used.

### How to view: `medicalpremium.md`
- provides in-depth descriptions and visualizations of the machine learning techniques outlined below
- exploratory analysis of the predictor variables
- interpretation of model estimates and outputs
- description and citation of the data set

### Overview: 
10 regression models are built and their parameters tuned using the train set:
- Forward stepwise linear regression 
- Backward stepwise linear regression 
- Ridge penalized linear regression 
- Lasso penalized linear regression 
- Principal components (5 directions) 
- Principal components (9 directions)
- Standard decision tree
- Bagging decision tree
- Random forest decision tree
- Boosting decision tree

Each model's performance is measured and compared using mean squared error (MSE) and the correlation coefficients (r-squared)
- see written functions `rsquared` and `MSE` for how these values are computed 
- see graph `mse.compare` for comparison of each model's MSE on the test set
- see table `rsquared.df` for comparison of each model's correlation coefficients 
