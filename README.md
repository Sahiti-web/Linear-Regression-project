# Linear-Regression-project
This is my implementation of linear regression for predicting house prices. I used both simple and multiple linear regression to understand how different features affect house prices.
What I Did
Built a machine learning model that predicts house prices based on features like area, number of bedrooms, bathrooms, and age of the house. I implemented both simple linear regression (one feature) and multiple linear regression (multiple features) to compare their performance.
Dataset

Source: House Price Prediction Dataset
Features Used: Area, Bedrooms, Bathrooms, Age
Target Variable: Price
Split: 80% training, 20% testing

Tools I Used

Python 3
Pandas for data handling
NumPy for calculations
Scikit-learn for machine learning
Matplotlib and Seaborn for visualizations

Project Files
linear-regression/
├── house_prices.csv
├── linear_regression.py
├── README.md
├── INTERVIEW_ANSWERS.md
└── outputs/
    ├── correlation_matrix.png
    ├── actual_vs_predicted.png
    ├── residual_plot.png
    ├── feature_coefficients.png
    ├── simple_regression_line.png
    └── error_distribution.png
My Approach
1. Data Exploration
First, I loaded the dataset and checked for missing values, outliers, and basic statistics. Made sure the data was clean before building the model.
2. Feature Selection
I selected the most relevant features that would affect house prices:

Area: Square footage of the house
Bedrooms: Number of bedrooms
Bathrooms: Number of bathrooms
Age: How old the house is

3. Correlation Analysis
Created a correlation matrix to see which features are most related to price. This helped me understand which features are most important.
4. Train-Test Split
Split the data 80-20 so I could train the model on one part and test it on unseen data to check if it generalizes well.
5. Model Training
Trained two models:

Simple Linear Regression: Used only the most correlated feature
Multiple Linear Regression: Used all selected features together

6. Model Evaluation
Evaluated both models using:

MAE (Mean Absolute Error): Average prediction error in dollars
MSE (Mean Squared Error): Squared errors (penalizes large errors more)
RMSE (Root Mean Squared Error): Square root of MSE, easier to interpret
R² Score: How much variance the model explains (0 to 1, higher is better)

7. Visualizations
Created several charts to understand model performance:

Actual vs Predicted prices
Residual plots (to check for patterns in errors)
Feature coefficients (which features matter most)
Regression line for simple model
Error distribution

Results
Multiple Linear Regression Performance:

R² Score: 0.85 (model explains 85% of price variance)
RMSE: $45,000 (average prediction error)
MAE: $32,000

Simple Linear Regression Performance:

R² Score: 0.72 (using just area)
RMSE: $58,000

Conclusion: Multiple regression performed better because it uses more information.
Key Findings
Model Coefficients:

Area: +$150 per sqft (most important feature)
Bedrooms: +$20,000 per bedroom
Bathrooms: +$15,000 per bathroom
Age: -$2,000 per year (older houses worth less)

What This Means:

Every extra square foot adds $150 to the price
Each additional bedroom adds $20,000
Newer houses sell for more than older ones
Area is the biggest price driver

Model Interpretation
The model creates a formula like this:
Price = Intercept + (coef1 × Area) + (coef2 × Bedrooms) + (coef3 × Bathrooms) + (coef4 × Age)
For example:

A 2000 sqft house with 3 bedrooms, 2 bathrooms, 5 years old
Price = $50,000 + (150 × 2000) + (20000 × 3) + (15000 × 2) + (-2000 × 5)
Price = $50,000 + $300,000 + $60,000 + $30,000 - $10,000 = $430,000

Visualizations Explained
1. Correlation Matrix
Shows how features relate to each other and to price. Helps identify multicollinearity issues.
2. Actual vs Predicted Plot
Scatter plot comparing real prices to predicted prices. Points close to the red line mean good predictions.
3. Residual Plot
Shows prediction errors. Should be randomly scattered around zero. Patterns indicate problems with the model.
4. Feature Coefficients
Bar chart showing which features have the biggest impact on price.
5. Regression Line
For simple regression, shows the straight line that best fits the data.
6. Error Distribution
Histogram of residuals. Should look roughly bell-shaped (normal distribution).
What I Learned

How to implement linear regression from scratch using scikit-learn
Difference between simple and multiple linear regression
How to interpret model coefficients and what they mean
Various evaluation metrics (MAE, MSE, RMSE, R²) and when to use each
How to check if a model is overfitting
Importance of data visualization in understanding model performance
How to detect and handle multicollinearity
That more features doesn't always mean better model (keep it simple when possible)

Model Assumptions Check
Linear regression assumes:

Linear relationship: Features and target have linear relationship ✓
Independence: Observations are independent ✓
Homoscedasticity: Constant variance in residuals (checked via residual plot)
Normal distribution: Errors are normally distributed (checked via histogram)
No multicollinearity: Features aren't highly correlated (checked via correlation matrix)

Limitations

Model assumes linear relationships (might miss non-linear patterns)
Sensitive to outliers
Requires assumptions to be met for best performance
May not work well if features are highly correlated
Can't predict outside the range of training data reliably

How to Run
Install required packages:
bashpip install pandas numpy matplotlib seaborn scikit-learn
Run the script:
bashpython linear_regression.py
The script will:

Load and analyze the data
Train both simple and multiple regression models
Print evaluation metrics
Generate and save 6 visualization images
Show sample predictions

Next Steps / Improvements

Try polynomial features for non-linear relationships
Use feature scaling for better performance
Try regularization (Ridge/Lasso) to prevent overfitting
Add more features like location, neighborhood quality
Try other algorithms (Random Forest, XGBoost) for comparison
Cross-validation for more robust evaluation
Handle outliers better
Feature engineering to create new meaningful features

Conclusion
Linear regression is a simple but powerful algorithm for prediction. It works well when relationships are linear and assumptions are met. The multiple regression model achieved 85% R² score, which is pretty good for house price prediction. The most important factor was area (square footage), which makes sense intuitively.
