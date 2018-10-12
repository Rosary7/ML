# ML Model- Linear_Regression - Using "scikit-learn" library

"""
- Developed using Spyder from Anaconda
- Using "scikit-learn" library (http://scikit-learn.org/stable/)
- "scikit-learn" requires Python (>= 2.7 or >= 3.4),NumPy (>= 1.8.2) and SciPy (>= 0.13.3)
- Anaconda has 'scikit-learn' by  default.
- Anaconda has NumPy and SciPy, by  default.

About "scikit-learn" library: 
-It has tools for data mining and data analysis
-it provides easy to use functions and predefined models which saves a lot of time
-Built on NumPy, SciPy, and matplotlib
-Open source, commercially usable - BSD license
-http://scikit-learn.org/stable/

"""
# Import "scikit-learn" library
from random import randint
from sklearn.linear_model import LinearRegression

# Generate Training Set
TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100
TRAIN_INPUT = list()
TRAIN_OUTPUT = list()
for i in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = a + (2*b) + (3*c)
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)

# Train The Model
predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

# Test Data
X_TEST = [[10, 20, 30]]
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_

print('TRAIN_INPUT:' , TRAIN_INPUT)
print()
print('TRAIN_OUTPUT:' , TRAIN_OUTPUT)
print()
print('outcome:' , outcome)
print('coefficients: ' , coefficients)


