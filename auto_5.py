import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
import requests as re
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
# Suppress warnings (optional)
warnings.filterwarnings('ignore')

print('---- Download the data ----')
def download(url, filename):
    """Downloads a file if it does not already exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = re.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
                print(f"File '{filename}' downloaded successfully.")
        else: 
            print(f"Failed to download file. Status code: {response.status_code}")
    else: 
        print(f"File '{filename}' already exists. Skipping download.")

# Download the dataset
download('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv', 'module_5_auto.csv')

# Load the dataset
df = pd.read_csv("module_5_auto.csv", header=0)

# Keep only numeric data
df = df._get_numeric_data()

# Drop unnecessary columns
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

# Display the first 5 rows
print(df.head(5))

"""===== Functions for Plotting ====="""
def DistributionPlot (RedFunction,BlueFunction,RedName,BlueName,Title):
    plt.figure(figsize=(8,6))
    ax1=sns.kdeplot(RedFunction,color='r',label=RedName)
    ax2=sns.kdeplot(BlueFunction,color='b',label=BlueName,ax=ax1)
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.legend()
    plt.show()

def PollyPlot (xtrain, xtest, y_train, y_test, lr,poly_transform):
    plt.figure(figsize=(8,6))
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object
    xmax=max([xtrain.values.max(), xtest.values.max()])
    xmin=min([xtrain.values.min(),xtest.values.min()])
    x = np.arange(xmin, xmax, 0.1)
    plt.plot(xtrain,y_train,'ro',label='Training data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.ylim([-10000, 60000])
    plt.xlabel('Horsepower', fontsize=12)
    plt.ylabel('Price')
    plt.title('Polynomial Regression: Actual vs Predicted', fontsize=14)
    plt.legend()
    plt.show()

"""=== Training and Testing ==="""
print('---- train_test_split ----')

y_data = df['price']
x_data = df.drop('price',axis=1)
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.10, random_state=0)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


"""=== LinearRegression ==="""
print('---- Simple Linear Regression ----')
lm  = LinearRegression()
#We fit the model using the feature "horsepower"
lm.fit(x_train[['horsepower']],y_train)
#Let's calculate the R^2 on the test data and train data
lm.score(x_test[['horsepower']],y_test)
lm.score(x_train[['horsepower']],y_train)
print("Test R²:", lm.score(x_test[['horsepower']], y_test))
print("Train R²:", lm.score(x_train[['horsepower']], y_train))

"""==== Cross-Validation Score ===="""
Rcross = cross_val_score(lm,x_data[['horsepower']],y_data,cv=4)
Rcross
print ( 'the mean of folds are : ',Rcross.mean(),"and the standard deviation is",Rcross.std() )
y_hat=cross_val_predict(lm,x_data[['horsepower']],y_data,cv=4)
print("First 5 predictions:",y_hat[0:5])


"""====  Overfitting, Underfitting and Model Selection ===="""
print('---- Multiple Linear Regression ----')
#We create Multiple Linear Regression objects and train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features
lm.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_train)
yhat_train = lm.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print("First 5 predictions using training data:",yhat_train[0:5])
yhat_test = lm.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print("First 5 predictions using test data :",yhat_test[0:5])
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)


"""==== Overfitting with polynomial regression ===="""
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.45, random_state=0)
pr = PolynomialFeatures (degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr
#let's create a Linear Regression model "poly" and train it.
poly=LinearRegression()
poly.fit(x_train_pr,y_train)
yhat = poly.predict(x_test_pr)
print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)
PollyPlot(x_train['horsepower'],x_test['horsepower'],y_train,y_test,poly,pr)
#The lower the R^2, the worse the model. A negative R^2 is a sign of overfitting.
print(poly.score(x_train_pr,y_train))
print(poly.score(x_test_pr,y_test))

#Let's see how the R^2 changes on the test data for different order polynomials and then plot the results
Rsq_test = [] ## Empty list to store R² scores for test data
order = [1,2,3,4] # Polynomial degrees
for n in order  : 
    # Create NEW polynomial transformer for each degree
    pf = PolynomialFeatures(degree = n)
    x_train_pf = pf.fit_transform(x_train[['horsepower']])
    x_test_pf = pf.fit_transform(x_test[['horsepower']])
    lm.fit(x_train_pf,y_train)
    Rsq_test.append(lm.score(x_test_pf, y_test))
plt.plot(order, Rsq_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.73, 'Maximum R^2 ') 
plt.show()

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

"""==== Ridge Regression ===="""
pr=PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

RigeModel = Ridge(alpha=1)
RigeModel.fit(x_train_pr,y_train)
yhat = RigeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)


Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)
width = 8
height = 6
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()