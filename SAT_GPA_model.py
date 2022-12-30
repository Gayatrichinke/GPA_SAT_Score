import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

sgpa = pd.read_csv(r"C:\Users\chink\Desktop\ML CA2\SLR_SAT_GPA.csv")
sgpa.describe()
sgpa.info()
sgpa.isna().sum()
sgpa.duplicated().sum()



import matplotlib.pyplot as plt # mostly used for visualization purposes

plt.bar(height = sgpa.GPA, x = np.arange(1, 200, 1))
plt.hist(sgpa.GPA) #histogram
plt.title('GPA')

plt.boxplot(sgpa.GPA) #boxplot
plt.title('GPA')

plt.bar(height = sgpa.SAT_Scores, x = np.arange(1, 200, 1))
plt.hist(sgpa.SAT_Scores) #histogram
plt.title('SAT_Scores')
plt.boxplot(sgpa.SAT_Scores) #boxplot
plt.title('SAT_Scores')


# correlation
np.corrcoef(sgpa.SAT_Scores, sgpa.GPA)

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly.
# Function for calculating a covariance matrix called cov()
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(sgpa.SAT_Scores, sgpa.GPA)[0, 1]
cov_output

plt.scatter(x = sgpa['SAT_Scores'], y = sgpa['GPA'], color = 'green')
plt.xlabel('SAT_Scores')
plt.ylabel('GPA')

# normalization
def norm_func(i):
    x = (i-i.min()) / (i.max()-i.min())
    return (x)

#normalization data frame (considering the numerical part of data)

df_norm = norm_func(sgpa.iloc[:, :])
df_norm.describe()

 #Scatter plot
plt.scatter(x = df_norm['SAT_Scores'], y = df_norm['GPA'], color = 'green')
 # no correlation in the data as it is distributed
 
 
 #Square root transformatiion

sgpa.insert(len(sgpa.columns), 'A_GPA',
          np.sqrt(sgpa.iloc[:,1:]))

 sgpa.insert(len(sgpa.columns), '1_SAT',
          np.sqrt(sgpa.iloc[:,0]))

plt.scatter(x = sgpa['SAT_Scores'], y = sgpa['A_GPA'], color = 'red')
plt.scatter(x = sgpa['1_SAT'], y = sgpa['GPA'], color = 'red')
plt.scatter(x = sgpa['1_SAT'], y = sgpa['A_GPA'], color = 'red')

 
 
# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('GPA ~ SAT_Scores', data = sgpa).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(sgpa['SAT_Scores']))

# Regression Line
plt.scatter(sgpa.SAT_Scores, sgpa.GPA)
plt.plot(sgpa.SAT_Scores, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = sgpa.GPA - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

# error = 0.5159457227723683

######### Model building on Transformed Data
# Log Transformation


plt.scatter(x = np.log(sgpa['SAT_Scores']), y = sgpa['GPA'], color = 'brown')
np.corrcoef(np.log(sgpa.SAT_Scores), sgpa.GPA) #correlation

model2 = smf.ols('GPA ~ np.log(SAT_Scores)', data = sgpa).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sgpa['SAT_Scores']))

# Regression Line
plt.scatter(np.log(sgpa.SAT_Scores), sgpa.GPA)
plt.plot(np.log(sgpa.SAT_Scores), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = sgpa.A_GPA - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# error = 1.1835916295131899

#### Exponential transformation


plt.scatter(x = sgpa['SAT_Scores'], y = np.log(sgpa['GPA']), color = 'orange')
np.corrcoef(sgpa.SAT_Scores, np.log(sgpa.GPA)) #correlation

model3 = smf.ols('np.log(GPA) ~ SAT_Scores', data = sgpa).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(sgpa['SAT_Scores']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(sgpa.SAT_Scores, np.log(sgpa.GPA))
plt.plot(sgpa.SAT_Scores, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res3 = sgpa.GPA - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

# error = 0.5175875893834132

#### Polynomial transformation


model4 = smf.ols('np.log(GPA) ~ SAT_Scores + I(SAT_Scores*SAT_Scores)', data = sgpa).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(sgpa))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = sgpa.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(sgpa.SAT_Scores, np.log(sgpa.GPA))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = sgpa.GPA - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# error = 0.5144912487746159

data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

from sklearn.model_selection import train_test_split

train, test = train_test_split(sgpa, test_size = 0.2)

finalmodel = smf.ols('np.log(GPA) ~ SAT_Scores', data = train).fit()
finalmodel.summary()




# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_GPA = np.exp(test_pred)
pred_test_GPA

# Model Evaluation on Test data
test_res = test.GPA - pred_test_GPA
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# error = 0.5537200187927894

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_GPA = np.exp(train_pred)
pred_train_GPA

# Model Evaluation on train data
train_res = train.GPA - pred_train_GPA
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

#Error = 0.5050285794889308



