# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:48:32 2018

@author: Byram
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
affairdata=sm.datasets.fair.load_pandas().data
affairdata["affair"]=(affairdata.affairs>0).astype(int)       

affairdata.groupby("rate_marriage").mean()



affairdata.educ.hist()
plt.title("Education Histogram")
plt.xlabel("education level")
plt.ylabel("frequency")


affairdata.rate_marriage.hist()
plt.xlabel("marriage rationg")
plt.ylabel("frequency")


#Let's take a look at the distribution of marriage ratings for those having affairs versus those not having affairs.
pd.crosstab(affairdata.rate_marriage,affairdata.affair.astype(bool)).plot(kind='bar')
plt.title('marriage rating distribution with respect to affairs')
plt.xlabel("marriage rating")
plt.ylabel('frequency')


#Let's use a stacked barplot to look at the percentage of women having affairs by number of years of marriage.
affairs_yrs_married=pd.crosstab(affairdata.yrs_married, affairdata.affair.astype(bool))
affairs_yrs_married.div(affairs_yrs_married.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True)
plt.title("affair percentage with respect to years married")
plt.xlabel('years married')
plt.ylabel('percentage')

y,x=dmatrices('affair~rate_marriage+age+yrs_married+children+\
               religious+educ+C(occupation)+C(occupation_husb)',
               affairdata,return_type="dataframe")
print(x.columns)


x=x.rename(columns={'C(occupation)[T.2.0]':"occ2",'C(occupation)[T.3.0]':'occ3',
                     'C(occupation)[T.4.0]':'occ4', 'C(occupation)[T.5.0]':'occ5', 
                     'C(occupation)[T.6.0]':'occ6',
                     'C(occupation_husb)[T.2.0]':'occh2', 'C(occupation_husb)[T.3.0]':'occh3',
                     'C(occupation_husb)[T.4.0]':'occh4', 'C(occupation_husb)[T.5.0]':'occh5',
                     'C(occupation_husb)[T.6.0]':'occh6'})

#We also need to flatten y into a 1-D array, so that scikit-learn will properly understand it as the response 
y=np.ravel(y)

#Logistic Regression
model=LogisticRegression()
model=model.fit(x,y)
#accuracy on training set
model.score(x,y)

#null error rate
y.mean()
#Only 32% of the women had affairs, which means that we could obtain 68% accuracy by always predicting "no". 

#examine the coefficients
pd.DataFrame(list(zip(x.columns,np.transpose(model.coef_))))

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
mod2=LogisticRegression()
mod2.fit(x_train,y_train)

#We now need to predict class labels for the test set. We will also generate the class probabilities
predicted=mod2.predict(x_test)
print(predicted)

#generate class probabilties
probab=mod2.predict_proba(x_test)
print(probab)

#evalutaion metrics
print(metrics.accuracy_score(y_test,predicted))
print(metrics.confusion_matrix(y_test,predicted))
print(metrics.classification_report(y_test,predicted))

