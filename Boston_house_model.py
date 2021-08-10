from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
house=datasets.load_boston()
train_x,test_x,train_y,test_y=train_test_split(house.data,house.target,test_size=0.2,random_state=1)
linear=LinearRegression()
linear.fit(train_x,train_y)
predict_y=linear.predict(test_x)
#basic sklearn functions using sckit learn library . 
