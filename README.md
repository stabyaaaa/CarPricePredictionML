Brief Summary

The project on Car Price Prediction has been implemented covering the requirements of the car company. 8129 sample of data were given by the client at the beginning, although some of the sample contained impure values. Some of initial details of the project has been listed below:

Number of raw sample given: 8129

Strong features: So, according to the correlation matrix, we can clearly assume that maximum power and engine size has major relation with the target variable. So, as the engine size and max power increase, the predicted price of the car should also increase. But, if we take a real world example, mileage and km_driven plays a vital role in reducing car price which is also negatively correlated in the above correlation matrix. Talking about the most important and non important features amongst the 12 features, year, owner, transmission, seats, fuel, and torque have been seen as the least important feature for this project. AS the owner of this company doesn't been comfortable with torque, their request to completely dropping the features has been fulfilled.

Algorithm Used: Random Forest with mean squared error of 0.109 Linear Regression : Mean: -0.34412598164746566 SVR - Score: Mean: -0.29885112810265557 KNeighbors Regressor: Mean: -0.13635452247177193 Decision-Tree Regressor : Mean: -0.11364628259997692 Random-Forest Regressor :Mean: -0.10975895232915081

Best MSE for grid search : -.11

Final mse using test split : 0.11
