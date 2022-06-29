import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

# Read csv file data into dataframe "data"
data = pd.read_csv("winequality.csv")

# Fill in missing values
data.fillna(data.mean(), inplace=True)

# Declare predictors that I determined have the greatest effect on wine quality
predictors = ['fixed acidity','citric acid', 'alcohol','density','pH','residual sugar']

#set up target, predictors, and split the training/testing partitions
x = data[predictors]
y = data['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Normalize data
scaler = StandardScaler()

StandardScaler(copy=True, with_mean=True, with_std=True)

# Apply the transformations to the data
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Create the first neural network
network = MLPRegressor(hidden_layer_sizes=(15), solver='lbfgs', max_iter=1000)
network.fit(x_train, y_train)

# Test the model against the new model and calculate accuracy
pred = network.predict(x_test)
# Find mean absolute error
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print()

# Create second neural network model using logistic activation function and default solver
network = MLPRegressor(hidden_layer_sizes=(15), activation='logistic', max_iter=1000)
network.fit(x_train, y_train)
pred = network.predict(x_test)
# Find mean absolute error
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print()




def pcaAnalysis(components):
    pca = PCA(n_components = components)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2', 'principal component 3',
                                          'principal component 4'])
    return principalDf
