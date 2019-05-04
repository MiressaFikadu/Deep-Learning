
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
seed=7
np.random.seed(seed)
path = "D:\iris.csv"
columns = ['SepalLength','SepalWidth','PetalLength', 'PetalWidth','class']
dataset = pd. read_csv(path, names=columns)
print(dataset.shape)
print(dataset.head(60))
#visualize
dataset.hist()
plt.show()
 #density
dataset.plot(kind='density', subplots=True, layout=(2,2), sharex=False, legend=False,fontsize=1)
plt.show()

sns.lmplot("SepalLength", "SepalWidth",
           data=dataset,
           fit_reg=True,
           hue="class",
           scatter_kws={"marker": "D",
                        "s": 20})

plt.show()
sns.lmplot("PetalLength", "PetalWidth",
           data=dataset,
           fit_reg=True,
           hue="class",
           scatter_kws={"marker": "D",
                        "s": 20})

plt.show()
#input and output
data=dataset.values
X= data[:,0:4].astype(float)
y=data[:,4]

#convert class to integers and one-hot encoding
encoder = LabelBinarizer()
encoder.fit(y)
y_en= encoder.fit_transform(y)
#or
encoder=LabelEncoder()
encoder.fit(y)
y_b= encoder.fit_transform(y)
print(y_b)
dummy_y=np_utils.to_categorical(y_b)
print(dummy_y)

X_train, X_test, y_train,y_test= train_test_split(X,y, test_size=0.20, random_state=seed)

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, solver='adam', verbose=0, random_state=seed,learning_rate_init=.1)
mlp.fit(X_train,y_train)

print("Training_set_accuracy: %f" % mlp.score(X_train, y_train))
print("Test_set_accuracy: %f" % mlp.score(X_test, y_test))

