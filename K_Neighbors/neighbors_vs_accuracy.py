
#%%

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

#split the dataset into training and testing sections
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state= 66
)

"""Set the parameters"""
training_accuracy = [] #sets training accuracy to empty list
test_accuracy = [] #sets test accuracy to empty list
#test breast cancer dataset from one to 10 nearest neighbors
neighbor_settings = range(1,10)


for n_neighbors in neighbor_settings:
  #build an instance of a model
  model = KNeighborsClassifier(n_neighbors= n_neighbors) #create a model with n_neighbors = n_neighbors
  model.fit(X_train, y_train) #fit model into training data

  #record accuracy of model in the training data
  training_accuracy.append(model.score(X_train, y_train))

  #record accuracy of the model in the testing data
  test_accuracy.append(model.score(X_test,y_test))

"""plot the results of the test"""
#plot the training accuracy
plt.plot(neighbor_settings, training_accuracy, label = "training accuracy")

#plot the testing accuracy
plt.plot(neighbor_settings, test_accuracy, label = "test accuracy")

#create labels and a legend
plt.title("The relationship between the number of neighbors and accuracy in k-NN models")
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.legend()


# %%
