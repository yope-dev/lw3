# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape
print(f'dataset.shape: {dataset.shape}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# head
print(f"dataset.head\n{dataset.head(20)}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# descriptions
print(f"dataset.describe\n{dataset.describe()}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# class distribution
print(f"class distribution\n{dataset.groupby('class').size()}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"Name: {name} Mean: {cv_results.mean()*100}% std: {cv_results.std()}")
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

 # Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)




# Evaluate predictions
print(f"accuracy_score {accuracy_score(Y_validation, predictions)*100}%\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"confusion_matrix\n{confusion_matrix(Y_validation, predictions)}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f"classification_report\n{classification_report(Y_validation, predictions)}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


