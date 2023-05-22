import sys

from numpy import percentile
from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.model_selection import train_test_split

sys.path.append('..')
import matplotlib.pyplot as plt
from src.transformation import *

plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('lines', markersize=10)
import warnings

warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('../dataset/breast-cancer-wisconsin.data')

# Drop Unnamed: 32 column and id column
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
# Encode the diagnosis column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
# Split the data into X and y
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Outlier detection KNN
clf_name = 'KNN'
clf = KNN()
clf.fit(X_train)
scores_pred = clf.decision_function(X) * -1
y_pred = clf.predict(X)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)

print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)
X_train_2d = X_train[['radius_mean', 'texture_mean']]
X_test_2d = X_test[['radius_mean', 'texture_mean']]
# Visualize the data
visualize(clf_name, X_train_2d, y_train, X_test_2d, y_test, y_train_pred, y_test_pred, show_figure=True,
          save_figure=True)
