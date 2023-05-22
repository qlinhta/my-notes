import sys

sys.path.append('..')
import matplotlib.pyplot as plt
from src.preprocessing import *
from src.stats import *
from src.transformation import *

plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
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

# Local Outlier Factor
threshold, scores, index_of_outliers = local_outlier_factor(df, 30, 0.5, -1.5)

# Plot
plt.subplots(figsize=(8, 8))
plt.title('Local Outlier Factor (LOF)')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], edgecolors='blue', facecolors='white', s=40., label='Data points')
plt.scatter(X.iloc[index_of_outliers, 0], X.iloc[index_of_outliers, 1], edgecolors='red', facecolors='red', s=50,
            marker='x',
            label='Outliers')
radius = (scores.max() - scores) / (scores.max() - scores.min())
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=1000 * radius, edgecolors='blue', facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlabel('Mean radius (standardized)')
plt.ylabel('Mean texture (standardized)')
plt.legend()
plt.savefig('../plots/lof.png')
plt.show()

X_new = X.drop(index_of_outliers, axis=0)
y_new = y.drop(index_of_outliers, axis=0)

fig, ax = plt.subplots(5, 6, figsize=(20, 20))
for variable, subplot in zip(X.columns, ax.flatten()):
    sns.distplot(X[variable], ax=subplot)
plt.savefig('../plots/distribution.png')
plt.show()

X_new = log_transformation(X_new, ['texture_mean', 'smoothness_mean', 'smoothness_worst'])
X_new = cube_root_transformation(X_new, ['compactness_worst',
                                         'concavity_worst',
                                         'texture_worst',
                                         'smoothness_se',
                                         'symmetry_mean',
                                         'compactness_mean'])

fig, ax = plt.subplots(5, 6, figsize=(20, 20))
for variable, subplot in zip(X_new.columns, ax.flatten()):
    sns.distplot(X_new[variable], ax=subplot)
plt.savefig('../plots/distribution_after_transformation.png')
plt.show()

print(check_normality(X_new))

# Gaussian list
gaussian_list = ['texture_mean_log', 'smoothness_mean_log', 'smoothness_worst_log', 'texture_worst_cbrt']
non_gaussian_list = X_new.columns.drop(gaussian_list)
print(non_gaussian_list)
print(gaussian_list)

X_new = standardize(X_new, gaussian_list)
X_new = normalize(X_new, non_gaussian_list)

fig, ax = plt.subplots(5, 6, figsize=(20, 20))
for variable, subplot in zip(X_new.columns, ax.flatten()):
    sns.distplot(X_new[variable], ax=subplot)
plt.savefig('../plots/distribution_normalized_standardized.png')
plt.show()

X_new.to_csv('../dataset/breast-cancer-wisconsin-processed.csv', index=False)
y_new.to_csv('../dataset/breast-cancer-wisconsin-processed-label.csv', index=False)
