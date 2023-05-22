## Project: Machine Learning

#### Author:

* [Ha Anh TRAN](#)
* [Quyen Linh TA](#)

#### Description:

* This project is a part of the course "Machine Learning"
  at [University Paris Dauphine, PSL](https://dauphine.psl.eu/en/).

#### Content:

* [Problem 1: Implementation Logistic Regression and LDA](#)
* [Problem 2: Working with real-world data to evaluate implemented models](#)

#### Execution:

* Requirements: `Python` version 3.6 or higher.
* Run file `Analysis.R` to get the results of statistical analysis
* Install all required packages by running:
  ```
  python3 setup.py install
  ```
  or you can just run the following command if you have `pip` installed:
  ```
  pip install -r requirements.txt
  ```
* Run file `make_data_beautiful.py` to start preprocessing data process
  ```
  python3 src/make_data_beautiful.py
  ```
* Run files `notebooks` to get the results of implemented models (Logistic
  Regression and LDA)
  ```
  ./Notebooks/{LogisticRegression, LDA, NN}.ipynb
  ```
* Run files `comparator.ipynb` to get the results comparing implemented models with `sklearn` models
  ```
    ./Notebooks/comparator.ipynb
  ```
* Run files `notebooks` to get the results of explaining implemented models.
  ```
    ./Notebooks/explain_model.ipynb
  ```

#### Results:

Table 1: Results of implemented models (Logistic Regression and LDA) and `sklearn` models

| Model               | Accuracy | Precision | Recall   | F1-score | ROC AUC  |
|---------------------|----------|-----------|----------|----------|----------|
| Logistic Regression | 0.981481 | 0.972973  | 0.972973 | 0.972973 | 0.979444 |
| LDA                 | 0.972222 | 1.0       | 0.918919 | 0.957746 | 0.959459 |
| Neural Network      | 0.990741 | 0.973684  | 1.0      | 0.986667 | 0.992958 |
| Linear SVM          | 0.981481 | 0.972973  | 0.972973 | 0.972973 | 0.979444 |
| Ridge               | 0.953704 | 1.0       | 0.864865 | 0.927536 | 0.932432 |
| XGBoost             | 0.962963 | 0.945946  | 0.945946 | 0.945946 | 0.958888 |

* The results of statistical analysis are in directory `plots/`
* The results of implemented models are in directory `src/output_plots/`
* Models are saved in directory `src/output_models/`
* HTML files for investigating missed predictions of logistic regression `src/logistic_missed_predict_investigate/`

#### Project structure:

* `src/`: source code
    * `data/`: data files
    * `output_plots/`: output plots
    * `make_data_beautiful.py`: preprocessing data
    * `main.py`: implementation of Logistic Regression and LDA
    * `comparator.py`: comparing implemented models with `sklearn` models
    * `logistic_missed_predict_investigate/`: investigating missed predictions of logistic regression
    * ...

* `AREA51/`: test and debug code
* `dataset/`: data files
* `Notebooks/`: notebooks
* `plots/`: analysis plots
* `Analysis.R/`: R scripts for analysis
* `README.md`: this file
* `requirements.txt`: list of necessary packages

#### Overview dataset and problem:

*

Dataset: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

* Problem: Predict whether the cancer is benign or malignant
* Data description:
    * 569 samples
    * 30 features
    * 2 classes: benign (357 samples) and malignant (212 samples)
    * 1 target: `diagnosis` (B: benign, M: malignant)
* Features description:
    * `id`: ID number
    * `diagnosis`: diagnosis of breast tissues (B: benign, M: malignant)
    * `radius_mean`: mean of distances from center to points on the perimeter
    * `texture_mean`: standard deviation of gray-scale values
    * `perimeter_mean`: mean size of the core tumor
    * `area_mean`: mean smoothness of the tumor
    * `smoothness_mean`: mean number of concave portions of the contour
    * `compactness_mean`: mean fractal dimension of the tumor
    * `concavity_mean`: mean radius of gyration of the tumor
    * `concave points_mean`: mean perimeter of the tumor
    * `symmetry_mean`: mean area of the tumor
    * `fractal_dimension_mean`: mean smoothness of the tumor
    * `radius_se`: standard error for the mean of distances from center to points on the perimeter
    * `texture_se`: standard error for standard deviation of gray-scale values
    * `perimeter_se`: standard error for the mean size of the core tumor
    * `area_se`: standard error for the mean smoothness of the tumor
    * `smoothness_se`: standard error for the mean number of concave portions of the contour
    * `compactness_se`: standard error for the mean fractal dimension of the tumor
    * `concavity_se`: standard error for the mean radius of gyration of the tumor
    * `concave points_se`: standard error for the mean perimeter of the tumor
    * `symmetry_se`: standard error for the mean area of the tumor
    * `fractal_dimension_se`: standard error for the mean smoothness of the tumor
    * `radius_worst`: "worst" or largest mean value for mean of distances from center to points on the perimeter
    * `texture_worst`: "worst" or largest mean value for standard deviation of gray-scale values
    * `perimeter_worst`: "worst" or largest mean value for the mean size of the core tumor
    * `area_worst`: "worst" or largest mean value for the mean smoothness of the tumor
    * `smoothness_worst`: "worst" or largest mean value for the mean number of concave portions of the contour
    * `compactness_worst`: "worst" or largest mean value for the mean fractal dimension of the tumor
    * `concavity_worst`: "worst" or largest mean value for the mean radius of gyration of the tumor
    * `concave points_worst`: "worst" or largest mean value for the mean perimeter of the tumor
    * `symmetry_worst`: "worst" or largest mean value for the mean area of the tumor
    * `fractal_dimension_worst`: "worst" or largest mean value for the mean smoothness of the tumor
* Target description:
    * `diagnosis`: diagnosis of breast tissues (B: benign, M: malignant)
* Note: `mean`, `se`, `worst` are computed for each image, resulting in 3 features
  for each of the original 30 features

#### TODO:

* [x] Implement `LDA`
* [x] Implement `Logistic Regression`
* [x] Understand the data and comprehend the problem
* [x] Data analysis with visualization in `R`, `Python`
* [x] Implement the statistical analysis for transformation of data
* [x] Outliers detection and investigation
* [x] Implement data transformation
* [x] Implement model evaluation, metrics, and hyperparameter tuning
* [x] Test the `LDA` and `Logistic Regression` models with post-processing data
* [x] Test the `LDA` and `Logistic Regression` models with pre-processing data
* [x] Tuning the hyperparameters of `Logistic Regression`
* [x] Misclassified data analysis
* [x] Evaluate the models and implement `SVM`, `Gaussian Naive Bayes`, `XGBoost` and `CatBoost`
* [x] Implement the ensemble model and compare the results
* [x] Interpret the results
* [ ] Write the report

#### References:

* Author of the
  dataset: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)):
    * Dr. William H. Wolberg, General Surgery Dept., University of
      Wisconsin, Clinical Sciences Center, Madison, WI 53792
    * W. Nick Street, Computer Sciences Dept., University of
      Wisconsin, 1210 West Dayton St., Madison, WI 53706
    * Olvi L. Mangasarian, Computer Sciences Dept., University of
      Wisconsin, 1210 West Dayton St., Madison, WI 53706
* [Logistic Regression lecture notes](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) by Tom Mitchell, Carnegie
  Mellon University
* [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) by
  [H. Wolkowicz](https://www.math.uwaterloo.ca/~hwolkowi/)
* [Logistic Regression](https://www.coursera.org/learn/machine-learning/resources/2QZ9T) by Andrew Ng, Stanford
  University
* [LDA detailed explanation](https://usir.salford.ac.uk/id/eprint/52074/1/AI_Com_LDA_Tarek.pdf) by Tarek Elgindy,
  University of Salford
* [Book Machine Learning in Action](https://www.manning.com/books/machine-learning-in-action) by Peter Harrington
* [Machine Learning A Probabilistic Perspective](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf) by Kevin
  P. Murphy

#### License:

* [MIT License](http://www.opensource.org/licenses/mit-license.php)