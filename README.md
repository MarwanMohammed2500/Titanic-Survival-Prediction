# Titanic Survival Prediction with Logistic Regression

This project focuses on analyzing and predicting survival outcomes for passengers aboard the Titanic using logistic regression. The repository provides an in-depth look into data preprocessing, feature engineering, model training, and evaluation steps.

---

## Project Overview

The objective of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on features such as age, sex, and ticket class. The project follows these key steps:

1. **Data Preprocessing**: Managing missing values, encoding categorical variables, and normalizing numerical features.
2. **Feature Engineering**: Optimizing features through selection and transformation for better model performance.
3. **Model Training**: Applying logistic regression with custom thresholding to balance precision and recall.
4. **Evaluation**: Measuring model effectiveness using metrics like accuracy, precision, recall, and F1-score.

---

## Dataset

The dataset provides detailed information about Titanic passengers, including:

- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Survival status (0 = No, 1 = Yes).
- **Pclass**: Ticket class (1 = First, 2 = Second, 3 = Third).
- **Name**: Passenger's name.
- **Sex**: Gender of the passenger.
- **Age**: Passenger's age.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Ticket fare.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

---

## Key Steps

### 1. Data Preprocessing
- **Handling Missing Values**:
  - `Age`: Imputed using the median.
  - `Embarked`: Imputed using the mode.
- **Feature Encoding**:
  - Categorical variables like `Sex` and `Embarked` were encoded using ordinal encoding.
- **Normalization**:
  - Transformations were applied to `SibSp` and `Parch` to reduce their weight when training the model.

### 2. Feature Engineering
- Removed features such as `Name`, `Cabin`, and `Ticket` due to irrelevance or high proportions of missing data.
- Experimented with transformations for skewed features to enhance model accuracy.

### 3. Model Training
- Logistic regression served as the primary classification algorithm.
- Custom thresholding was implemented to balance precision and recall effectively.
- Sampling methods, including undersampling and oversampling, were tested to address class imbalance.

### 4. Evaluation
The model's performance was evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions.
- **Precision**: The accuracy of positive predictions.
- **Recall**: The proportion of actual positives identified.
- **F1-Score**: The harmonic mean of precision and recall.

---

## Results

| Metric       | Train (%) | Test (%) |
|--------------|-----------|----------|
| Accuracy     | 72.05     | 75.84    |
| Precision    | 59.51     | 61.43    |
| Recall       | 85.09     | 90.13    |
| F1-Score     | 70.04     | 73.07    |

**Insights**:
- The model exhibits a good balance between precision and recall after applying threshold adjustments.
- Custom feature transformations and selection improved predictive performance.
- Future enhancements could focus on refining feature engineering and handling class imbalances more effectively.

---

## Usage
### Custom Thresholding
To adjust the prediction threshold:
```python
thresh = best_thresh
custom_thresh_pred_test = (custom_thresh_positive_probs_test >= thresh).astype(int)
custom_thresh_pred_train = (custom_thresh_positive_probs_train >= thresh).astype(int)
```

### Sampling Techniques
- **Random Undersampling**:
  ```python
  from imblearn.under_sampling import RandomUnderSampler
  UnderSample = RandomUnderSampler(random_state=42)
  UnderSample_x_train, UnderSample_y_train = UnderSample.fit_resample(x_train, y_train)
  ```

- **Random Oversampling**:
  ```python
  from imblearn.over_sampling import RandomOverSampler
  OverSample = RandomOverSampler(random_state=42)
  OverSample_x_train, OverSample_y_train = OverSample.fit_resample(x_train, y_train)
  ```
- **Mixing Between Random Undersampling and Random Oversampling**:
  ```python
  underSample = RandomUnderSampler(random_state=42, sampling_strategy=0.7)
  underSample_x_train, underSample_y_train = underSample.fit_resample(x_train, y_train)

  overSample = RandomOverSampler(random_state=42)
  overSample_x_train, overSample_y_train = overSample.fit_resample(underSample_x_train, underSample_y_train)
  ```
### Adjusting Class Weights
```python
log_reg_with_class_weights = LogisticRegression(class_weight="balanced", max_iter=500)
log_reg_with_class_weights.fit(x_train.to_numpy(), y_train.to_numpy().ravel())
```

### Cross-Validation
```python
cv_log_reg = LogisticRegressionCV(max_iter=500, class_weight="balanced", n_jobs=-1, scoring="f1")
cv_log_reg.fit(x_train.to_numpy(), y_train.to_numpy().ravel())
```

## Best Model Class
```python
class CustomThresholdLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, thresh=0.5, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        self.thresh = thresh
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
            verbose=self.verbose,
            warm_start=self.warm_start,
            n_jobs=self.n_jobs,
            l1_ratio=self.l1_ratio
        )
        # Fit the model
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        thresh = self.thresh
        proba = self.model_.predict_proba(X)[:, 1]
        return (proba >= self.thresh).astype(int)
    
    def predict_proba(self, X):
        X = check_array(X)
        return self.model_.predict_proba(X)
```
#### **Constructor (`__init__`)**:
This method initializes the class parameters. The `__init__` method accepts several hyperparameters, some of which are typical of the logistic regression model and others specific to this custom implementation (e.g., `thresh` for the custom threshold).

- **Parameters**:
  - `thresh`: A custom threshold for classification. If the predicted probability is greater than or equal to this threshold, the predicted class will be 1; otherwise, it will be 0.
  - `penalty`: The regularization method ('l2' or 'l1'). 'l2' is the default.
  - `dual`: A boolean indicating whether to solve the dual problem (used for optimization).
  - `tol`: Tolerance for stopping criteria.
  - `C`: Inverse of regularization strength; must be a positive float.
  - `fit_intercept`: Whether or not to include an intercept in the model.
  - `intercept_scaling`: Scaling of the intercept term (used for certain solvers like 'liblinear').
  - `class_weight`: Used to assign weights to different classes.
  - `random_state`: Seed used for random number generation.
  - `solver`: The solver algorithm to use for optimization (e.g., 'lbfgs', 'liblinear').
  - `max_iter`: Maximum number of iterations for the optimization.
  - `multi_class`: Multi-class handling method. It's marked as deprecated in scikit-learn and should be handled by the solver itself.
  - `verbose`: Controls verbosity during the fitting process.
  - `warm_start`: Whether to reuse the solution of the previous call to `fit`.
  - `n_jobs`: Number of CPU cores to use for computation.
  - `l1_ratio`: Used when the penalty is elastic net (a combination of 'l1' and 'l2' regularization).

#### **Fit Method (`fit`)**:
The `fit` method trains the logistic regression model using the training data. It initializes a `LogisticRegression` model from scikit-learn with the provided parameters and fits it to the input data `X` and target labels `y`.

- **Process**:
  1. The input data (`X`) and labels (`y`) are validated using `check_X_y` to ensure they are in the correct format.
  2. A `LogisticRegression` model is initialized with the specified hyperparameters.
  3. The model is trained using `fit(X, y)`.
  4. The fitted model is stored in `self.model_` for later use in predictions.

- **Return**:
  - The method returns the instance (`self`) to allow for chaining calls to methods (common in scikit-learn).

#### **Predict Method (`predict`)**:
The `predict` method makes class predictions based on the fitted logistic regression model.

- **Process**:
  1. It first validates the input features `X` using `check_array`.
  2. It calculates the predicted probabilities of the positive class using `self.model_.predict_proba(X)[:, 1]`. This gives the probability for each sample belonging to class 1 (the positive class).
  3. It compares each probability with the specified threshold (`self.thresh`). If the probability is greater than or equal to the threshold, the sample is predicted as class 1; otherwise, it's predicted as class 0.
  4. The method returns the class predictions as an integer array (0 or 1).

- **Return**:
  - A binary array where each element corresponds to the predicted class (0 or 1) for the corresponding input sample.

#### **Predict Probability Method (`predict_proba`)**:
The `predict_proba` method returns the predicted class probabilities for the input data `X`.

- **Process**:
  1. It validates the input features `X` using `check_array`.
  2. It returns the probabilities for each class (not just the positive class) using `self.model_.predict_proba(X)`.

- **Return**:
  - An array of predicted probabilities for each class, where each element corresponds to the probability of each class for each sample.
---

## Used Dependancies
This Project uses 4 main libraries for I/O and modeling:
- NumPy
- Pandas
- Matplotlib
- SciKit-Learn

And another for resampling:
- imblearn

---
## Future Work

- Integrate ensemble methods to improve recall and overall model robustness.
- Explore additional feature transformation and selection strategies for better predictive performance.
---
# Author
This Project is the work of Marwan Mohammed
