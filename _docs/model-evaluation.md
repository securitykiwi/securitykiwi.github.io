---
title: Model Evaluation
author: 
---

<style>p {text-align: justify;}</style>

Model evaluation is the assessment of machine learning algorithm's performance after it has been trained. We define a performance metric and measure the model our algorithm has created against it, or multiple metrics if we desire. The output will tell us how well our algorithm has performed and we can consider how to proceed. Do we need to improve features to gain more insight into the data in case of underfitting? Should we perform tuning to improve accuracy or does it look like our model has failed, and we need to consider another model? Evaluation answers these questions.

Evaluation measures change with the type of system you are analysing. The information below concentrates on the two most common tasks you may face with security-focused machine learning; regression, classification. We primarily work with scikit-learn examples, however, TensorFlow has an equally large number of evaluation metrics. See <a href="https://www.tensorflow.org/api_docs/python/tf/keras/metrics" target="_blank">TensorFlow metrics</a> documentation for more.

### Contents

* [Cross-validation](#cross-validation)
* [Evaluation Metrics](#evaluation-metrics)
    * [Regression Evaluation Metrics](#regression-evaluation-metrics)
    * [Classification Evaluation Metrics](#classification-evaluation-metrics)
* [Summary](#summary)

## Cross-Validation

<i>Cross-validation</i> is a method for evaluating the performance throughout a whole dataset, the methods we discuss below in Evaluation Metrics are typically performed comparing the test and training dataset. The cross-validation method still splits the dataset into two, trains the model on the training set and evaluates the model on the test set. However, it repeats the second and third stage. This additional validation allows an algorithm to be tested against a larger portion of the dataset, removing the weaknesses of a method which only tests against one section of a test dataset. Cross-validation is also used for model tuning and feature selection.

Cross-validation is a method which involves additional training steps, so it can take longer. However, as it is exposed to more of the dataset it has advantages such as generally avoiding overfitting, where the algorithm learns to represent the data too closely and cannot generalize to new data. Since cross-validation sees more of the dataset, it also avoids selection bias, which is an unintentional possibility when we split our dataset.

### Holdout

We have used cross-validation to some extent already. <i>Holdout</i> is a type of cross-validation and the process of splitting the dataset into a training and test dataset. This ensures the data is unseen and accurate evaluation can take place. The dataset kept aside, which is not used for training, may be referred to as the holdout set. However, we will refer to it as the test dataset/set.

Holdout is the most common and easiest method to use, we only train one model and we can analyse and use our model. We have implemented holdout using scikit-learns `train_test_split` function, which splits the dataset into two segments.

### K-Fold Cross-Validation

K-Fold cross-validation allows us to obtain the benefits we mentioned above. The process splits our dataset up into chunks and trains a model using the individual chunks. As there can be any number of chunks depending on the size of the dataset, it can be said that we split the dataset into _K_ chunks and train _K_ times. For example, we have K = 5 so we train 5 times, each trained on a separate chunk. We get the performance by averaging the result from each training session.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/" alt="An example of K-Fold Cross-Validaiton splitting up a dataset and training on individual chuncks." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Example of K-fold cross-validation method.</p>

### Repeated Random Sub-Sampling

The final type of cross-validation we will discuss is repeated random sub-sampling. This method is similar to K-fold cross-validation, however, _K_ is not equally sized chunks of the dataset. Instead, we train on randomly selected samples from the entire set in the form of a percentage, i.e we train on 10% of the dataset selected at random. We then train multiple models on each, as in K-fold cross-validation.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/" alt="An example of K-Fold Cross-Validaiton splitting up a dataset and training on individual chuncks." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Example of repeated random sub-sampling in cross-validation.</p>

The weakness of this approach is its randomness, we may never select some data points and we may select multiple points more than once. However, randomness provides protection from selection bias. K-fold may hold the best balance, depending on your application.

# Evaluation Metrics

Typically, evaluation metrics compare the accuracy of the algorithm on the training set and the test set. We can use the techniques below to gain inside into the performance of our model and find potential issues, depending on the metric. Evaluation metrics can be used as stand-alone measures which you as the programmer consider. However, they are much more powerful when used also used for optimization, where we train multiple models and pick the model which returns the best metric.

This section is split into two categories of evaluation metrics for different tasks; regression and classification.

#### Metrics

* [Regression Evaluation Metrics](#regression-evaluation-metrics)
    * [Mean Absolute Error](#mean-absolute-error)
    * [Mean Squared Error](#mean-squared-error)
    * [Root Mean Squared Error](#root-mean-squared-error)
    * [Coefficient of Determination (R2, R^2, R Squared)](#coefficient-of-determination-r2-r2-r-squared)
* [Classification Evaluation Metrics](#classification-evaluation-metrics)
    * [Classification Accuracy](#classification-accuracy)
    * [Precision](#precision)
    * [Recall](#recall)
    * [F-Score](#f1-f-score-f-measure)


Both Scikit-learn and TensorFlow have functions which provide easy access to calculate various evaluation metrics. For example:

Scikit-learn

```python
# Import Mean Squared Error (MSE)
from sklearn import metrics
print('MSE:', metrics.mean_squared_error(y_train, y_pred))
```

TensorFlow

```python
# Evaluate the model on the test data
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)
```

## Regression Evaluation Metrics

### Mean Absolute Error

Mean Absolute Error (MAE) is used to estimate the accuracy of a predicted value. MAE calculates the average difference between actual values and predicted values. MAE outputs a positive value between 0 and 1, lower values are better.

An example of MAE on a dataset of 5 values. Try running this in a notebook and changing `predicted_values` to the same as `actual_values` one by one and note the metric change.

```python
from sklearn.metrics import mean_absolute_error
actual_values = [2, 6, 0.5, 4, 1.5]
predicted_values = [3, 5.5, 1, 4, 2]
mae = mean_absolute_error(actual_values, predicted_values)
print(mae)
```

### Mean Squared Error

Mean Squared Error (MSE) is used to estimate the accuracy of a predicted value. MSE is the sum of the squared difference between a predicted value and the actual value across all data points, divided by the number of data points. MSE outputs a positive value, the lower the value the less the difference between the predicted and actual values. MSE outputs a positive value between 0 and 1, lower values are better.

An example of MSE on a dataset of 5 values. Try running this code and experimenting with the `predicted_values`.

```python
from sklearn.metrics import mean_squared_error
actual_values = [2, 6, 0.5, 4, 1.5]
predicted_values = [3, 5.5, 1, 4, 2]
mse = mean_squared_error(actual_values, predicted_values)
print(mse)
```

### Root Mean Squared Error

Root Mean Square Error (RMSE) is used to show how close to the line of best fit our data points are. RMSE is the root of the mean squared error, giving us the standard deviation of the prediction errors (the difference between points). Standard deviation describes how much members of a group (of data points) differ from the mean. RMSE outputs a positive value between 0 and 1, lower values are better.

There is no RMSE function, however, we can use the square root function. Here is the same example as above, try experimenting again.

```python
from sklearn.metrics import mean_squared_error
from math import sqrt
actual_values = [2, 6, 0.5, 4, 1.5]
predicted_values = [3, 5.5, 1, 4, 2]
mse = mean_squared_error(actual_values, predicted_values)
rmse = sqrt(mse)
print(rmse)
```

### Coefficient of Determination (R2, R^2, R Squared)

The coefficient of determination, more commonly known as R<sup>2</sup> (R Squared), is another method to describe how well a model fits a dataset and describes variation as a percentage. R<sup>2</sup> is calculated by dividing the square error of predicted values and the square error of actual values minus 1. R<sup>2</sup> outputs a positive value between 0 and 1, values closer to 1 are better. For example, an R2 of 0.9 tells us that the correlation between values is high, whereas a R<sup>2</sup> of 0 or a very low R<sup>2</sup> indicates there is no correlation between values - we cannot make a prediction.

An example of R<sup>2</sup> using scikit-learns `r2_score`. Try experimenting with `predicted_values` and observing the result.

```python
from sklearn.metrics import r2_score
actual_values = [2, 6, 0.5, 4, 1.5]
predicted_values = [3, 5.5, 1, 4, 2]
r2 = r2_score(actual_values, predicted_values)
print(r2)
```

## Classification Evaluation Metrics

Many classification evaluation metrics use terminology relating to the state of the prediction, we briefly discuss this here to aid understanding. These terms refer to the <a href="https://en.wikipedia.org/wiki/Sensitivity_and_specificity" target="_blank">sensitivity and specificity</a> of an algorithm.

**True Positive (TP)** the actual value was true and the prediction returned true. E.g. an email was spam and was classified as spam.

**True Negative (TN)** the actual value was true and the prediction returned false. E.g. an email was spam and was classified as not spam.

**False Positive (FP)** the actual value was false and the prediction returned true. E.g. an email was not spam and was classified as spam.

**False Negative (FN)** the actual value was false and the prediction returned false. E.g. an email was not spam and was not classified as spam.

<div style="text-align:center;"><img src="/assets/images/evaluation-and-tuning/confusion-matrix-sensitivity-specificity.png" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">A confusion matrix of sensitivity and specificity.</p>

### Classification Accuracy

Classification accuracy is the measure of correct predictions in the total number of predictions. Classification accuracy works well for classes which are balanced but does not work well for unbalanced classes. For example; 70% of a dataset is non-spam, 30% is spam is OK. Whereas, 95% non-spam and 5% spam is not OK. Unbalanced classes skew the result, making it an unsuitable metric.

An example of `accuracy_score` from scikit-learn on a 4 digit dataset. In this example, one predicted value does not match the actual value, resulting in an accuracy of 75%, i.e 1 in 4 is incorrectly classified.

```python
from sklearn.metrics import accuracy_score
actual_values = [2, 6, 2, 4]
predicted_values = [2, 6, 1, 4]
acc = accuracy_score(actual_values, predicted_values)
print(acc)
```

### Precision

Precision measures how well our predicted value compares to the actual value. In our spam example, how many instances labelled spam were actually spam.

An example using `average_precision_score` from scikit-learn. The actual values fall within binary classes (0 or 1) and the predicted values are percentage predictions. Try experimenting with the values. Note the threshold, surpassing this would place them in a class, hence the behaviour of the output.

```python
from sklearn.metrics import average_precision_score
actual_values = [1, 0, 0, 1]
predicted_values = [0.2, 0.6, 0.4, 0.9]
precision = average_precision_score(actual_values, predicted_values)
print(precision)
```

### Recall

Recall measures the ratio of the true positive rate against the true positive rate plus the false-negative rate (recall = tp / (tp + fn)).  In our example, it measures the proportion of spam which was identified as spam by an algorithm. A false negative, in this case, is spam which was mislabelled as not spam. In this case, recall returns the ratio of instances which were actually spam. The ratio is 0 to 1, with a higher value representing a better outcome.

An example using scikit-learn `recall_score`. 

```python
from sklearn.metrics import recall_score
actual_values = [1, 0, 0, 1]
predicted_values = [1, 1, 0, 0]
recall = recall_score(actual_values, predicted_values, zero_division=1)
print(recall)
```

### F1 (F-score, F-measure)

F1, also known as the F-Score or F-Measure, is a measure of accuracy. F1 uses precision and recall to create its score, this provides the benefits of both in one metric. The output is 0 to 1, with higher values representing better accuracy. For example, if precision is low and recall is low, the F-Score will be low. If precision is high and recall is low, the F-Score will be high. This is useful when you want to be sure you have good precision and recall - in our spam example, you want to know spam has been labelled spam correctly and with low false rates.

An example using the scikit-learn `f_score`.

```python
from sklearn.metrics import f1_score
actual_values = [1, 0, 0, 1]
predicted_values = [1, 1, 0, 0]
f1_score(actual_values, predicted_values) 
```

## Summary 

On this page, we discussed the most used methodology and evaluation technique, holdout. We moved onto discussing cross-validation techniques which provide more a more robust technique against selection bias and allow the whole dataset to be validated, rather than a small part.

We discussed a number of evaluation metrics for different types of task and briefly discussed how they can be used and where they are useful.

In the next page we discuss tuning your technique to achieve the best results.

---

## References

<ol>
    
<li>TensorFlow (2020a) <i>Training and evaluation with the built-in methods</i>. TensorFlow Documentation. <a href="https://www.tensorflow.org/guide/keras/train_and_evaluate" target="_blank">https://www.tensorflow.org/guide/keras/train_and_evaluate</a></li>

<li>TensorFlow (2020b) <i>Tensorflow Model Analysis Metrics</i>. TensorFlow Documentation. <a href="https://www.tensorflow.org/tfx/model_analysis/metrics" target="_blank">https://www.tensorflow.org/tfx/model_analysis/metrics</a></li>

<li>Scikit-Learn (2020a) <i>Metrics and scoring: quantifying the quality of predictions</i>. <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation" target="_blank">https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation</a></li>

<li>Scikit-Learn (2020b) <i>sklearn.metrics.accuracy_score</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html</a></li>

<li>Scikit-Learn (2020c) <i>sklearn.metrics.average_precision_score</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score</a></li>

<li>Scikit-Learn (2020d) <i>sklearn.metrics.recall_score</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html</a></li>

<li>Wikipedia (2020a) <i>Mean Squared Error</i>. <a href="https://en.wikipedia.org/wiki/Mean_squared_error" target="_blank">https://en.wikipedia.org/wiki/Mean_squared_error</a></li>

<li>Wikipedia (2020b) <i>Coefficient of Determination</i>. <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination" target="_blank">https://en.wikipedia.org/wiki/Coefficient_of_determination</a></li>

<li>Wikipedia (2020c) <i>Precision and Recall</i>. <a href="https://en.wikipedia.org/wiki/Precision_and_recall" target="_blank">https://en.wikipedia.org/wiki/Precision_and_recall</a></li>

<li>Wikipedia (2020d) <i>F1 Score</i>. <a href="https://en.wikipedia.org/wiki/F1_score" target="_blank">https://en.wikipedia.org/wiki/F1_score</a></li>
    
</ol>
