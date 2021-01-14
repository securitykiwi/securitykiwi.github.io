---
title: Training Regression Models
author:
---

<style>p {text-align: justify;}</style>

Regression is used to predict house prices, stock prices and in the security field, it has been used to predict denial of service (DoS) severity and forecast 'cyber weather' as we saw with Park et al in the introduction - our initial learning here won't be as complex. This page serves as an introduction to training regression models.

This page is split into two sections for the two different functions regression can be used; prediction and classification. We will discuss three regression techniques, each more powerful than the last - or more suited to specific tasks - and we will build an example of each.

Each section includes a link to a Google CoLab live environment for you to follow along with the code if you haven't installed Anaconda / Jupyter Notebooks.

#### Contents

* [Regression Predictors](#regression-predictors)
    * [Linear Regression](#linear-regression)
    * [Polynomial Regression](#polynomial-regression)
* [Regression Classifiers](#regression-classifiers)
    * [Logistic Regression](#logistic-regression)
* [Summary](#summary)

# Regression Predictors

## Linear Regression

Linear regression is a technique from the field of statistics, many aspects of machine learning are borrowed from other areas including statistics and computer science. <i>Linear regression</i> assumes there is a linear relationship between these values and attempts to predict a linear outcome - the best fit line is straight. In machine learning <i>Regression</i> is a supervised technique (takes labelled data) which processes data to find relationships between a value (x) and a predicted value (y). A line can be drawn on a graph to best fits this data. A linear relationship is defined as a proportional relationship between changes in related values; when a value increases so does the other, when a value decreases so does the other.

A popular example is the number of cricket (the insect) chirps per minute mapped to temperature; as temperature increases crickets chirp more frequently, their chirps slow down as temperature decreases. You can predict temperature from the frequency of cricket chirps, and you can predict the number of cricket chirps per minute from temperature.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/crickets-temp-graph.jpg" alt="A graph illustrating cricket chirps versus temperature." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">An example of a linear graph, temperature versus cricket chirps.</p>

Any straight line can be represented by the equation y = mx + b. In our example, we are trying to predict the temperature from the number of chirps per minute, y represents the predicted temperature value we seek. m is the slope of the line, x is the number of chirps per minute, and b is the intersection of the line with the y-axis (the y-intercept). If you want to think about how this works, consider our aim is to draw the best fit line, we need at least two points to get the slope m and y-intercept b gives us a third. We can now draw a line. We won't use math on this page again, it will help should you read about linear regression anywhere else.

Despite linear regressions strengths for some tasks, it has downsides for others. It assumes a linear relationship between data inputs, so it is not useful for non-linear data. It is a relatively simple technique and cannot handle complexity. However, it has found a place in many applications due to its simplicity and ease of understanding. As well as non-security applications such as forecasting house prices, sales and the stock market, linear regression techniques have been used to forecast cyber threat intelligence with high accuracy. We discussed in the introduction <a href="/docs/machine-learning-examples-in-security/" target="_blank">Machine Learning Examples in Security</a> how Park et al forecast 'cyber weather', accurately warning of mass worm attacks within large networks.

## Ordinary Least Squares

<i>Cost functions</i> determine the optimal predicated values for us to create a line which best fits input data. Cost functions measure the difference between the actual and predicted values, this difference is called the "error". The scikit-learn Linear Regression implementation uses the <i>Ordinary Least Squares</i> regression technique, where the aim is to minimise the sum of the distance from each data point to the regression line squared and the sum of all squared errors. This can be better understood with the graph below.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/ordinary-least-square-graph.jpg" alt="A graph illustrating  least square, minimising the surface area of each square creates the optimal line." style="width:700px;"/></div>

<p style="text-align: center; font-style: italic;">An illustrative example of least squares applied to data points.</p>

The graph shows the purpose of the ordinary least squares function can be said to create a line which has reduced the surface area of each square to its maximum potential, thus creating a line which best fits each data point.

Of course, this complexity is hidden from the user. All we as programmers do is pass parameters into the `linear_model.LinearRegression()` function, while being aware of the limitations of the technique and assumptions which linear regression makes.

## Train a Linear Regression Model

Let's train a linear regression model using scikit-learns `linear_model.LinearRegression()` function. We will keep things simple and use a prediction dataset built into scikit-learn, the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html" target="_blank">boston house prices dataset</a>, which we will use it to predict house price trends. In later sections, we will focus more security-specific tasks which require dataset wrangling and further considerations. Here we want to understand how the model is trained and gain some exposure to machine learning with scikit-learn and the ease of using machine learning libraries.

Follow along with the code for a linear regression model in a <a href="https://colab.research.google.com/drive/1BNDgoE-_qaG25sKYoVpw-Xm536r4l_Rr" target="_blank">Google CoLab</a> live environment.

#### Import packages

First, we need to import various packages (also called libraries) to allow us to manipulate, view and process the data. Pandas for data manipulation, matplotlib for visualizing data, the Boston dataset and scikit-learn's regression algorithm.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
```

#### Get the data

Now we have access to those packages, we load the dataset (by assigning it dataset to a variable) and create a pandas <i>DataFrame</i> from it. If you've forgotten DataFrames from the <a href="/docs/exploring-datasets/" target="_blank">Datasets and Data Collection</a> section, they are similar to tables and a powerful tool in Pandas allowing easy manipulation of data, with less code and fewer headaches. The raw Boston dataset doesn't include column headings, they're squirrelled away in a separate variable, we add them back here for increased understanding.

```python
# Assign dataset to variable
boston = load_boston()
# Load data from dataset into Panda's DataFrame and assign DataFrame to variable
data = pd.DataFrame(boston.data)
# Add column names to DataFrame
data.columns = boston.feature_names
# Print DataFrame
data.head()
```

We can also visualize the data to better understand what we are working with. Below we visualize the price, the value we will later predict, as a histogram using matplotlib `hist()` function. You can try applying the various methods we discussed in <a href="/docs/exploring-datasets/" target="_blank">Exploring Datasets</a> here too.

```python
# Plot histagram of price (boston.target)
plt.figure(figsize=(4, 3))
plt.hist(boston.target)
plt.xlabel('price ($1000s)')
plt.ylabel('count')
plt.tight_layout()
```

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/boston-prices-hist.jpg" alt="A histagram showing the distribution of boston house prices. Prices resemble a mountain, most house prices are around 200,000 dollars." style="width:350px;"/></div>

<p style="text-align: center; font-style: italic;">A histagram showing the distribution of boston house prices.</p>

#### Assign values to axis

Linear regression requires two independent variables, it's likely any dataset you work on will need to be manipulated to split the independent variables into x and y variables. To illustrate this, we'll add the price to the DataFrame, then take it away as you would with another dataset. You may recognise the code, we discussed a similar process in the <a href="/docs/dataset-preperation/" target="_blank">Dataset Preperation</a> page.

```python
# Add PRICE column filled with prices from boston.target
data['PRICE'] = boston.target
# Print DataFrame, now with PRICE column
data.head()
```
Now to remove that extra column.

```python
# Drop the column named 'PRICE'
data.drop('PRICE', axis = 1)
```

Moving on, we assign our independent variables to the x and y axis.

```python
# Assign values to x and y
x = data
y = boston.target
```

#### Split the dataset

Now we must split our dataset into a training and test datasets, keeping some data aside to test the accuracy of our method after training. Scikit-learn includes a function for this `train_test_split()`, which we use on x and y to create test and training segments for each. Below we pass `text_size` and `random_state` parameters, keeping 30% of the dataset aside for testing. The `random_state` value controls the amount of shuffle applied to the dataset, and when provides consistency across function calls (each time you use the function). See the functions <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">documentation</a> for more.

```python
from sklearn.model_selection import train_test_split
# Split the dataset, keeping 30% aside for the test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 4)
```

#### Train a model

Invoking the `LinearRegression()` function, we train the algorithm by fit the training data to it. `lr` stands for LinearRegression, just a short variable name.

```python
# Assign LinearRegression algorithm to variable 
lr = LinearRegression()
# Train the model using the training sets 
lr.fit(x_train, y_train)
```

#### Predict prices

Now our model is trained, we apply it to form some predictions and visualize the results.

```python
# Predict values using LinearRegression() lr
y_pred = lr.predict(x_train)

# Plot price predictions
plt.figure(figsize=(6, 4))
plt.scatter(y_train, y_pred)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel("Actucal price ($1000s)")
plt.ylabel("Predicted price ($1000s)")
plt.tight_layout()
```
<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/boston-prices-predicted.jpg" alt="A graph showing the predicted and actual prices of boston houses." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">A graph of the predicted and actual prices of boston houses.</p>

#### Evaluate

We can calculate metrics, such as the Mean Squared Error (MSE) by using the scikit-learn metrics package. We will discuss further methods of evaluation in the <a href="docs/model-evaluation/" target="_blank">Model Evaluation</a> page ahead.

```python
from sklearn import metrics
# Calculate and print MSE
print('MSE:', metrics.mean_squared_error(y_train, y_pred))
```

```text
MSE: 19.07368870346903
```

## Mean Squared Error

Mean Squared Error (MSE) is a method of estimating the accuracy of a predicted value, it measures the average squared difference between a predicted value and the actual value. MSE is a common way of measuring the accuracy of a linear regression model. We used scikit-learn's MSE function `mean_squared_error()` from the metrics package above to compare the training data (actual values) and the predicted values to determine the error. MSE outputs a positive value, the lower the value the less the difference between the predicted and actual values and the better the technique you are measuring has performed.


# Polynomial Regression

Polynomial regression is a technique which allows us to work with more complex data which does not fit to a straight line. We can use a simple scikit-learn function `PolynomialFeatures()` to preprocess our data and utilise the same linear regression function `LinearRegression()` above on non-linear data. For example, the Boston Housing Dataset contains some non-linear relationships. 

The graph below shows the relationship between the `boston.target` value, the median house value (MEDV) and the 'lower status of the population' (LSTAT), a measurement of the proportion of adults without, some high school education and proportion of male workers classified as labourers.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/non-linear-data-boston.jpg" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Median house value (MEDV) vs. LSTAT, non-linear data.</p>

As you can see, a straight line would not accurately represent the data. We implement polynomial regression below.

You can follow along in a <a href="https://colab.research.google.com/drive/1DWHoKK-BdWBC3YXpLqzMPVYvr0YglQ62" target="_blank">Google CoLab</a> live environment.

#### Import packages

```python
import operator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
```

#### Load dataset

Load the data in the same way as we did previously.

```python
# Assign dataset to variable
boston = load_boston()
# Load data from dataset into Panda's DataFrame and assign DataFrame to variable
data = pd.DataFrame(boston.data)
# Add column names to DataFrame
data.columns = boston.feature_names
# Print DataFrame
data.head()
```

#### Assign data to axis

We assign the LSTAT colum to the x axis, and the MEDV data to the y axis.

```python
x = data['LSTAT']
y = boston.target
```
You can visualize the data on a scatter graph here.

```python
# Plot scatter graph of x, y
plt.scatter(x,y, s=10)
# Axis labels
plt.xlabel("Lower Status of Pop (LSTAT)")
plt.ylabel("Mediam House Value (MEDV)")
# Show graph
plt.show()
```

#### Create polynomial features

Transform the data points into polynomial features using the scikit-learn function.

```python
# Assign PolynomialFeatures() function to variable for use
poly = PolynomialFeatures(degree=2, include_bias=False)
# Tranform and fit the x axis. Reshape transforms the data to the correct size for the operation.
x_poly = poly.fit_transform(x.values.reshape(-1,1))
```

#### Make predictions

Invoke `LinearRegression()` as `pr` (polynomial regression) and train the model using `fit()`. Once trained make predictions `predict()`.

```python
# Assign the LinearRegression() function to variable for use
pr = LinearRegression()
# Train model
pr.fit(x_poly, y)
# Make predictions
y_pred = pr.predict(x_poly)
```

#### Visualize results

Below we visualize our results, fitting a curved best fit line onto the dataset. Some wrangling was required to get a single line rather than a mess of lines, I believe this is because the PolynomialFeatures() didn't order its output. Below we sort the values and plot the resulting single line. Thanks to a blog post on Towards Data Science for the sorting fix.

```python
# Plot data
plt.scatter(x, y, s=10)
# Graph labels
plt.xlabel("Lower Status of Pop (LSTAT)")
plt.ylabel("Mediam House Value (MEDV)")
# These lines courtesy of Towards Data Science post
sort_axis = operator.itemgetter(0)
sort = sorted(zip(x,y_pred), key=sort_axis)
x, y_pred = zip(*sort)
# Plot line
plt.plot(x, y_pred, color='#ee8866')
# Plot layout styles (default)
plt.show()
```

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/boston-polynomial-regression.jpg" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">A graph showing polynomial regression fitting a non-linear best fit line.</p>


# Regression Classifiers

The regression techniques we have discussed so far are not suited to classification problems. Andrew Ng, a world-renowned Stanford AI scientist and course leaders of the popular Coursera machine learning course, describes the issue with little math in an early Stanford <a href="https://youtu.be/HZ4cvaztQEs?t=2972" target="_blank">YouTube video lecture</a>. Basically, linear regression is sensitive to outliers and large differences between independent variables. Below we work with Logistic Regression which is suitable for classification tasks.

## Logistic Regression

The logistic function, also called the sigmoid function, forms the core of logistic regression. The technique performs the same action as linear regression but compresses input values between the range of 0 and 1, represented by an S-shaped curve. These values can be mapped to probabilities, for example, 0.2 is 20% 0.85 is 85%. It can be said that logistic regression creates a probability of an input sitting within a class. For example, is this email spam or not?

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/sigmoid-activation-function.jpg" alt="Logistic Regression also known as the Sigmoid function, an S-shaped curve between y axis values 0 and 1." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Logistic function, also known as the sigmoid function.</p>

We will use scikit-learn's `LogisticRegression()` function implements the technique.

## Train a Logistic Regression Model

We will work with a large dataset, moving on from our built-in datasets for this example, and use the <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">MNIST</a> handwritten digit dataset. 70,000 sample images of handwritten digits which we are going to identify using machine learning. Scikit-learn has a handy function for getting this data `fetch_openml`. The function access the open source machine learning datasets from <a href="https://www.openml.org/" target="_blank">www.openml.org</a>, a large repository hosted by two Dutch universities; <a href="https://www.tue.nl/en/research/research-areas/data-science/" target="_blank">TU Eindhoven</a>, <a href="http://datamining.liacs.nl" target="_blank">Leiden University</a>.

You can follow along in a <a href="https://colab.research.google.com/drive/1uuD2EYS3BTCipvEPgC3vd-2hXdv5UJcZ" target="_blank">Google CoLab</a> live environment.

#### Get data

First, we import the various package we will need.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```
We load the data via `fetch_openml`, you can see futher information on the <a href="https://www.openml.org/d/554" target="_blank">MNIST dataset on OpenML</a> or the <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">original authors page</a>.

```python
# Get MNIST dataset
mnist = fetch_openml('mnist_784')
```

#### Split the dataset

We split the dataset into training and test datasets. MNIST has 60,000 training points and 10,000 testing points. We use a smaller number to decrease train time. The reduction doesn't negatively affect accuracy, this was tested through multiple runs at different sizes.

```python
# Split dataset into 20K training and 10K test data
train_img, test_img, train_lbl, test_lbl = train_test_split(
 mnist.data, mnist.target, train_size=20000, test_size=10000, random_state=0)
```

#### View the data

We can view the MNIST data. We loop through the images and labels in the training dataset and plot them to subplots using matplotlib.

```python
# View the dataset. Loop through image and label and create matplotlib sublots for each
plt.figure(figsize=(15,4))
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
 plt.title('Training: %s\n' % label, fontsize = 20)
```

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/mnist-dataset-example.jpg" alt="Example handwritten digits from the MNIST dataset." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">Examples of handwritten digits from the MNIST dataset.</p>

#### Train the model

Now we can train our logistic regression classifier. We set a specific optimization algorithm (solver), in this case, this one seemed to work best. And a tolerance to stopping criteria, without `tol` set to a small value the model failed to converge, that is finding an optimal fit. We will discuss convergence in the section ahead [Evaluation & Tuning]().

```python
# Assign LogisticRegression() function to variable for use.
# Set to specific solver, and set tolerance for stopping criteria (tol)
lr = LogisticRegression(solver='saga', tol=0.1)
# Train model
lr.fit(train_img, train_lbl)
```

#### View results

We can view a number of prediction by printing out a range, single predictions can be printed passing a single value, e.g `test_img[0]`.

```python
# Predict and print predictions 0 - 10
lr.predict(test_img[0:10])
```

Finally, we print the accuracy of the predictions versus the test data.

```python
# Assign and print accuracy score
score = lr.score(test_img, test_lbl)
print(score)
```

```text
0.9127
```

Our logistic regression model achieved 91% accuracy.


## Summary

We have learned linear regression is a supervised machine learning technique, using labelled data to predict values which have a linear relationship with input data. Due to its simplicity linear regression is not suitable for complex tasks, nor is it suited to classification problems due to its sensitivity to outliers and large differences between input values and prediction output. We implemented and learned a number of functions to create a linear regression model in scikit-learn to predict the price of houses in Boston.

We learned a more flexible technique which allows us to find the line of best fit on datasets which are not suited to straight lines, polynomial regression. The Scikit-learn implementation uses the `LinearRegression()` and `PolynomailFeaturs()` functions to perform this.

Finally, we learned about Logistic Regression. A technique which allows us to work with classification problems using regression, which typically are not suited to techniques such as linear regression. We processed the MNIST dataset to classify hand-draw digits.

On the next page, we learn about neural networks and how they are trained.

---

## References

<ol>

<li>Kaggle (2020a) <i>Boston Housing: Housing Values in Suburbs of Boston</i>. <a href="https://www.kaggle.com/c/boston-housing" target="_blank">https://www.kaggle.com/c/boston-housing</a></li>

<li>Kaggle (2020b) <i>Predict number using Logistic Regression with 92%</i> <a href="https://www.kaggle.com/pranjalsrv7/predict-number-using-logistic-regression-with-92" target="_blank">https://www.kaggle.com/pranjalsrv7/predict-number-using-logistic-regression-with-92</a></li>

<li>LeCun, Y., Cortes, C., and Burges, C. (2002) <i>The MNIST Dataset of handwritten digits</i>. <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">http://yann.lecun.com/exdb/mnist/</a></li>

<li>OpenML (2014) <i>mnist_784</i>. <a href="https://www.openml.org/d/554" target="_blank">https://www.openml.org/d/554</a></li>

<li>Open Data StackExchange (2019) <i>What does “lower status” mean in “Boston house prices dataset”?</i> <a href="https://opendata.stackexchange.com/questions/15740/what-does-lower-status-mean-in-boston-house-prices-dataset" target="_blank">https://opendata.stackexchange.com/questions/15740/what-does-lower-status-mean-in-boston-house-prices-dataset</a></li>

<li>Park, H., Sung-Oh, D., Lee, H., and Hoh, P. (2012) <i>Cyber Weather Forecasting: Forecasting Unknown Internet Worms Using Randomness Analysis</i>. Springer. <a href="https://doi.org/10.1007/978-3-642-30436-1_31">https://doi.org/10.1007/978-3-642-30436-1_31</a></li>

<li>Ray, S. (2019) <i>A Quick Review of Machine Learning Algorithms</i>. IEEE. <a href="https://ieeexplore.ieee.org/document/8862451">https://ieeexplore.ieee.org/document/8862451</a></li>

<li>Scikit-Learn (2020) <i>sklearn.datasets.load_boston</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html</a></li>

<li>Scikit-Learn (2020) <i>sklearn.datasets.fetch_openml</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html</a></li>

<li>Scikit-Learn (2020) <i>sklearn.linear_model.LogisticRegression</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression</a></li>

<li>Scikit-Learn (2020) <i>Linear Models: Logistic Regression</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression" target="_blank">https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression</a></li>

</ol>
