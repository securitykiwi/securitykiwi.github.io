---
title: Dataset Preparation
author: 
---

<style>p {text-align: justify;}</style>

Data needs to be prepared and cleansed before you process it using a machine learning algorithm. Datasets often include duplicate records, unfinished entries or other aspects which are undesirable and may _skew or prevent your model from working as expected_.

On this page, we will work on a number of practical exercises cleaning up a number of common problems in a dataset using Pandas and Scikit-learn. We follow on from the previous page which discussed how to initially explore a dataset to gain valuable insight into a dataset. This page assumes you have followed the instructions in <a href="/docs/environment-setup/" target="_blank">Environment Setup</a>, which describes the steps to install Anaconda with Jupyter Notebooks.

Each section on this page has an associated Google CoLab live environment specific to the code discussed. Please experiment with the code in Jupyter Notebooks or Google CoLab.

#### Contents

* [Handing Non-Numerical Data](#handing-non-numerical-data)
* [Feature Scaling](#feature-scaling)
* [Merge Datasets](#merge-datasets)
* [Remove a Column](#remove-a-column)
* [Remove Duplicate Entries](#remove-duplicate-entries)
* [Replace Values](#replace-values)
* [Filter Outliers](#filter-outliers)


### First Steps

{% include video.html %}

Let's get some data to work on:

```python
import requests

DOWNLOAD_REPO = "https://raw.githubusercontent.com/krisbolton/machine-learning-for-security/master/"
DOWNLOAD_FILENAME = DOWNLOAD_REPO + "ugr16-july-week5-first5k.csv"
DATASET_FILENAME = "ugr16-july-week5-first5k.csv"

response = requests.get(DOWNLOAD_FILENAME)
response.raise_for_status()
with open(DATASET_FILENAME, "wb") as f:
    f.write(response.content)
print("Download complete.")
```

Check the data is OK:

```python
import pandas as pd
df = pd.read_csv("ugr16-july-week5-first5k.csv")
df.info()
```

The output should look like:

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4999 entries, 0 to 4998
Data columns (total 13 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   2016-07-27 13:43:21  4999 non-null   object 
 1   48.380               4999 non-null   float64
 2   187.96.221.207       4999 non-null   object 
 3   42.219.153.7         4999 non-null   object 
 4   53                   4999 non-null   int64  
 5   53.1                 4999 non-null   int64  
 6   UDP                  4999 non-null   object 
 7   .A....               4999 non-null   object 
 8   0                    4999 non-null   int64  
 9   0.1                  4999 non-null   int64  
 10  2                    4999 non-null   int64  
 11  209                  4999 non-null   int64  
 12  background           4999 non-null   object 
dtypes: float64(1), int64(6), object(6)
memory usage: 507.8+ KB
```

## Handing Non-Numerical Data

_Most machine learning methods can't handle non-numerical data_. Feeding a deep neural network a vector of words containing animals (`[fish, dog, cat, owl, ...]`) and attempting to predict the species wouldn't work, the technique simply can't understand non-numerical data. Encoding datasets to numerical values unlocks the use of most machine learning algorithms. Later in the <a href="/docs/introduction-to-training-models/" target="_blank">Training Models</a> section we work with the Iris dataset, containing measurement values on three different species of flower. The dataset doesn't include the Latin names of species, they have been encoded as integers. A snippet of a vector which may read `[iris setosa, iris virginica, iris versicolor]` in Latin now reads `[0, 1, 2]`, allowing the data to be processed by a neural network.

Within security datasets, non-numerical data often represents a category of data, a protocol type, or a label denoting the type of traffic. Below we will encode this non-numerical categorical data to numerical categorical representations, UDP becomes 0, TCP becomes 1, for example. First, there are two types of data we will deal with, <i>ordinal</i> and <i>nominal</i>, each needs to be processed in a different way. 

**Ordinal data** has an order to the category (i.e. there is a relationship between values) and the distance between the categories is not known. An example of ordinal data is satisfaction, from 1-5 how satisfied were you? 1. very unsatisfied, 2. unsatisfied, 3. neural, 4. satisfied, 5. very satisfied.

**Nominal data** has distinct categories without order, i.e. there is no relationship between values. An example of nominal data is protocol types, UDP, TCP are distinct categories, there is no value in the order of the data.

You may have noticed in the output of our dataset above we have a non-numerical data type (dtype) an object with a number of them representing categories - background, a category of traffic, UDP, the protocol category and the .A... which is one representation of a flag. We can replace these non-numerical values with encoded values representing categories which our machine learning systems can understand.

Follow along with handling non-numerical data in a <a href="https://colab.research.google.com/drive/1qOWJYHmsG4ITumFmH0kTKO-YB8NaeKdR" target="_blank">Google CoLab</a> live environment.

### Ordinal Data

{% include video.html %}

Scikit-learn provides the function `OrdinalEncoder()` which encodes ordinal categorical features as an integer array. 

```python
from sklearn.preprocessing import OrdinalEncoder
protocol_col = df['UDP']
protocol_col
```

Our single protocol column looks like this before transformation:

```text
0       UDP
1       TCP
2       TCP
3       TCP
4       UDP
       ... 
4994    UDP
4995    UDP
4996    UDP
4997    UDP
4998    UDP
Name: UDP, Length: 4999, dtype: object
```

We transform the column and assign the result to col_encoded. The `values.reshape(-1,1)` is required, otherwise scikit-learn throws an error about 1D data, as it expects 2D. You can find solutions like this by Googling the process you're trying to accomplish, or the error, there's always a <a href="https://stackexchange.com/" target="_blank">StackExchange</a> Q&A or medium article in my experience.

```python
encode = OrdinalEncoder()
col_encoded = encode.fit_transform(protocol_col.values.reshape(-1,1))
col_encoded
```

Encoded column:

```text
array([[2.],
       [1.],
       [1.],
       ...,
       [2.],
       [2.],
       [2.]])
```

Now we have to replace the unencoded column in our dataframe with the encoded data. Pandas provide the `assign()` function for just for this - it will create a copy of the dataframe with a replaced column to be specific. Many of these data science libraries act in this safe way, creating new objects rather than alter an original. Always check the documentation if you need safety though.

```python
encoded_col_df = df.assign(UDP=col_encoded)
encoded_col_df
```

On the dataframe `df` we assign the new column `col_encoded` to replace the column `UDP` and assign the result to `encoded_column_df`. `encoded_col_df`:

```text
	    2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0	0.1	2	209	background
0	    2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	    2.0	.A....	0	0	2	167	background
1	    2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	1.0	.AP...	0	0	9	2082	background
2	    2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	1.0	.AP...	0	0	9	7118	background
3	    2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	1.0	.AP...	0	0	4	674	background
4	    2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	2.0	.A....	0	0	2	188	background
...	...	...	...	...	...	...	...	...	...	...	...	...	...
4994	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	52389	2.0	.A....	0	0	1	93	background
4995	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	54637	2.0	.A....	0	0	1	93	background
4996	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	58070	2.0	.A....	0	0	1	105	background
4997	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	58304	2.0	.A....	0	0	1	88	background
4998	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	58331	2.0	.A....	0	0	1	101	background
```

As we can see, UDP has been replaced by the value 2.0 and TCP by the value 1.0.

### Nominal Data

Scikit-learn provides the function `OneHotEncoder()` to encode nominal data. It is named after the technique which it uses, creating binary attributes per category; one value equal to 1 (hot) and another 0 (cold), or "one-hot encoding". So why would you use one-hot encoding over another method? It might appear subtle but encoding methods such as `OrdinalEncoder()` will make your machine learning model assume there is a relationship between values and produce poor results. `OneHotEncoder()` ensures there is no way relationships can be inferred when there are none.

An example of one hot encoded data:

```text
Cyan  Magenta  Yellow  Black
 1       0        0      0
 0       1        0      0
 0       0        1      0
 0       0        0      1
```

<i>n</i> categorical features are encoded into <i>n</i> binary features. Four colours, so four categories of binary features are created.

```text
Colour  |  Category
--------|-----------
Cyan    |     1
Magenta |     2
Yellow  |     3
Black   |     4
```

How is one hot encoding better than the above? Simply encoding a numerical value per category. Machine learning models only act on the data given, they are not smart. Internal calculations within the system would infer meaning which does not exist and ultimately skew the output of the model. An illustrative example, 1 + 2 = 3, cyan + magenta doesn't equal yellow. We happen to be working with an example in which the categories could combine to make another, however, working with other objects, predicting the price of stocks for example. One company + another does not make another. It creates bizarre and meaningless results. One hot encoding fixes this issue.

#### One hot encoding UGR'16

Lets apply one-hot encoding using the `OneHotEncoder()` function to a column from our UGR'16 sample dataset to see how it's done.

We import `OneHotEncoder`, assign the UDP column to a variable named `protocol_col` and print the result.

```python
from sklearn.preprocessing import OneHotEncoder
protocol_col = df['UDP']
protocol_col
```
The contents of `protocol_col`, the same as our previous example.

```text
0       UDP
1       TCP
2       TCP
3       TCP
4       UDP
       ... 
4994    UDP
4995    UDP
4996    UDP
4997    UDP
4998    UDP
Name: UDP, Length: 4999, dtype: object
```
Now we assign the `OneHotEncoder()` function to the `hot_encoder` variable, applying the encoding on the `protocol_col`. Once again, the `values.reshape(-1, 1)` is required to reshape the data for the encoder, otherwise, an error is thrown.

```python
hot_encoder = OneHotEncoder()
hot_encoded_col = hot_encoder.fit_transform(protocol_col.values.reshape(-1,1))
hot_encoded_col
```
`hot_encoded_col` contains a SciPy (another python library) <i>sparse matrix</i>, which only stores the location of the non-zero elements vastly reducing the demand for memory. We can convert it to a NumPy array with the `toarray()` function so we can see the binary matrix we expect.

```python
hot_encoded_col.toarray()
```

```text
array([[0., 0., 1.],
       [0., 1., 0.],
       [0., 1., 0.],
       ...,
       [0., 0., 1.],
       [0., 0., 1.],
       [0., 0., 1.]])
```


More exmaples (non-security related) on encoding are discussed in scikit-learn's documentation on pre-processing categorical features: <a href="https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features" target="_blank">encoding categorical features</a>.

## Feature Scaling

Feature scaling solves an important issue when working with data of different scales in machine learning, most algorithms are not optimised to work well when they compare values of greatly differing scales, where one input is measured in double digits (e.g. 50) and another in five (e.g. 50,000). We will discuss two methods to fix this issue; normalization and standardization.

Follow along with normalization and standardization in a <a href="https://colab.research.google.com/drive/17HvlZHnolaLUyEWhZ_-w9z3l9WcNtphB" target="_blank">Google CoLab</a> live environment.

### Normalization

Scikit-learn provides a function called `MinMaxScaler()` which rescales values to range from 0 to 1 by subtracting the minimum value and dividing by the maximum value minus the minimum, hence MinMax.

In this example, we will scale the flow duration (labelled 48.380) column from our UGR'16 sample dataset.

```python
from sklearn.preprocessing import MinMaxScaler
duration_col = df['48.380']
duration_col
```

```text
0       48.380
1       50.632
2       51.052
3       46.996
4       48.852
         ...  
4994     0.000
4995     0.000
4996     0.000
4997     0.000
4998     0.000
Name: 48.380, Length: 4999, dtype: float64
```
We assign the `MinMaxScaler()` to the `scaler` variable, reshaping the data as in previous examples and fitting and transforming before the scaler scales the values.

```python
scaler = MinMaxScaler()
scaled_col = scaler.fit_transform(duration_col.values.reshape(-1,1))
scaled_col
```

```text
array([[0.41327821],
       [0.43251555],
       [0.43610333],
       ...,
       [0.        ],
       [0.        ],
       [0.        ]])
```


### Standardization

Scikit-learn provides a function called `StandardScaler()` which scales data by subtracting the mean value, so the value has a zero mean, and divides by the standard deviation to create unit variance among the values. So standardization does not compress values between a certain range, like MinMax, it scales values by changing the value to a new 'centre point' and creating variance based on the standard deviation between values so the data retains its shape or meaning. 

A quick reminder on standard deviation and mean if its been a while. Standard deviation is the measure of how much variation there is between values, a low standard deviation shows values are close together (close to the mean), a high standard deviation shows values are spread apart. Mean is the total of the values divided by how many values there are.

In our example below we will standardise the flow duration column:

```python
from sklearn.preprocessing import StandardScaler
duration_col = df['48.380']
duration_col
```

```text
0       48.380
1       50.632
2       51.052
3       46.996
4       48.852
         ...  
4994     0.000
4995     0.000
4996     0.000
4997     0.000
4998     0.000
Name: 48.380, Length: 4999, dtype: float64
```
Assign the `StandardScaler()` to the `std_scaler` variable, reshaping the data as in previous examples and fitting and transforming before the scaler scales the values, much like our last example.

```python
std_scaler = StandardScaler()
std_scaled_col = std_scaler.fit_transform(duration_col.values.reshape(-1,1))
std_scaled_col
```

```text
array([[10.24103278],
       [10.72441497],
       [10.81456618],
       ...,
       [-0.14352779],
       [-0.14352779],
       [-0.14352779]])
```


## Merge Datasets

Sometimes you may need to merge two or more datasets into one. Pandas has a handy `merge` function  for <i>DataFrames</i> which allows us to perform database-style merges. If you're unfamiliar with "database-style" merges, it refers to a number of different types of merge which offer a slightly different outcome. These are called "<i>joins</i>"; inner join, outer join, left join, right join or inner, outer, left and right, respectively. A DataFrame is a data structure which resembles a two-dimensional table containing columns and rows, Pandas DataFrames come with a number of functions to allow easy manipulation.

Follow along with merging datasets in a <a href="https://colab.research.google.com/drive/1LYMAB4c_BmUCpvC_ZkHdYJBODseR6u19" target="_blank">Google CoLab</a> live environment.

Example of a DataFrame in Jupyter Notebooks:

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/pandas-dataframe-example.jpg" alt="An example of a Pandas DataFrame displaying the UGR'16 dataset.." style="width:600px;"/></div>

<p style="text-align: center; font-style: italic;">A DataFrame containing the UGR'16 dataset.</p>


Performing a merge of two datasets:

We have two tiny datasets here, the first and last 5 rows of the 5k rows UGR'16 dataset from the previous page, stored as `df1` and `df2`.

df1

```python
df1 = df.head(5)
df1
```

```text
    2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0	0.1	2	209	background
0   2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	UDP	.A....	0	0	2	167	background
1	2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	TCP	.AP...	0	0	9	2082	background
2	2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	TCP	.AP...	0	0	9	7118	background
3	2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	TCP	.AP...	0	0	4	674	background
4	2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	UDP	.A....	0	0	2	188	background
```

df2

```python
df2 = df.tail(5)
df2
```

```text
     	2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0	0.1	2	209	background
4994	2016-07-27 13:43:35	0.0	143.72.8.137	42.219.154.98	53	52389	UDP	.A....	0	0	1	93	background
4995	2016-07-27 13:43:35	0.0	143.72.8.137	42.219.154.98	53	54637	UDP	.A....	0	0	1	93	background
4996	2016-07-27 13:43:35	0.0	143.72.8.137	42.219.154.98	53	58070	UDP	.A....	0	0	1	105	background
4997	2016-07-27 13:43:35	0.0	143.72.8.137	42.219.154.98	53	58304	UDP	.A....	0	0	1	88	background
4998	2016-07-27 13:43:35	0.0	143.72.8.137	42.219.154.98	53	58331	UDP	.A....	0	0	1	101	background
```

Now lets merge them. Note we are using the <i>outer join</i> type, the <i>union</i> of the two tables, i.e. all the rows of each combined. The default behaviour of merge is an inner join, where the merge uses the intersection of keys to perform the merge, i.e. the rows they have in common. There are further nuances and differences compared to database joins, which we won't discuss here. See this <a href="https://stackoverflow.com/questions/38549/what-is-the-difference-between-inner-join-and-outer-join" target="_blank">StackExchange Q&A</a> for a quick guide to joins, and see the Pandas documentation on <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging" target="">Database-style DataFame joining/merging</a> for a more comprehensive explanation.

```python
df_merged = pd.merge(df1, df2, how='outer')
df_merged
```

Merged output:

```text
    2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0	0.1	2	209	background
0	2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	UDP	.A....	0	0	2	167	background
1	2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	TCP	.AP...	0	0	9	2082	background
2	2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	TCP	.AP...	0	0	9	7118	background
3	2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	TCP	.AP...	0	0	4	674	background
4	2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	UDP	.A....	0	0	2	188	background
5	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	52389	UDP	.A....	0	0	1	93	background
6	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	54637	UDP	.A....	0	0	1	93	background
7	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	58070	UDP	.A....	0	0	1	105	background
8	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	58304	UDP	.A....	0	0	1	88	background
9	2016-07-27 13:43:35	0.000	143.72.8.137	42.219.154.98	53	58331	UDP	.A....	0	0	1	101	background
```

## Remove a Column

Pandas provides the `drop` function to remove information from a DataFrame. Below we will remove a column from our tiny UGR'16 dataset snippet.

View code in a <a href="https://colab.research.google.com/drive/1Tjg8KZGnl8Vije_-j6087T8Owpkdm5qk" target="_blank">Google CoLab</a> live environment.

Current dataframe:

```python
df3 = df.head(5)
df3
```

For argument's sake, lets remove the 9th column with the column heading '0', which contains entries of just zeros.

```python
df3.drop(columns='0')
```

Dataframe after `drop`:

```text
    2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0.1	2	209	background
0	2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	UDP	.A....	0	2	167	background
1	2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	TCP	.AP...	0	9	2082	background
2	2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	TCP	.AP...	0	9	7118	background
3	2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	TCP	.AP...	0	4	674	background
4	2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	UDP	.A....	0	2	188	background
```
You can drop multiple columns at once, `df.drop(columns=['NAME', 'NAME'])` for example. For more, see the Pandas documentation on the <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html" target="_blank">pandas.DataFrame.drop</a> function.

## Remove Duplicate Entries

Pandas provides a few useful functions for assessing and dealing with duplicate entries within a dataset. First, lets create an example dataset with duplicated data. We'll do this by adding the first 5 rows of the UGR'16 dataset to a new dataframe twice.

Follow along in a <a href="https://colab.research.google.com/drive/1s98VS73JeyJkGjsSRvUzNFbgjgHrHpzq" target="_blank">Google CoLab</a> live environment.

```python
# create two dataframes of duplicates
dup1 = df.head(5)
dup2 = df.head(5)

# the concatenate function requires an iterable single object
frames = [dup1, dup2]

# concatinate the dataframes dup1 and dup2 and assign to df4
df4 = pd.concat(frames)

# print df4
df4
```

```text
    2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0	0.1	2	209	background
0	2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	UDP	.A....	0	0	2	167	background
1	2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	TCP	.AP...	0	0	9	2082	background
2	2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	TCP	.AP...	0	0	9	7118	background
3	2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	TCP	.AP...	0	0	4	674	background
4	2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	UDP	.A....	0	0	2	188	background
0	2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	UDP	.A....	0	0	2	167	background
1	2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	TCP	.AP...	0	0	9	2082	background
2	2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	TCP	.AP...	0	0	9	7118	background
3	2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	TCP	.AP...	0	0	4	674	background
4	2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	UDP	.A....	0	0	2	188	background
```

Check for duplicate rows using the `duplicated()` function which outputs a boolean (true/false) check on each row.

```python
df4.duplicated()
```

```text
0    False
1    False
2    False
3    False
4    False
0     True
1     True
2     True
3     True
4     True
dtype: bool
```

Pandas provides an equally easy way to remove duplicate rows, `drop_duplicates()`.

```python
df4.drop_duplicates()
```

```text
2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0	0.1	2	209	background
0	2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	UDP	.A....	0	0	2	167	background
1	2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	TCP	.AP...	0	0	9	2082	background
2	2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	TCP	.AP...	0	0	9	7118	background
3	2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	TCP	.AP...	0	0	4	674	background
4	2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	UDP	.A....	0	0	2	188	background
```

You can also remove duplicates per column, rather than by row, by passing the column name:

```python
df.drop_duplicates(['COLUMN'])
```

## Replace Values

Sometimes we need to replace values. We can use techniques such as those described in the handling non-numerical data, however, there are many reasons we may want to replace values so we discuss more techniques here.

You can follow along in a <a href="https://colab.research.google.com/drive/1nnbscoGTZUdsScKBdRF-gpzGRgAv354S" target="_blank">Google CoLab</a> live environment.

### Replace Values with Panda's

Pandas provides the function `replace()` to find and replace data values. Lets create a new dataframe and replace the non-numerical values 'UDP' and 'TCP' with a representational number 0. Many machine learning networks do not handle non-numerical data well, we discuss this in a section below.

Create a new dataframe with the first 5 rows from UGR'16:

```python
df5 = df.head(5)
df5
```

Replace UDP and TCP with representational values:

```python
df5.replace(['UDP','TCP'], [0, 1])
```

DataFrame after replacement:

```text
    2016-07-27 13:43:21	48.380	187.96.221.207	42.219.153.7	53	53.1	UDP	.A....	0	0.1	2	209	background
0	2016-07-27 13:43:21	48.380	42.219.153.7	187.96.221.207	53	53	0	.A....	0	0	2	167	background
1	2016-07-27 13:43:25	50.632	42.219.153.191	62.205.150.146	80	1838	1	.AP...	0	0	9	2082	background
2	2016-07-27 13:43:25	51.052	62.205.150.146	42.219.153.191	1838	80	1	.AP...	0	0	9	7118	background
3	2016-07-27 13:43:27	46.996	92.225.28.133	42.219.155.111	443	59867	1	.AP...	0	0	4	674	background
4	2016-07-27 13:43:27	48.852	143.72.8.137	42.219.154.107	53	60019	0	.A....	0	0	2	188	background
```

You can also use the `replace()` function to replace a single value rather than two (or more) as we did. For example, instances of 'UDP' with 0.

```python
df.replace('UDP', 0)
```
You can find more information in the Pandas documentation on the <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html" target="_blank">panadas.DataFrame.replace</a> function

### Replace Missing Values

Datasets may include missing values, which we may decide to replace (or remove). Replacing missing values is such a common task scikit-learn has the `SimpleImputer()` function which is designed for this task.

In our example below, the number -1 represents the unknown value when a sensor has not recorded any information. Typically, our sensor reports values between 0 and 100. We have decided to replace all of these values with the median value. This may not be appropriate for all datasets, this is just an example of a tool at your disposal.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=-1, strategy='median')
imputer = imputer.fit([[24, -1, 4, 87], 
                      [77, -1, 8, 13],
                      [44, 43, 22, -1]])
x = ([[24, -1, 4, 87], 
     [77, -1, 8, 13],
     [44, 43, 22, -1]])
imputer.transform(x)
```

## Filter Outliers

Outliers can affect the efficiency of machine learning algorithms, skewing how an algorithm learns when it is exposed to dramatically different data points.

Let's create a new dataframe, this time with 50 rows.

```python
df6 = df.head(50)
df6
```

### Filtering Outliers: By Range

In our example, we want to create a subset of our data which has eliminated outliers in the source port column of our data, which is labelled as '53'. In the example below, we assign the column to the variable source_port and print its contents.

```python
source_port = df6['53']
source_port
```

For our example purpose, we consider any port number over 10,000 to be considered an outlier. We filter them using the less-than operator (<) during the NumPy function abs (absolute) which in this context gives us the individual values of entries for comparison using our less-than operator.

```python
import numpy as np
filtered_src_port = source_port[np.abs(source_port) < 10000]
filtered_src_port
```

### Filtering Outliers: By Value

In this example we want to filter out specific values and create a new dataframe without those values. For our example, we do not want TCP traffic in our new dataframe.

Follow along in a <a href="https://colab.research.google.com/drive/1kEpIvuC0vjnfCQeg9-SC0lj27TtU3zXV" target="_blank">Google CoLab</a> live environment.

New dataframe with all columns:

```python
df7 = df.head(10)
df7
```

Confusingly, our protocol column is called UDP. This is because the UGR'16 dataset doesn't contain headings, so the first row of values becomes the column headings in our dataframe.

```python
no_tcp_df = df7[df7.UDP != 'TCP']
no_tcp_df
```

## Summary

We have discussed a number of specific techniques designed to fix common issues in datasets. You will find these skills are useful for most projects you work on, and we will use many of these techniques in projects later on in this course.

The next section introduces you to Training Models, this is where you will create your first machine learning algorithm and be exposed to neural networks and key concepts relating to building and using machine learning systems.

---

## References

<ol>
    
<li>McKinney, W. (2017) <i>Python for Data Analysis</i>. O'Reilly Media. <a href="https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662" target="_blank">https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662</a></li>      
<li>Pandas (2020a) <i>pandas.DataFrame</i>. <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html" target="_blank">https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html</a></li>

<li>Pandas (2020b) <i>pandas.DataFrame.assign</i>. <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html" target="_blank">https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.assign.html</a></li>

<li>Pandas (2020c) <i>pandas.DataFrame.merge</i>. <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html" target="_blank">https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html</a></li>

<li>Pandas (2020d) <i>pandas.DataFrame.replace</i>. <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html" target="_blank">https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html</a></li>

<li>Pandas (2020e) <i>User Guide: Database-style DataFrame joining/merging</i>. <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging" target="_blank">https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging</a></li>  

<li>Scikit-learn (2020a) <i>sklearn.preprocessing.OrdinalEncoder</i>. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html</a></li>  

<li>Scikit-learn (2020b) <i>sklearn.preprocessing.OneHotEncoder</i>. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html</a></li>

<li>Scikit-learn (2020c) <i>sklearn.preprocessing.MinMaxScaler</i>. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html</a></li>

<li>Scikit-learn (2020d) <i>sklearn.preprocessing.StandardScaler</i>. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html</a></li>

<li>Scikit-Learn (2020e) <i>sklearn.impute.SimpleImputer</i>. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html</a></li>

</ol>
