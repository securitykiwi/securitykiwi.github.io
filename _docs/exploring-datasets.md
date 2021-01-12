---
title: Exploring Datasets
author:
---

<style>p {text-align: justify;}</style>

We will explore a small sample of the UGR'16 dataset together in this section. UGR16 contains network traffic informaiton and is designed to test modern Intrusion Detection Systems (IDS).

This page and the next go into detail about useful data manipulation utilities using the python programming libraries <i>Pandas</i> and <i>Scikit-learn</i>. These libraries are bundled with the software <i>Anaconda</i>, if you have followed the <a href="/docs/environment-setup" target="_blank">Environment Setup</a> tutorial you will already Anaconda installed.

* <a href="https://pandas.pydata.org/" target="_blank">Pandas</a> is a data analysis library for the python language which provides a great deal of useful tools and functions for data analysis, including data preperation.
* <a href="https://scikit-learn.org/stable/" target="_blank">Scikit-learn</a> is a machine learning libary which includes a number of preprocessing tools which we will learn to use below.

Follow along in a <a href="/docs/environment-setup" target="_blank">Jupyter Notebook</a> or in a <a href="https://colab.research.google.com/drive/13rMDvYJnpMY52hSaLItdP1j_521mmxKT" target="_blank">Google CoLab</a> live environment.

## Get the Data

The UGR16 dataset is huge, much like all of the datasets listed in <a href="/docs/existing-datasets-and-data-sources/" target="_blank">Existing Datasets & Data Sources</a>. The 'July Week 5' csv file is _52GB uncompressed_, so I have provided <a href="" target="_blank">the first 5,000 rows</a> for us to look at, it's only 483kb.

{% include video.html %}

You can upload files into Jupyter Notebook using the 'upload' button on the right.

![](https://securitykiwi.b-cdn.net/images/jupyter-upload.png)

You can also use code to download a dataset, allowing reusability and automaiton. We will use this method throughout the course. You will build up a series of these code 'tools' as you work through this course which will be useful for your own projects. The script below will download the dataset into the working directory of Jupyter Notebooks, so you can access the dataset from within a notebook. The code follows a popular format of modularity, with seperate variables allowing slightly increase ease of use.

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

`DOWNLOAD_REPO` is the URL a repository containing datasets, `DOWNLOAD_FILENAME` is the name of the file we want to download contained in that repository, these are combined in line 2. `DATASET_FILENAME` allows you to get the filename when it is created locally. We then use the `requests` library to fetch the dataset, check for errors (`.raise_for_status()`), create a <i>file object</i> using `open()`, create a <i>file writer</i> `write()` using the content of the request, and finally print a message so we know when it's done.

## Explore the Data

Now we have the csv file 'ugr16-july-week5-first5k' in Jupyter we can use Pandas to read the csv and start exploring it. Pandas is a Python library used for data manipulation and analysis, with particularly useful functionality such as dataframes, a data structure containing rows and columns.

## Get an Overview of a Dataset

Let's get a summary of the data within a dataset. 

{% include video.html %}

### Data Overview

We `import` pandas as the variable `pd` (a convention), read the contents of the CSV file into the variable `df` (stands for dataframe (another convention)) and we use the `info()` method on df which provides a basic summary of the dataframe.

```python
import pandas as pd
df = pd.read_csv("ugr16-july-week5-first5k.csv")
df.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4999 entries, 0 to 4998
Data columns (total 13 columns):
2016-07-27 13:43:21    4999 non-null object
48.380                 4999 non-null float64
187.96.221.207         4999 non-null object
42.219.153.7           4999 non-null object
53                     4999 non-null int64
53.1                   4999 non-null int64
UDP                    4999 non-null object
.A....                 4999 non-null object
0                      4999 non-null int64
0.1                    4999 non-null int64
2                      4999 non-null int64
209                    4999 non-null int64
background             4999 non-null object
dtypes: float64(1), int64(6), object(6)
memory usage: 507.8+ KB
```

`info()` shows us information about each column in the dataset as rows in this output. General information is provided, number of entries (remember most data structures count from 0, 0 to 4,999 means there are 5,000 entries), 13 columns, memory usage and information about those 13 columns. The first column in `info()` is the heading of the dataset columns (in this case the dataset creators didn't use headings), the second is the number of instances of a record, third the type of entry (in this case it cannot be null) and then the data type. More informaiton can be found on the documentation about the <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html" target="_blank">info()</a> function.

### Visual Overview of Numerical Data

Let's visualise the different numerical data within the dataset using <i>matplotlib</i> and its <i>histogram</i> feature.

```python
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(30,15))
plt.show()
```

![](https://securitykiwi.b-cdn.net/images/ugr16-histogram.png)


## View Data from the Dataset

So far we have used two methods to get an overview of the data within the dataset. Lets actually view some of the data. 

### View Data Snippets

The `head()` function prints the first <i>n</i> rows from a dataset, the default is 5, however, you can pass values within the parenthases (e.g. `head(50)` for the first 50). The `tail()` function shows entires from the end of a dataset. Viewing snippets like this allows us to see the actual values within our dataset without viewing the whole thing - with datasets in the order of gigabytes, opening such large files can be a task in itself.

You may need to scroll right to see all of the columns in the table below.

```python
df.head()
```

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/ugr16-dataset-head-5.png" alt="View of the first five rows of the URG16 dataset." style="width:700px;"/></div>

<p style="text-align: center; font-style: italic;">First five rows of the UGR16 Dataset.</p>

The UGR'16 dataset is a netflow capture of real traffic data from an ISP and synthetic attack data. When the dataset is visualised you'll notice it has no column headings, it's just data. To figure out what we're looking at we search the research paper which first presented the URG'16 dataset. In the seciton describing their creation methodology we find the tool they used (nfdump) and they describe the data represtened in each column.

* Date and time of the end of a flow
* Duration of the flow
* Source IP Address
* Destination IP Address
* Source Port
* Destination Port
* Protocol
* Flag
* Forwarding status
* Type of Service (ToS) byte
* Packets exchanged
* Bytes exchanged
* Label

### Add Column Headings

We can add the column headings to the individual columns to aid our understanding as we examine the dataset. These need to be removed later when we feed our data into our machine learning algorithm. Below, we assign the names of the 13 columns as a list to the dataframe.

```python
df.columns = ['Date time', 'Duration', 'Source IP',
              'Destination IP', 'Source Port', 'Destination Port',
              'Protocol', 'Flag', 'Forwarding status', 'ToS',
              'Packets', 'Bytes', 'Label']
df.head()
```

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/ugr16-dataset-dead-5-with-headings.png" alt="View of column names added to pandas DataFrame." style="width:700px;"/></div>

<p style="text-align: center; font-style: italic;">Column headings added to UGR16 Dataset dataframe.</p>


## Summary

This page has described practical skills to explore datasets, allowing you to view and understand the data you wish to work with. Exploring datasets is a key stage in any project, the skill is used to decide if a dataset is appropriate for your needs. Once you have decided to move forward with a specific dataset exploring the data allows you to see what needs to be altered and fixed - its highly unlikely any dataset you choose will be perfect for your chosen project.

The next page discussed necessary skills and techniques you can employ to make these transformations and fixes to your chosen dataset.


---

<p style="text-align: center;">Feedback is welcome!</p>

<p style="text-align: center;">Get in touch <a href="mailto:securitykiwi@protonmail.com">securitykiwi [ at ] protonmail.com</a>.</p>

---

## References

<ol>
    
<li>Maciá-Fernández, G., Camacho, J., Magán-Carrión, R., García-Teodoro, P., and Therón, R. (2018) <i>UGR‘16: A new dataset for the evaluation of cyclostationarity-based network IDSs</i>. Elsevier.
 <a href="https://www.sciencedirect.com/science/article/pii/S0167404817302353" target="_blank">https://www.sciencedirect.com/science/article/pii/S0167404817302353</a></li>

<li>McKinney, W. (2017) <i>Python for Data Analysis</i>. O'Reilly Media. <a href="https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662" target="_blank">https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662</a></li>

<li>NFDUMP (2014) NFDUMP Overview. <a href="http://nfdump.sourceforge.net/" target="_blank">http://nfdump.sourceforge.net</a></li>

<li>University of Granada (2016) <i>UGR'16: A New Dataset for the Evaluation of Cyclostationarity-Based Network IDSs.</i> <a href="https://nesg.ugr.es/nesg-ugr16/" target="_blank">https://nesg.ugr.es/nesg-ugr16/</a></li>
    
</ol>
