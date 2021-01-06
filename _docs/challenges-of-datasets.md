---
title: Challenges of Datasets
author:
---

<style>p {text-align: justify;}</style>

There are several challenges within datasets to consider which may adversely effect the quality of the output of a machine learning model. This page disucsses the most common issues you will come across and must consider each time you use or create a dataset for machine learning. 

## Data Quality

_The quality of the system is only as good as the quality of the data it uses_. Exploring datasets before using them and searching for common issues is a required step before deciding to use a dataset. Most datasets you obtain or make yourself will need to be cleansed in some way. Incomplete entries may have been acceptable during the dataset creation, however, machine learning algorithms are sensitive to this type of issue and you will have to decide if you are to ignore entires which include missing data or to fill in the missing data (for example with average values (don't make up values)). Similarly, outliers will have to be considered and a decision made to remove them, such unique instances may adversely effect the training of the model.

These challenges of data quality can be expressed in a number of ways. Below are some of the most common data quality considerations:

* **Accessibility** - a dataset may be restricted, and difficult to access. Typically, this doesn't bode well if you need more informaition on how the data was obtained.
* **Timeliness** - the time delay from the creation of the dataset to its utilization.
* **Authorization** - whether an individual has the right to use the data.
* **Credibility** - the reliability of the author(s) and the methodology emplouyed on the data.
* **Documentation** - comprehensive documentation is needed to fully evaluate a dataset. Definitions, ranges of valid values, standard formats, business rules, etc. 
* **Accuracy** - accuracy can be difficult to measure when there is no known reference value, when this occurs accuracy should be considered in the content of the application of the data.
* **Integrity** - in this context data integrity refers to the completeness of a particular data point. Data values are standardized and all characteristics are correct.
* **Completeness** - can description a data point with multiple components, refering to the values of all components of a single datum are valid. 
* **Fitness** - refers to how well the dataset matches a users needs.
* **Readability** - simply, is the data readable? Can the contents be understood and explained by known terms, attributes, units, abbreviations, etc?
* **Structure** - the difficulty in transforming the most common types of data - semi-structured or unstructured data - to structured data.

Each of these elements should be considered for your data. Some will be more applicalbe than others, but all should be considered.

## Data Quantity

The quantity of data is important for machine learning algorithms. Sufficient data should exist to split a single dataset into at least two datasets; one for training and one for testing. The amount of data required is down to the algorithm in use, typically thousands of data points are needed, however, for some algorithms millions of data points may be required. Quantity will be subjective to the technique you are using and the data you have access to. We can find useful information for our own projects in the work others have done before us and understanding the technical needs of the particular technique.

## Representitive Data

How well the data within a dataset represents the data which your model will operate on should be considered. For example, arguments have been made that the DARPA datasets are unrepresentitive of todays network traffic due to their age (21 years+). This unrepresentitive data may perform well, or poorly on test data, leading researchers / programmers to make incorrect deductions based on its performance. These logical errors are difficult to diagnose when they happen, output will look fine but the system may not perform as expected in all circumstances. This is why we work in stages and move systematically through each, creating and testing against requirements to reduce the probability of significant mistakes.

When considering an existing dataset it is good practice to search Google and open academic research from <a href="https://scholar.google.com/" target="_blank">Google Scholar</a>, <a href="https://ieeexplore.ieee.org" target="_blank">IEEE Xplore</a>, <a href="https://dl.acm.org/" target="_blank">ACM Digital Library</a> and others. Searching the name of the dataset and looking for researchers discussing its use, and appending your search with "limitations" - specificly looking for researchers discussing the limitations of the dataset. The page <a href="/resources/how-to-conduct-research/" target="_blank">How to Conduct Research</a> discusses how to research in detail for those who wish to continue beyond this course with a scientific mindset.

## Niche Applications

Applications with a relatively small or recently developed research base may not have created robust datasets. For example, my topic of research in higher education focused on the detection of <i>advanced persistent threats</i> (APT) and their movement through a network. Intrusion detection datasets exist which on the surface appear promising, but once you have created a set of requirements, a list of 13 datasets decreased to 1. This final dataset still didn't match my needs entirely. The nature of APTs means they attack a network over months, most intrusion datasets only cover several days. In addition, APTs often use zero-day exploits, most datasets don't include these 'advanced' attack techniques.

In this case you have two choices. Create your own dataset by recording traffic from an appropraitely designed test network over an approrpaite period - a project in itself. Or you can break your system down and test elements on approrpaite datasets. The latter opens up greater oppurtunity for logical errors on your part - you may miss elements for testing which greatly affect performance. However, it will likely be faster than creating a robust dataset yourself.

## Summary

This page has introduced you to a number of common challenges with datasets in machine learning. Some are difficult to diagnose logical issues which won't be readily apparently until your system produces output you weren't expecting. These issues are mitigated by the creation and use of structure in your project. The main method of strucutre we discussed was the use of requirements, where we create a list of elements which must be present to ensure the output of our system meets our goals and purpose, remains consistnet and is suitable for our chosen algorithm.

The remainder of the challenges we discussed are mitigated by awareness and skill, which you have gained by reading this page and will gain by using the skills you learn from the following two pages, which provide you with practical python skills to explore and prepare data for use.

---

## References

<ol>

<li>Cai, L., and Zhu, Y. (2015) <i>The Challenges of Data Quality and Data Quality Assessment in the Big Data Era</i>. Data science Journal. <a href="https://datascience.codata.org/articles/10.5334/dsj-2015-002" target="_blank">https://datascience.codata.org/articles/10.5334/dsj-2015-002</a></li>

<li>McKinney, W. (2017) <i>Python for Data Analysis</i>. O'Reilly Media. <a href="https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662" target="_blank">https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662</a></li> 

<li>Wikipedia (2020) <i>Big Data: Critique of Big Data Execution</i>. <a href="https://en.wikipedia.org/wiki/Big_data#Critiques_of_big_data_execution" target="_blank">https://en.wikipedia.org/wiki/Big_data#Critiques_of_big_data_execution</a></li>

</ol>
