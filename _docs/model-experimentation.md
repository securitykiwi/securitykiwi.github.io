---
title: Model Experimentation
author: 
---

<style>p {text-align: justify;}</style>

_Machine learning is an experimental task_, some models will work better with different datasets and the type and quality of a dataset also affect the algorithm choice. Trying a number of different algorithms and comparing their results is a necessary step to finding the best solution for your task. This page will discuss how to approach machine learning experimentation and how to choose algorithms. 

The sections that follow are closely linked to this process, model evaluation is the next stage where we evaluate the models we have chosen to experiment with. Following model evaluation is model tuning, where we tweak our model setting to gain the best performance.

## Consider the Data

Consider the data you are going to work with. Whether you have the data you are going to work with, or you are yet to find a suitable dataset, consider a set of requirements which are dictated by the data. Review the <a href="/datasets-and-data-collection/considering-data/" target="_blank">Considering Data</a> page for creating requirements and considering datasets.

While requirements can help you explore issues in more depth, you can quickly narrow down choices of algorithms by considering the type of data. Will the data be labelled? Then a supervised technique will be appropriate. Is the data unlabelled? An unsupervised technique would be best there.

In nearly all cases you will need to explore and cleanse the data, viewing what you have to work with and removing repeated datapoints, converting non-numerical values to numerical values, and so on. Return to the <a href="/datasets-and-data-collection/exploring-datasets" target="_blank">Exploring Datasets</a> and <a href="/datasets-and-data-collection/dataset-preperation" target="_blank">Dataset Preperation</a> pages to remind yourself of the processes and techniques we have previously discussed.

## Consider the Problem Type

Consider what you are trying to do. What does it boil down to? Does your problem boil down to a classification task? Are you trying to predict a numerical value over time? Considering this, combined with considering the data, allows you to narrow down supervised vs. unsupervised techniques and then type. By the end of this process, you will have found your combination. For example, "I am going to be working with a labelled dataset of network data over time, containing attack traces which need to be found.", "Ok, I need a supervised, classification technique". You're still in the early stages, but you have enough information to begin research to find out what others have done, draw inspiration and get started.

#### A Useful Tool

The <a href="https://samrose3.github.io/algorithm-explorer/" target="_blank">Algorithm Explorer</a>, created by Sam Rose, is a useful tool to help you narrow down choice as a beginner. It asks the question, "What are you trying to solve?" and presents a number of common algorithms used for that technique. The tool isn't exhaustive and simply provides basic information, however, it is useful for beginners.

## What Have Others Done?

Discovering what researchers have done in the same problem area you wish to work in is the best way to figure out what techniques you can use. Researchers will have conducted high-quality analysis on their techniques and considered various challenges and issues, as well as likely documenting any issues which arose during their project or which they were unable to fix.

Searching academic search engines, such as <a href="https://scholar.google.com/" target="_blank">Google Scholar</a>, <a href="https://ieeexplore.ieee.org" target="_blank">IEEE Xplore</a>, <a href="https://dl.acm.org/" target="_blank">ACM Digital Library</a>, <a href="https://www.sciencedirect.com/" target="_blank">Science Direct</a> and websites such as <a href="https://www.researchgate.net/" target="_blank">Research Gate</a> for open access research is a good start.

Many enthusiasts, academics, scientists and programmers write blogs on their projects which are incredibly useful, so simply searching Google is also a good idea.

The page <a href="/docs/how-to-conduct-research/" target="_blank">How to Conduct Research</a> should be your next reading assignment at this stage.

## Summary 

On this page, we discussed the stage of model experimentation, a process involving training multiple different models using different techniques to find the model which performs best. The process has three stages; consider the data which dictates the type of machine learning (supervised vs. unsupervised), consider the problem types which dictates the type of algorithm (classification, clustering, etc) and using this to search academic databases, and Google, to find what others have done with the same or similar problems.

This page has focused on the theoretical side of experimentation, the next page <a href="/docs/model-evaluation/" target="_blank">Model Evaluation</a> discusses a number of techniques to measure the performance of a machine learning system.

---

## References

<ol>

<li>McKinney, W. (2017) <i>Python for Data Analysis</i>. O'Reilly Media. <a href="https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662" target="_blank">https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662</a></li>

<li>Rose, S. (2020) <i>Algorithm Explorer</i>. <a href="https://samrose3.github.io/algorithm-explorer/" target="_blank">https://samrose3.github.io/algorithm-explorer/</a></li>

</ol>
