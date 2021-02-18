---
title: Challenges in Training Models
author:
---

<style>p {text-align: justify;}</style>

Challenges in training can be encountered which result in the model's accuracy being lower than expected. These can be caused by the data, features or parameters. These challenges may be fixed during the stage which comes after training - tuning. We discuss tuning in detail in the next section <a href="/docs/model-tuning/" target="_blank">Evaluation & Tuning</a>.

However, other issues may be more difficult to fix and troubleshoot. They may require the dataset to be further processed or a completely different method may be required. The number of scenarios is large, so this page will discuss the main issues which you may face training your first machine learning system.

## The Data

In the section <a href="/docs/challenges-of-datasets/" target="_blank">Datasets and Data Collection</a> we discussed the challenges with datasets. Much like troubleshooting any system, we must start from the lowest level. Here, that is the data. We must ask the same questions we did before we created our model.

**Is there enough data?** Machine learning algorithms require a non-trivial amount of data in order to learn the relationships within the dataset. This amount varies on the task and the algorithm used, however, it's in the order of thousands as a minimum.

**Is the data representative?** Reconsider if the data actually holds the insights you need to make the predictions or classifications you are attempting to make.

**Is the quality of the data high enough?** Has the dataset been processed to remove noise, outliers and other values which may drastically affect the way the algorithm understands the data? We discussed the techniques for these transformations in the page <a href="/docs/dataset-preparation/" target="_blank">Dataset Preparation</a>.

**Is the data appropraite for the algorithm choice?** Have you scaled the features to ensure the algorithm can process them appropraitely? We discusses feature scaling in the <a href="/docs/dataset-preparation/" target="_blank">Dataset Preperation</a> page.

## The Method

Once we have considered the data, we consider the method. Is the algorithm appropriate for the data you want to work on? Simple questions, such as have you picked a supervised method for labelled data or an unsupervised method for unlabelled data help us answer this.

The number of applications for machine learning is immense, looking at others work; whether research, commercial or even a tutorial can help us understand if our method is appropriate. Reading around the subject, your application and the model you wish to use will give you the knowledge to use it appropriately. This course aims to give you a great deal of information on a number of models and techniques as they are added over time, see the <a href="/docs/introduction-to-algorithms-and-techniques/" target="_blank">Algorithms and Techniques</a> page.

## The Model

Once we have ruled out issues with the data and the method, we move on to the model created by the algorithm. The model is closely related to the method we chose and we have the opportunity to change and fix it, much like we can process and clean data further.

Here we will discuss issues we may have with our model, in the next section <a href="/docs/model-tuning/" target="_blank">Evaluation and Tuning</a> we will discuss how we use various parameters to improve these issues and ultimately the accuracy of our model.

### Underfitting and Overfitting

Models which <i>underfit</i> have produced output which is too generalized, the algorithm has failed to find the trend within the data and cannot understand new data it is exposed to. Figure 1 shows an example of a line of best fit which has underfit the data.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/underfit-fit-graph.jpg" alt="." style="width:300px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 1: An example of an underfit model best fit line.</p>

Models which <i>overfit</i> have produced output which has been skewed by outliers in the training data or become too representative of the training set, this means the algorithm will not work well when exposed to new data. Figure 2 shows an example of a best-fit line which has overfit a dataset.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/overfit-graph.jpg" alt="." style="width:300px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 2: An example of an overfit model best fit line.</p>

Models which have a <i>good fit</i> have learned the trend of the data and will generalize well to new data. Figure 3 shows an example of a best-fit line which has fit well to the training set.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/well-fit-graph.jpg" alt="." style="width:300px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 3: An example of a well balanced best fit line.</p>

**Underfitting** requires you change the data or the algorithm. There simply isn't the information for the algorithm chosen to derive useful insight, or the algorithm is not suitable for the task. Changing the data doesn't mean we need to add more data, assuming the data passes the checks we discussed at the beginning of this page, we can add more features to derive insight from.

**Overfitting** can be controlled by reducing the complexity of a model. This can be done by removing irrelevant features, stopping training before it begins to overfit, reducing the size of a neural network (pruning), or regularization. Regularization reduces the size of model parameters such as weight (in neural networks) thus reducing the effect of large variances in input data, resulting in more stable learning.

Overfitting and underfitting are challenges which you will likely come across when training algorithms, we will discuss these further, along with practical examples in the next section.

## Summary

On this page, we reiterated the importance of ensuring the data is representative, high quality (low noise, few outliers) and there is enough data (quantity). We considered the importance of trying different algorithms and knowing when its the right time to move to a different technique given the data is high quality.

We discussed the challenges of underfitting and overfitting, where our algorithm is unable to find a trend within the data or creates a trend which is too specific to the training set, in both cases resulting in an algorithm that does not generalize well to new data beyond the training set.

In the next section, we discuss the evaluation of models and how we tune algorithms to get the best performance out of them.

---

## References

<ol>
    
<li>Cai, L., and Zhu, Y. (2015) <i>The Challenges of Data Quality and Data Quality Assessment in the Big Data Era</i>. Data science Journal. <a href="https://datascience.codata.org/articles/10.5334/dsj-2015-002" target="_blank">https://datascience.codata.org/articles/10.5334/dsj-2015-002</a></li>

<li>McKinney, W. (2017) <i>Python for Data Analysis</i>. O'Reilly Media. <a href="https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662" target="_blank">https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662</a></li>    
<li>Wikipedia (2020a) <i>Big Data: Critique of Big Data Execution</i>. <a href="https://en.wikipedia.org/wiki/Big_data#Critiques_of_big_data_execution" target="_blank">https://en.wikipedia.org/wiki/Big_data#Critiques_of_big_data_execution</a></li>

<li>Wikipedia (2020b) <i>Overfitting</i>. <a href="https://en.wikipedia.org/wiki/Overfitting" target="_blank">https://en.wikipedia.org/wiki/Overfitting</a></li>


</ol>
