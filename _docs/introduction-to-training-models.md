---
title: Introduction to Training Models
author:
---

<style>p {text-align: justify;}</style>

This section will teach you how different machine learning algorithms are trained, various challenges and solutions, and how we as the programmer can manipulate algorithms to best fit our data. We approach this through a series of tutorials using different machine learning techniques. This will expose us to training several times without being repetitive and hopefully make the training process clear and illuminate the inner workings of different machine learning systems as we learn the theory. 

We start with Linear Regression and move on to Neural Networks and Deep Learning for several pages, and finish with an overview of challenges and solutions which are applicable to a number of machine learning techniques. Much of this chapter is focused on neural networks as they are a popular and interesting technique within machine learning, we will give ample time to other techniques however. This page introduces you to important aspects of how machine learning models train and learn.

Although this training section has discusses a number of algorithms it is not intended as a reference, for a more complete list of algorithms see the <a href="/docs/introduction-to-algorithms-and-techniques/" target="_blank">Algorithms & Techniques</a> section. 

The next section <a href="/docs/model-experimentation/" target="_blank">Model Evaluation and Tuning</a> is linked to training. Training is only the beginning of a three-part core of any machine learning implementaiton: train, evaluate and tune.

## What is Training?

Training, if you recall, is the defining factor of a machine learning algorithm allowing a computer program to learn from data without being explicitly programmed. Remember this example from chapter one:

#### Explicit program

```java
If email contains "pills";
If email contains "p!lls"; 
If email contains "pillz";
...
Then mark spam;
```

#### Learning program

```java
Try to classify emails;
Change self to reduce errors;
Repeat;
```

This example illuminated a machine learning programs brevity; it is clear a programmer no longer has to think of every single occurrence of a spam word attempting to evade a filter, however, it did not illuminate <i>how</i> a machine learning system learns. This page will describe elements of training and learning, and further pages in this section will dive into practical examples where we train a linear regression model and a simple neural network.

Training involves the careful consideration of data, as we have discussed within the <a href="/docs/considering-data" target="_blank">Considering Data</a> section, once the data has been prepared the dataset needs to be split into training and testing datasets. Machine learning algorithms are trained on a training set and then exposed to a test set which it has never seen before. Through this, we can get an idea of the accuracy of our systems before we expose them to live data and make appropriate changes. We will discuss this in the next section <a href="/docs/model-experimentation" target="_blank">Model Experimentation</a> after we further discuss training and try our hand at training a Linear Regression prediction algorithm, a single-layer and a multi-layer neural network.

Much like with the process of preparing data we saw in <a href="/docs/considering-data" target="_blank">Considering Data</a>, scikit-learn provides a function which allows us to split a dataset easily. We will see this in action on the next page, where we train a Linear Regression algorithm.

## Types of Systems

As we discussed in the introduction, there are four types of machine learning systems commonly used to categorise different systems which we will briefly look at again and expand on.

* **Unsupervised** a system which learns by inference from unlabelled data.
* **Supervised** a system which learns what the 'correct answer' is from labelled data.
* **Semi-supervised** a system which learns from partially labelled data.
* **Reinforcement** a system which learns by maximising a 'reward' given from a function.

We will focus on supervised and unsupervised, as the other techniques are technically part of those categories.

### Supervised Learning

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/labelled-data-supervised-example.jpg" alt="." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">A visual example of a labelled dataset.</p>

Supervised algorithms take labelled data and the system learns from the examples and applies this learning onto new data. The labels act as a teacher to show the algorithm correct answer. Supervised techniques allow you to conduct regression, predicting a continuous value from input data (prediction) and classification of input values. For example, predicting future attack traffic volume. Labels in our example image above show either background traffic or attack traffic. These human-readable values would be transformed into numerical values representing these two categories to be processed by an algorithm.

Common examples of supervised algorithms include:

* Linear regression
* Decision tree
* Neural Networks (inc. deep learning)
* Support vector machine
* Naive bayes

### Unsupervised Learning

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/unsupervised-example-graph.jpg" alt="." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">A visual example of an unlabelled dataset.</p>

Unsupervised algorithms take unlabelled data and the system attempt to understand patterns and discover underlying meaning or trends. Think of the labels as a teacher, here there is no teacher. This type of system is often used when the patterns and trends are unknown to the user.

A popular use of unsupervised techniques is clustering data. For example, providing an unsupervised technique a dataset of unlabelled data of land mammals including the number, size, number of legs, weight and various other data points, the algorithm would be able to cluster the data into surprisingly accurate segments which represent individual mammal species.

Common examples of unsupervised algorithms include:

* Association rule
* k-means
* k-NN (K Nearest Neighbor)
* Hidden Markov model
* Hierarchical clustering
* Principal component analysis

## Types of Learning

Machine learning algorithms can learn in two ways; online and batch.

**Online learning** is where the system learns sequentially as new data is processed (individual data points or batches of data points) in an incremental manner. As new data is created it can be passed to the system and it will be processed and results created. This is the most convenient method, allowing the system to adapt to new data. The fact online learning reacts to new data easily is also a weakness, as it is sensitive to bad data. This can lead to the performance of the system degrading quickly should a sensor fail or someone purposefully submit bad data to the system.

**Batch learning** conducts the entire training process in a batch, not in an incremental manner. The actual internal process may learn incrementally. Batch learning is less convenient, as the system has to be retrained to learn from new data. This process can take significant amounts of time considering the size of data sets, and cost more in terms of computing and financial resources.

## How Systems Generalize

Humans are said to have general intelligence, we can apply our learning to a general set of activities rather than a specific or individual action. We discussed this in the introduction, were the ultimate goal of artificial intelligence is too great a general intelligence rather than a system which only works on a single problem - say playing the game of Go. In the context of machine learning <i>generalization</i> is an important concept which refers to the ability of the system to apply to new data, not just the data on which it was trained.

### Model-based Learning

Model-based learning creates a model from a selection of training data - the system creates a representation of what it has learned from training data and applies that learning to new data and attempts to fit that data within its model. The diagram below shows an example system designed to classify data, the greyed area shows where the system has decided a class boundary lies. Everything within this boundary is part of one class, every data point outside is part of a different class. A new instance (in orange) has been placed within the class represented by the grey area. So to the algorithm, this data point belongs to the class represented by the grey boundary.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/model-based-learning-example.jpg" alt="." style="width:470px;"/></div>

<p style="text-align: center; font-style: italic;">A new instance sits within the model boundry.</p>

It is worth pointing out here, the word "model" not only refers to a representation of data (as in model-based learning), but also to a complete trained system, which is the typical use of the word "model" within this course (and elsewhere). E.g. "machine learning model".

### Instance-based Learning

Instance-based learning can be thought of as a similarity-based learning system, as these techniques measure the similarity between trained data and data to which they are exposed on and instance-by-instance basis. The diagram below shows an example of this. In this example, the instances next to the highlighted instance are considered similar. 

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/instance-based-learning-example.jpg" alt="." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">The simmilarity of a new instance is tested against existing data.</p>


## Decision Boundaries

Machine learning algorithms alter internal parameters as or after (online vs. batch) they are exposed to data. The decision to make a prediction, classification, cluster or other action on an input is handled by the algorithm using various techniques borrowed from the field of mathematics (the complexity is hidden from the user of libraries such as Scikit-learn and TensorFlow). Following from our examples in model-based and instance-based learning, it can be useful to observe how machine learning systems see data. We can do this by visualising the decision boundaries in applicable tasks (for example, classification and prediction).

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/sklearn-svm-make-classification.jpg" alt="." style="width:350px;"/></div>

<p style="text-align: center; font-style: italic;">Decision boundries visualised on a two category dataset.</p>

The image above shows the decision boundaries between feature clusters in two types of support vector machine which were trained on the scikit-learn sample dataset `make_classification`. The left image used a linear SVM, which attempts to fit features into a series of decision brackets, visualised here as vertical columns. The colours represent the confidence of the algorithm. The right image shows a radial bias function (RBF) SVM, which utilises the RBF which is a higher accuracy method for measuring similarity between features. This is evident in the accuracy of each, the linear SVM is 93% accurate, while the RBF SVM is 95%, this gap opens up significantly depending on the data.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/sklearn-svm-make-circular.jpg" alt="." style="width:350px;"/></div>

<p style="text-align: center; font-style: italic;">Decision boundries visualised on a 'circular' dataset.</p>

This difference in accuracy based on the technique used is clearly shown here. The same two SVM algorithms, trained on scikit-learns `make_circles` dataset. This dataset is designed to mimic 'circular' datasets, the previous `make_classificaiton` was more generically intended to create illustrative groupings for classification examples. The left image, linear SVM, achieves 40% accuracy as it fails to fit the features within its ridged columns, while the right image, RBF SVM is able to better conform to the shape of the data by measuring similarity between features more fluidly, reaching 88% accuracy. More examples are available in the <a href="https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html" target="_blank">scikit-learn documentation on classifier comparison</a>, along with the code to create decision boundary images like those above.


## Pre-training

Some machine learning algorithms can benefit from pre-training. Pre-training is the process of passing a dataset through one type of algorithm before passing it through another type, specifically for the purpose of creating 'better' data for the second algorithm to process. We discussed this process briefly in the introduction section where we looked at examples from academia. In the example Ding et al pre-trained data to increase the efficiency of their secondary algorithm, a Deep Belief Network (DBN). The process reduced the noise by removing outliers and essentially increased the resolution of the data passed to the DBN.

Pre-training is distinct from pre-processing, although it has a similar goal. Pre-processing is the process of cleansing and preparing data before it is processed by an algorithm with the goal of producing better results or forming data in a particular way for a specific algorithm. Pre-processing includes feature scaling, standardisation etc.


## Dimensionality Reduction

You will come across dimensionality in two primary aspects in machine learning; high-dimensional space and dimensionality reduction.

High-dimensional space is a way of representing data in a space spanned by the data's attributes and an individual data points position related to attribute values. Think of this as a 2D graph, with x and y positions, a single data point placed in space (on the graph) according to its x and y attributes. Add a third dimension and see how you can now derive more information from the data, prior you were confined to 2D space, now you can move in 3D. High-dimensional space is 4D and above. Just as a graph with one axis holds less information than a graph with two axis, high-dimensions allow information to be represented by more attributes. This comes at a cost, it is computationally expensive to compute this type of data and its hard for humans to understand to create the models in the first place.

Dimensionality reduction is the reduction of the number of dimensions to increase the efficiency of training; reducing the training time and often increasing the accuracy of the technique for example. We will explore specific methods of dimensionality reduction in a later section dedicated to it. For the moment, knowing dimensionality reduction exists is sufficient.

## Summary

We have discussed a number of new concepts on this page. We discussed supervised and unsupervised learning and specific techniques in each. We learned another way to categorise algorithms, by the way they learn; batch or online learning.

* **Batch learning** trains data in one big batch and must be retrained to include new data.
* **Online learning** trains incrementally as new data is added after being trained.

We moved on to learn about how systems generalize - how they apply what they have learned to new data. 

* **Model-based** learning creates a representation (model) of what it has learned and fitted new data into those models.
* **Instance-based** learning focuses on the similarity between training data and new data.

We visualised decision boundaries to understand how algorithms decide where a data point lies, giving some insight into how algorithms actually work. We briefly discussed pre-training and dimensionality reduction to expose the reader to these concepts which will be part of pages and processes ahead and form important stages which may come up in other reading.

Coming up next in this Training Models section is the practical side, where we learn about a few specific techniques and train models to understand how machine learning is conducted in practice.

---

<p style="text-align: center;">Feedback is welcome!</p>

<p style="text-align: center;">Get in touch <a href="/contact/" target="_blank">securitykiwi [ at ] protonmail.com</a>.</p>

---

## References

<ol>

<li>Bostrom, N. (2014) <i>Superintelligence: Paths, Dangers, Strategies</i>. Oxford University Press, Oxford. <a href="https://en.wikipedia.org/wiki/Superintelligence:_Paths,_Dangers,_Strategies" target="_blank">https://en.wikipedia.org/wiki/Superintelligence:_Paths,_Dangers,_Strategies</a> </li>

<li>Ding, Y., Chen, S., and Xu, J. (2016) Application of Deep Belief Networks for opcode based malware detection. IEEE. <a href="https://ieeexplore.ieee.org/document/7727705">https://ieeexplore.ieee.org/document/7727705</a></li>

<li>Grobler, J. (2020) <i>Classifier comparison</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html" target="_blank">https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html</a></li>

<li>Scikit-Learn (2020a) <i>sklearn.datasets.make_classification</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html</a></li>

<li>Scikit-Learn (2020b) <i>sklearn.datasets.make_circles</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html</a></li>

<li>Skillicorn, D. (2012) <i>Understanding High-Dimensional Spaces</i>. Springer. <a href="https://www.springer.com/gp/book/9783642333972" target="_blank">https://www.springer.com/gp/book/9783642333972</a></li>

</ol>
