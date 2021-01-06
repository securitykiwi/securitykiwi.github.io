---
title: Types of Machine Learning
author: 
tags: [setup]
---

<style>p {text-align: justify;}</style>

There are four broad categories of machine learning:

* Supervised
* Unsupervised
* Semisupervised
* Reinforcement Learning
 
### Supervised Machine Learning

Supervised learning requires a _labelled_ training dataset where each data point has an associated label, essentially showing the algorithm the correct answer. A classic example of supervised learning is categorising email spam; an algorithm is shown what spam and non-spam emails look like. This type of problem is called a classification task, a task suited to supervised techniques. Human-readable labels will need to be transformed into numerical values representing the category of data (e.g. spam vs. not spam) for the algorithm to understand. 

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/labelled-data-supervised-example.jpg" alt="An image illustrating a labelled dataset, each data entry has a label." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">An example of a labeled dataset.</p>

The example in the image above shows a labelled network capture dataset, with numerical values representing features such as port number, and a label column representing the type of traffic. One aim of processing this data might be to predict attack traffic using the examples to train a system.

### Unsupervised Machine Learning

Unsupervised learning presents the algorithm with _unlabelled_ data. The algorithm learns by inference, there are no correct answers for it to learn from. Unsupervised techniques are useful for exploratory data analysis to find hidden patterns in data, typical tasks can include clustering, anomaly detection and visualization. Consider how these tasks cannot be labelled by a human operator - the point is to obtain the understanding of the data.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/unsupervised-example-graph.jpg" alt="An image illustrating an unlabelled dataset, data entries do not have labels." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">An example of an unlabelled dataset.</p>

The example in this image shows an unlabelled network capture dataset, numerical values represent network information as they did in the example above. However, now we do not have labels we don't know what type of traffic is in the dataset. This type of dataset could be used to cluster data points, which we can then perform additional analysis and consider what the results tell us. 

### Semisupervised

Semisupervised algorithms are made up of supervised and unsupervised techniques, allowing _partially labelled_ data to be analysed. Labelled data may be only partially labelled for a variety of reasons; cost, as human-driven labelling of every data point, is expensive or the data source may create results which are only partially labelled. Partially labelled data may also exist as datasets are merged.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/partially-labelled-data-example.jpg" alt="An image illustrating a partially labelled dataset, only some data entries have a label." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">An example of a partially labeled dataset.</p>

You may have made partially labelled data yourself - your photos. You may have labelled (tagged) some images from holidays, specific locations, but it is unlikely all of your photos are labelled.

### Reinforcement Machine Learning

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/reinforcement-learning-diagram.png" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">The cylce of action and reward in reinforcement systems.</p>

Reinforcement learning utilises a principle of _reward_ that is accumulated as the agent moves towards the desired outcome. Reinforcement learning does not necessarily require a correct output, a sub-optimal result will still progress the overall performance of the algorithm. Considering its unsupervised nature, it could be considered an unsupervised technique, however, it is so distinct from other techniques it is often given its own category. The image above shows the cycle of a reinforcement system. Where state refers to the current condition of the system (wall hit/success for example), reward, leading to a new action until success is achieved.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/reinforcement-learning-maze.png" alt="." style="width:350px;"/></div>

<p style="text-align: center; font-style: italic;">An illustration of a reinforcement system's attemtps to move through a maze.</p>

To more clearly understand the concept consider the maze above. The machine learning system's goal is to go from the start of the maze to the end, but it cannot see the maze. A rule is when it hits a wall it must start again. Through thousands of iterations, a system will learn a path through the maze. Shown in our maze with only 4 attempts; in the beginning, it just immediately hits the wall, so it makes a change. Attempt 2 is slightly better. Attempt 3 is slightly better than attempt 2, and so on.

##  Summary

The type of machine learning depends on the type of data you are using, and the task you wish to complete. One of the first steps we consider as we embark on a machine learning project is what algorithm to use. Part of this process involves thinking about what type of machine learning is required - what we have discussed here. You will see more of this as we learn and as we conduct our own projects in later sections.

The next page discusses the stages of a machine learning project.

---

## References

<ol>

<li>Muller, A. and Guido, S. (2016) <i>Introduction to Machine Learning with Python: A Guide for Data Scientists</i>. O'Reilly. <a href="https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/" target="_blank">https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/</a> </li>

<li>Pedrycz, W. and Chen, S. (2016) <i>Sentiment Analysis and Ontology Engineering: An Environment of Computational Intelligence</i>. Springer. <a href="https://www.springer.com/gp/book/9783319303178" target="_blank">https://www.springer.com/gp/book/9783319303178</a> </li> 

<li>Wikipedia (2020a) <i>Supervised learning</i>. <a href="https://en.wikipedia.org/wiki/Supervised_learning" target="_blank">https://en.wikipedia.org/wiki/Supervised_learning</a> </li> 

<li>Wikipedia (2020b) <i>Unsupervised learning</i>. <a href="https://en.wikipedia.org/wiki/Unsupervised_learning" target="_blank">https://en.wikipedia.org/wiki/Unsupervised_learning</a> </li> 
    
<li>Wikipedia (2020c) <i>Semi-supervised learning</i>. <a href="https://en.wikipedia.org/wiki/Semi-supervised_learning" target="_blank">https://en.wikipedia.org/wiki/Semi-supervised_learning</a> </li> 

<li>Wikipedia (2020d) <i>Reinforcement learning</i>. <a href="https://en.wikipedia.org/wiki/Reinforcement_learning" target="_blank">https://en.wikipedia.org/wiki/Reinforcement_learning</a> </li> 

</ol>
