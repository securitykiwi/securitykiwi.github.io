---
title: Optimization Algorithms
author: 
---

<style>p {text-align: justify;}</style>

This page aims to give you an overview of optimization algorithms and help you understand their place in machine learning systems. This page will help you understand how machine learning algorithms find optimal values and the mechanism of learning. 

Full understanding of every detail is not required to use machine learning libraries. However, gaining an understanding of fundamentals will help you implement your own projects and go beyond following tutorials. If you do not fully understand what is discussed here by the end of the page, don't despair. Using these techniques in tutorials, projects and further reading will help illuminate the concepts.

## What is an Optimization Algorithm?

Machine learning techniques use optimization algorithms to increase the accuracy of their output. These optimization functions are borrowed from the field of mathematics and solve the optimization problem presented to machine learning algorithms - to improve the quality of the prediction or classification as the algorithm is trained.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/gradient-descent-graph.jpg" alt="gradient descent algorithm finding the minimum loss." style="width:300px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 1: gradient descent algorithm searching for the minimum loss (thus the highest accuracy).</p>

## Terminology

Before we go any further, we must discuss a number of terms to shape our future discussion.

### Parameters

Before we discuss specific optimization algorithms we should discuss parameters. There are two different types of parameters; model parameters and hyper-parameters.

**Model parameters** are internal model variables usually not set by the programmer used and updated by the model during training. Model parameters are saved at the end of training and represent the skill of the model - the parameters have been well optimised so the model is "good", or "bad" as the case may be. An example of a model parameter is a weight, which in neural networks is updated to alter when information is passed onto the next node, altering how the network performs.

**Hyper-parameters** are external variables to the model and may be part of the manual tuning process of the model by the programmer. For example, the number of iterations an algorithm makes over a dataset can be tuned to increase performance (time to completion). 

Optimization algorithms are internal to a machine learning algorithm, the choice of optimization algorithm by the programmer can alter how quickly a method trains. For example, a programmer may choose one algorithm over another due to its known efficiency and success with the technique, based on the documentation of the particular algorithm implementation within a software library (e.g. TensorFlow).


### Convergence

Convergence is a term describing the outcome of the process created by a loss function, as the function gets closer to its optimal value and the loss decreases (and accuracy increases). An algorithm is said to converge when the loss function has reached a trend which can decrease no longer, or theoretically reaches 0, although in practice reaching a 0 value is unlikely. Another way to describe convergence is the algorithm has reached its optimal state - using the current settings and method, the algorithm can not become any better.

Lets look at an example in code to explore further.

On the Training Neural Networks page we experimented with a multi-layered neural network designed to compute XOR (exclusive OR) arguments. Let's try it again, but with the parameters changed to purposefully miss convergence.

**Setup Code**

```python
# Import Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

```python
# Plot algorithm decision boundary.
# Code based on http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
def plot_decision_boundary(classifier, X, y, title):
    xmin, xmax = np.min(X[:, 0]) - 0.05, np.max(X[:, 0]) + 0.05
    ymin, ymax = np.min(X[:, 1]) - 0.05, np.max(X[:, 1]) + 0.05
    step = 0.01
    
    cm = plt.cm.coolwarm_r
    #cm = plt.cm.RdBu
    
    thr = 0.0
    xx, yy = np.meshgrid(np.arange(xmin - thr, xmax + thr, step), np.arange(ymin - thr, ymax + thr, step))
    
    if hasattr(classifier, 'decision_function'):
        Z = classifier.decision_function(np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis])))
    
    else:
        Z = classifier.predict_proba(np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis])))[:, 1]
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']), alpha=0.6)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xticks((0.0, 1.0))
    plt.yticks((0.0, 1.0))
    plt.title(title)
```

**Compute XOR**

```python
# Compute XOR (purposefully non-convergent)
# Setting the input samples.
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]],
            dtype=np.double)
y_XOR = np.array([0, 1, 1, 0])
# Create a MLPClassifier.
mlp = MLPClassifier(hidden_layer_sizes=(5,),
    activation='tanh',
    max_iter=1000,
    random_state=10)
# Train model
mlp.fit(X, y_XOR)
# Plot and display the decision boundary
plot_decision_boundary(mlp, X, y_XOR, 'XOR')
plt.show()
# Get predicted values and print
pred = mlp.predict_proba(X)
print("MLP's XOR probabilities:\n[class0, class1]\n{}".format(pred))
```

Above, we have changed the max iterations parameter, `max_iter`, from 10,000 to 1,000. When we run this code, it results in a Scikit-Learn warning: "ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.", with the XOR visualization incomplete. The left graph is the non-converged XOR prediction and the right is the complete, convergent, XOR prediciton from the previous page. Said another way, the left graph did not find an answer and the right graph did find an answer.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/non-convergent-xor-neural-network.jpg" alt="Comparison of two graphs, left graph shows incomplete prediction due to non-convergence. Right graph shows complete prediction from previous page." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">Comparison of non-convergent and convergent predictions of XOR.</p>

### Learning Rate

The learning rate is a parameter which affects the size of the step an algorithm takes towards the optimal loss within gradient descent. It is often used as a tuning parameter, along with other variables, which are used in various configurations to control the internal state of the algorhtm to obtain the best results.

Let's explore the XOR computation again, this time altering the learning rate to see what happens.

```python
# Compute XOR
# Setting the input samples.
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]],
            dtype=np.double)
y_XOR = np.array([0, 1, 1, 0])
# Create a MLPClassifier.
mlp = MLPClassifier(hidden_layer_sizes=(5,),
    activation='tanh',
    learning_rate_init=0.1,
    max_iter=10000,
    random_state=10)
# Train model
mlp.fit(X, y_XOR)
# Plot and display the decision boundary
plot_decision_boundary(mlp, X, y_XOR, 'XOR')
plt.show()
# Get predicted values and print
pred = mlp.predict_proba(X)
print("MLP's XOR probabilities:\n[class0, class1]\n{}".format(pred))
```

Max iterations are set back to 10,000. In the code above we have added the `learning_rate_init` parameter to the `MLPClassifier`. This controls the initial learning rate of the algorithm, by default when a user does not specify, the learning rate is set to 0.001. We ran the code three times with the `learning_rate_init` set to defaut, 0.1 and 0.5. The image below shows the `MLPClassifiers` predictions from the three runs. The left-most graph shows a good XOR prediction, however, as we have increased the `learning_rate_init` parameter the predictions have suffered and are no longer representative of an XOR computation. 

<div style="text-align:center;"><img src="/assets/images/training-algorithms/xor-comparison-learning-rate-change.jpg" alt="Comparison of three XOR prediction graphs. Left-most shows a XOR prediction with a learning rate of 0.001, the middle grpah shows the difference at 0.1 learning rate and the right-most grpah shows the different with the learning rate at 0.5. " style="width:700px;"/></div>

<p style="text-align: center; font-style: italic;">Learning rates: left-most: 0.001, middle: 0.1, right-most: 0.5.</p>

As we increase the learning rate, we increase the distance between the 'steps' an algorithm takes towards its goal. The algorithm takes larger steps and has less chance of finding the optimal path as it may essentially step over it. This is seen as the output becomes much worse as we increase the learning rate. 

This experiment is for illustrative purposes, changing the learning rate by large increments was bound to produce bad results. The experiment has exposed us to learning rate in action and what happens as we change it.

Feel free to explore with convergence and learning rate, you can also seek out new parameters within the scikit-learn <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html" target="_blank">documentation</a> on the `MLPClassifier`.

# Optimization Algorithms

## Gradient Descent

<i>Gradient descent</i> is a process used to find the lowest error rate by updating model parameters and following the direction of the slope until it reaches a valley; the valley representing a low error rate and thus high accuracy. Slopes? Valleys? Complex math is often made more digestible by visualizing the problem. This type of thinking can be strange to begin with, stay with me as I explain.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/gradient-descent-graph.jpg" alt="gradient descent algorithm finding the minimum loss." style="width:300px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 1: gradient descent algorithm graphed.</p>

Figure 1 shows gradient descent, the algorithm follows the slope of the curve until it finds the low point. Each point on the graph is after a change to internal model parameters, slowly the algorithm learns to increase accuracy (and reduce error/loss). However, you can plot in more dimensions, and this can dramatically shift your understanding of how machine learning actually works. Figure 2 shows a 3D graph of <i>stochastic gradient descent</i> (SGD) searching for the optimum loss (we will explain different varieties of gradient descent below).

<div style="text-align:center;"><img src="/assets/images/training-algorithms/huang-et-al-gradient-descent.jpg" alt="3D representation of gradient descent finding the minimum loss." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 2: SGD algorithm finding the minimum loss, 3D. Huang, et al.</p>

Figure 2 is a representation of <i>stochastic gradient descent</i> from <a href="#huang-et-al" target="_blank">Huang et al</a>. Figure 2 shows a 3D representation of the problem space the algorithm has to optimize. The blue dot is the representation of parameters and their change through time (the black line) as the algorithm is exposed to new data and the loss is optimized (reduced as much as possible). Projected below the 3D representation in figure 2 is another type of representation you may come across; <i>topographic</i>, the same as you may be familiar with from a geographic map showing the height of the terrain. Figure 2 shows the same concept as Figure 1, one is 2D and the other 3D.

An analogy may illuminate how gradient descent works. Gradient descent can be seen as a blind-folded mountaineer attempting to find their way down by only feeling the gradient of the terrain. Gradient descent may find different paths each time it is run. As gradient descent 'feels' its way to a low point it may get stuck and consider a location the minimum and consider the data-optimized, when in fact there is a lower point which it hasn't found. This is often referred to as "being trapped in local minima".

In practice, there are three types of gradient descent which we will discuss here. The first, batch gradient descent calculates the gradients for an entire dataset and performs one update. As such, it is slow and does not work with online learning, as we discussed in the <a href="/training-models/introduction-to-training/" target="_blank">Introduciton to Training</a> page, where our model can handle new data on-the-fly. Batch gradient descent is only suitable for working on entire datasets. This also has the weakness of not working with large datasets, which do not fit into memory, as it works on the entire set.

## Stochastic Gradient Descent

<i>Stochastic Gradient Descent</i> (SGD) is a type of gradient descent which solves the weaknesses of whole-dataset learning and slow speed. Stochastic gradient descent performs calculations and updates parameters for each training example, allowing it to process data streams rather than the whole dataset and process them faster. “<i>Stochastic</i>” means random, in this case, it refers to the addition of a small random amount to the variance between data points (the distance between them). This results in an algorithm which is more resilient to getting trapped in local minima, allowing it to break out of valleys to find the global minimum (the true minimum value). This creates 'random jumping' and can be best describes visualized:

<div style="text-align:center;"><img src="/assets/images/training-algorithms/sgd-graph-comparison.jpg" alt="Stochastic gradient descent versus gradient descent. Stochastic gradient descent randomly jumps as the error rate decreases, while the gradient descent graph is a much smoother descending gradient." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 3: Gradient descent vs. stochastic gradient descent.</p>

Figure 3 shows how the addition of a small random amount to the variance of each data point processed, creates jumps allowing the algorithm to escape local minima, but maintain its downward slope-finding to converge at a typically lower loss rate than batch gradient descent (above).


## Mini-batch Gradient Descent

<i>Mini-batch gradient descent</i> merges the behaviour of batch gradient descent and stochastic gradient descent to obtain the best of both. The technique iterates over small batches of a dataset applying the small random variance to the batch, resulting in a more stable convergence (more stable 'jumps'). Batch sizes become a parameter the programmer can tune to affect performance, varying between applications. The computation of batches, rather than individual dataset examples allows machine learning libraries, such as TensorFlow, to apply further optimizations to the process to increase performance.

Many applications of stochastic gradient descent refer to this type, mini-batch gradient descent. It is not uncommon to find applications of SGD which operate on mini-batches without specifically calling themselves min-batch SGD.

### Momentum

The addition of mini-batches to gradient descent allows the best of the two previous algorithms, adding <i>momentum</i> can further increase the efficiency of an optimization algorithm. Momentum allows an algorithm to maintain forward movement and reach a more optimal loss rate. Techniques such as stochastic gradient descent can suffer from hesitation in navigating ravines. 

If we use our blind-folded mountain climber analogy, our algorithm is feeling its way down a slope and comes to a ravine - a feature smaller and narrower than a valley - so the loss rate decreases and movement stops. Momentum allows our mountaineer more forward movement, providing the possibility of finding a better way down the mountain - a better loss rate. In our algorithm, this results in faster convergence and reduced unnecessary sideways movement (oscillation) as it 'feels' each side of the ravine. In practice, momentum is achieved by increasing weight updates which decrease the gradient and reduced weight updates which change direction.

## More Optimization Algorithms

There are many more optimization algorithms suited to different types of task and neural network architecture. We won't go through them all, as that would create a dense page which would serve little value in practically increasing your knowledge - it will be best to explain those as we use them in later projects or when discussing specific deep learning architectures. This page has served to introduce you to optimization concepts and foundational optimizers on which others are based.

## Optimization Algorithms in Libraries

You have already used optimization algorithms if you have been following along with the code exercises in previous chapters. In scikit-learn the `perceptron()` function shares its underlying implementation with `SDGClassifier()`. As you may guess from its name, SDGClassifier stands for stochastic gradient descent classifier. It uses stochastic gradient descent as its optimization function. Similarly, later when we use TensorFlow, you simply pass optimization function names via parameters (e.g. classifier("optimizer name")). If you have looked into TensorFlow deep learning before you may have noticed "Adam" mentioned or used as a parameter, it is an optimizer named after adaptive moment estimation. 

## Summary

This page has introduced foundational optimization algorithms which are used to reduce a loss function, increasing the accuracy of a machine learning systems output as it iterates over a dataset. We have discussed a number of terms which help us describe important aspects; model parameters, internal parameters usually not set by the programmer and hyperparameters, external parameters which are used by the programmer to control learning. Convergence is used to describe how an algorithm finds an optimal value and cannot get any better with the current settings. Among other important terms.

We discussed the most widely used optimization algorithm, gradient descent, and its variations and derivatives. The main take away from the method is how it attempts to find its way down a slope to find the optimal low loss rate (thus a high accuracy) and how additions such as a small random addition to the variance between data points as it iterates over the dataset can significantly change its behaviour - allowing it to 'jump' out of local minima to typically find a more optimal solution (the global minima - the true low loss).

Next, we discuss the challenges of training machine learning systems.

---

<p style="text-align: center;">Feedback is welcome!</p>

<p style="text-align: center;">Get in touch <a href="mailto:securitykiwi@protonmail.com">securitykiwi [ at ] protonmail.com</a>.</p>

---

## References

<ol>

<li>Goldstein, T. (2018) <i>GitHub Repo: tomgoldstein / loss-landscape</i>. <a href="https://github.com/tomgoldstein/loss-landscape" target="_blank">https://github.com/tomgoldstein/loss-landscape</a></li>

<li id="huang-et-al">Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J., and Weinberger, K. (2017) <i>Snapshot Ensembles: Train 1, get M for free</i>. arXiv.org. <a href="https://arxiv.org/abs/1704.00109" target="_blank">https://arxiv.org/abs/1704.00109</a></li>

<li>Li, H., Xu, Z., Taylor, G., Studer, C., and Goldstein, T. (2018) <i>Visualizing the Loss Landscape of Neural Nets</i>. arXiv.org. <a href="https://arxiv.org/abs/1712.09913" target="_blank">https://arxiv.org/abs/1712.09913</a></li>

<li>Li, F., Karpathy, A., and Johnson, J. (2016) <i>CS231n, Lecture 3: Loss functions and Optimization</i>. [PDF]. <a href="http://cs231n.stanford.edu/slides/2016/winter1516_lecture3.pdf" target="_blank">http://cs231n.stanford.edu/slides/2016/winter1516_lecture3.pdf</a></li>

<li>Ruder, S. (2017) <i>An overview of gradient descent optimization algorithms</i>. arXiv.org. <a href="https://arxiv.org/abs/1609.04747" target="_blank">https://arxiv.org/abs/1609.04747</a></li>

<li>Raschka, S. (2020) <i>Machine Learning FAQ: SGD Methods</i>. <a href="https://sebastianraschka.com/faq/docs/sgd-methods.html" target="_blank">https://sebastianraschka.com/faq/docs/sgd-methods.html</a></li>

<li>Stanford (2020) <i>CS231n Convolutional Neural Networks for Visual Recognition: Introduction</i>. <a href="https://cs231n.github.io/optimization-1/" target="_blank">https://cs231n.github.io/optimization-1/</a></li>

<li>Sun, S., Cao, Z., Zhu, H., and Zhao, J. (2019) <i>A Survey of Optimization Methods from a Machine Learning Perspective</i>. arXiv.org. <a href="https://arxiv.org/abs/1906.06821" target="_blank">https://arxiv.org/abs/1906.06821</a></li>

<li>Wikipedia. (2020a) <i>Gradient Descent</i>. <a href="https://en.wikipedia.org/wiki/Gradient_descent" target="_blank">https://en.wikipedia.org/wiki/Gradient_descent</a></li>

<li>Wikipedia. (2020b) <i>Maxima and Minima</i>. <a href="https://en.wikipedia.org/wiki/Maxima_and_minima" target="_blank">https://en.wikipedia.org/wiki/Maxima_and_minima</a></li>

<li>Wikipedia. (2020c) <i>Stochastic Gradient Descent</i>. <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent" target="_blank">https://en.wikipedia.org/wiki/Stochastic_gradient_descent</a></li>

</ol>
