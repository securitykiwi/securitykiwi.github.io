---
title: Training Neural Networks
author:
---

<style>p {text-align: justify;}</style>

On this page we will build and train two neural networks; a single-layer network to categorise simple data and a multi-layer network to compute XOR logic. Along the way we will learn how data passes through a neural network and how activation functions work to gain a deeper understanding of training neural networks. We start with a short explanation of what a neural network is and its architecture, and get practical by using scitkit-learns to implement a simple neural network - a <i>Perceptron</i>.

The section below will be our first exposure to complex-looking math. We're using it to describe values and the processes involved, intertwined with English. If you're not comfortable with math, please do read it, _don't skim or skip over it_.

#### Contents

* [Single Layer Neural Network](#single-layer-neural-network)
    * [What is a Neural Network?](#what-is-a-neural-network)
    * [Training a Simple Neural Network](#training-a-simple-neural-network)
* [Multi-Layered Neural Networks](#multi-layered-neural-networks)
    * [Types of Artificial Neuron](#types-of-artificial-neuron)
    * [Training a Multi-Layered Neural Network](#training-a-multi-layered-neural-network)
* [Tuning in Brief](#tuning-in-brief)
* [Summary](#summary)

# Single Layer Neural Network

## What is a Neural Network?

A neural network is a collection of artificial neurons, arranged in layers which perform different tasks. A Perceptron is a single-layered artificial neural network used for binary classification. The perceptron is no longer widely used due to its simplicity and better techniques being developed in the sixty-plus years since its creation by Frank Rosenblatt. However, it is perfect for our initial learning. The image below shows an illustration of a single biological neuron annotated to describe a single artificial neuron's function. 

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/artifical-neuron.png" alt="A biological neuron annotated with aspects from artificial neural networks, describing ANNs process information." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 1: annotated biological neuron.</p>

A biological neuron receives input signals from its dendrites from other neurons and sends output signals along its axon, which branches out and connect to other neurons. In the illustration above, the input signal is represented by <i>x0</i>, as this signal ‘travels’ it is multiplied (<i>w0 x0</i>) based on the weight variable (<i>w0</i>). The weight variables are learnable and the weight's strength and polarity (positive or negative) control the influence of the signal. The influence is determined by summing the signal input, weight and bias (<i>∑ wi xi + b</i>) which is calculated by the activation function f. If the influence is above a certain threshold the neuron fires, passing data forward through the process. The diagram below shows a simplified view from above.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/perceptron-expanded.png" alt="An expanded view of a perceptron, showing annotations for each stage of processing." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 2: annotated artifical neuron.</p>

Don't worry if you don't fully understand this right now, the discussion and examples below will illuminate this process. Pause here and compare the two diagrams above, follow the written description along the path of the data through the neuron.

#### Math Breakdown

The Greek letter sigma (∑) is used to represent summation. In this context it can be considered analogous to a `for` loop in programming:

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/math-summation-neuron.png" alt="An image describing how the greek letter sigma is used in math and its analogy to a for loop in code." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 3: Sigma symbol explained.</p>

So in our case, the result of the input data and weight multiplication and addition of bias are summed within an activation function <i>f(∑ wi xi + b)</i>. If the result is over a certain threshold the output is passed along. We will discuss activation functions in a later section, here we just know they are chosen for aspects surrounding how sensitive they are - how easily they 'fire'.

### Activation Functions in Brief

The frequency of the firing of an artificial neuron is determined by an activation function. It determines at what threshold the neuron will fire and what the output will look like. This section describes the Sigmoid activation function to bring clarity to the operation of an artificial neuron. There are a number of activation functions which are suitable in different situations.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/sigmoid-activation-function.jpg" alt="A graph showing the sigmoid activation function." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 4: Sigmoid activation function.</p>

The graph above shows the Sigmoid activation function, which will convert all input signals into positive values between 0 and 1. The use of other activation functions with wider values, or negative values change the frequency of the firing of a neuron and will suit different types of data and purposes.

## Training a Simple Neural Network

We will use the Iris dataset, a dataset containing botanical information which has become somewhat of a “hello world” for learning neural networks. Conveniently, the Iris dataset is included in scikit-learn so we don't need to download it separately. The dataset is tiny by machine learning standards, containing only 150 samples. Typically,  machine learning algorithms require thousands of data points to learn effectively, complex algorithms such as convolutional neural networks can require millions for visual recognition tasks. A perceptron can use such a small dataset effectively because of its simplicity. This is perfect for our learning.

The Iris data set consists of 150 samples of four features per sample, measuring the length and width of sepals and petals (in cm), with 50 samples for three species of Iris (Iris setosa, Iris virginica and Iris versicolor). First featured in British statistician and biologist Ronald Fisher’s research article in 1936.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/iris-dataset-species-examples.png" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Images of each of the three flower species. Credit: Ashok Kumar.</p>

We aim to classify flowers from the Iris dataset into their species categories based on the predictions made from the measurements. I encourage you to follow along in <a href="/docs/environment-setup/" target="_blank">Jupyter Notebooks</a>.

Google Colab environments are coming soon.

#### Import Dependancies

```python
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
```

#### Load the Dataset

```python
# Load dataset and assign data to vertices
iris = datasets.load_iris()
x = iris.data
y = iris.target
```

#### View the Data

```python
# Print measurements for the first 5 features 
x[:5]
```
```text
array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       [5. , 3.6, 1.4, 0.2]])
```

We print the 2D array we assigned to `x`. A 2D array is an array within an array, in concept equivalent to rows and columns in a table - this can be seen in the output above.

The features are the measurements mentioned earlier; the height and width of the sepals and petals. Above we print data for 5 samples.

```python
# Print all 150 indices of x axis (the class of each sample)
y[:150]

```

```text
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

```

`y` contains a 1D array with integers representing the three classes of flower within the dataset; Iris setosa, Iris virginica and Iris versicolor - the labels. As we mentioned on the page <a href="/datasets-and-data-collection/dataset-preperation/#handing-non-numerical-data" target="_blank">Dataset Preperation</a> machine learning algorithms can't handle non-numerical data, this is an example of how data is represented so it can be processed and understood by the system.


#### Visualise the Dataset

To get a better idea of what the dataset actually holds, lets visualise it.

```python
# Import dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
 
# Plot graph
plt.figure(2, figsize=(8, 6))
plt.clf()
# Create scatter graph 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
# Axis labels
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xticks(())
plt.yticks(())
# Display graph
plt.show()
```

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/iris-data-graph.jpg" alt="An graph of the iris flower data seperated into categories." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 5: Iris dataset visualised.</p>

The graph shows the distribution of the three flower types, categorized by two of the features; sepal width and sepal length. 


#### Split the Dataset

We must split the dataset to create a test and training dataset. The training data will be processed by the classifier and not used again once training is complete. Once we have trained our system, we can expose it to the new data in the other portion of the dataset. This separation allows us to later compare the accuracy between the different portions of the dataset and see if the algorithm is learning efficiency. Scikit-learn includes handy functions to split datasets for this purpose.

```python
# Set aside 30% of the data set for training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
```

#### Scale Features

Feature scaling is the process of transforming values within a dataset to the same scale to remove issues which may negatively affect an algorithms ability to compare wildly different values during processing. Machine learning algorithms do not work well when data set values vary by large amounts.

Scikit-learn includes the `StandardScaler()` function, which performs standardization. Values are altered to have a zero mean and divided by the standard deviation, standardising unit variance (the distance between data points). This alters the scale of the dataset while retaining the 'distance' between values. Now the algorithm can understand the 'distance' without interference by comparing wildly different values and more accurately predict.

```python
# Scale features
sc = StandardScaler()
sc.fit(x_train)

# Apply the scaler to the split datasets
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
```

#### Training a Perceptron

Now for the actual training. Training machine learning algorithms using libraries such as scikit-learn is surprisingly easy. Algorithms are encapsulated within easy to use functions, here we use the `Perceptron()` function to invoke a perceptron object with the following parameters:

`max_iter` is set to a maximum of 50 iterations over the data set. Iterations are also referred to as "epochs", "epoch" is often used in research papers.

`eta0` is a constant value by which updates are multiplied. This will ensure the values change and don’t stagnate, it can be thought of as bias within the network. We will discuss bias later in this chapter.

`verbose` is a boolean flag (0 or 1, off or on), when set to 1 it will print information for each epoch.


```python
# Create a perceptron with 50 iterations over the dataset, and a learning rate of 0.3
ppn = Perceptron(max_iter=50, eta0=1, verbose=1)
 
# Train the perceptron
ppn.fit(x_train_std, y_train)
```
```text
Perceptron(alpha=0.0001, class_weight=None, eta0=1, fit_intercept=True,
      max_iter=50, n_iter=None, n_jobs=1, penalty=None, random_state=0,
      shuffle=True, tol=None, verbose=0, warm_start=False)
```

If we apply aspects of the scikit-learn perceptron to our example from figure 2.0 we get:

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/scikit-perceptron.png" alt="A diagram showing an annotated perceptron." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 6: Simplified neuron view adapted to Iris dataset.</p>

#### Prediction

Implementing prediction using the algorithm is as simple as calling the `predict()` function. The difficulty comes from ensuring the dataset is clean and representative, your assumptions are sound and the system design is flawless. Of course, this is a simple example.

```python
# Apply the trained perceptron on the X data to make predicts for the y test data
y_pred = ppn.predict(x_test_std)
```

#### Results

```python
# Print the predicted y test data
y_pred
```

```text
array([1, 2, 2, 1, 1, 0, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 0, 1, 2, 1,
       2, 2, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1,
       1])
```

```python
# Print the true y test data
y_test
```

```text
array([1, 2, 2, 1, 1, 0, 2, 1, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 0, 1, 2, 0,
       2, 2, 0, 1, 0, 2, 2, 2, 0, 1, 1, 0, 0, 1, 0, 1, 2, 2, 0, 1, 0, 1,
       1])
```

```python
# Print the accuracy of the implementation
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

```text
Accuracy: 0.89
```

89% accuracy. This was achieved following the recommended parameter settings found in the scikit-learn documentation. However, trial and error manipulation of the parameters can be an important method in machine learning, so don’t be afraid to experiment. Always consider how the parameters are being used internally, how they affect the data and experiment to see how changing them affects the result.

Try changing the parameters for yourself to see how that affects the accuracy. See the scikit-learn perceptron documentation to see other parameters which you can manipulate. Hyperparameter manipulation is an important and necessary part of tuning and getting the best out of neural network implementations. We will discuss tuning as part of the next section, once we have got a grip of training and how algorithms learn during training.

Although we achieved a high rate of accuracy a real model would require evaluation and further consideration. Measuring accuracy in this way can mislead you into thinking your algorithm is complete and performs well. This is only part of creating and using machine learning algorithms.


## Scikit-Learn Under the Hood

Programming libraries like scikit-learn are created to make the programmer's life easier by providing commonly needed functions written by others. With this as their main aim libraries may not implement algorithms straight from textbooks. Additionally, as libraries are created with modularity in mind, their underlying implementation will likely be more complex than expected.

Scikit-learn’s perceptron is no different. The `perceptron()` function shares the underlying implementation of scikit-learns `SGDClassifier()` function, `SGDClassifier(loss=“perceptron”, eta0=1, learning_rate=“constant”, penalty=None)` is equivalent to Perceptron(). Scikit-learns perceptron's functionality differs from traditional simple perceptrons as a result. Traditionally perceptrons required data to be linearly separable, that is to be able to be separated cleanly without overlap - imagine a graph with a single line separating two distinct groups of data points on each side. Scikit-learn's implementation is more flexible because it is implemented as an extension of a more flexible approach which is functionally the same as a perceptron. 


# Multi-Layered Neural Networks

Artificial neurons are typically situated in layers of a neural network defined by the type of work done by the layer. Input and output neurons sit in the first and last layer respectively - the input and output layers - and hidden neurons sit in the hidden middle layer(s).

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/multilayer-nn.png" alt="A diagram of a three layered neural network." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 7: An example of a multi-layered neural network.</p>

The diagram above shows a typical architecture of a neural network. In this example, individual neurons are grouped into layers in which neurons in adjacent layers are connected (fully-connected), but are not connected within the same layer. 

Consider how this network architecture operates and why it is designed this way. Data moves through the network from left to right. Nodes are not connected to any other node in the same layer, this would allow nodes to be influenced by their neighbour's data severely affecting the results. Data is manipulated in isolation in each node and passed forwards, to be manipulated by the nodes of the next layer. This isolation allows neurons to be trained on distinct data if neurons are connected directly the purpose of the layer is blurred and results suffer.

The neural networks we have experimented with and visualized here are feedforward neural networks. Information is passed forwards to the next layer and never backwards to the previous layer. How information is passed around a neural network can drastically change how the network functions. For example, recurrent neural networks (RNN) implement 'memory' by passing information backwards from neurons near the output neurons, allowing the network to retain an amount of information. This can be seen in the diagram below, where information is passed backwards in the hidden layer.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/recurrent-neural-network.png" alt="A diagram of a recurrent neural network." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 8: A diagram of a recurrent neural network architecture.</p>

## Types of Artificial Neuron

We’ve hinted at the existence of different types of neurons which serve different purposes. In this section, we will discuss the different types of neurons which will make up a neural network. Networks don’t have to have every type of neuron, and some neurons can have multiple purposes, but each affects how a network learns.

Neuron naming conventions differ between sources; neurons might also be known as "units" or "nodes".

### Input and Output Neurons

Input and output neurons can be thought of as placeholders which represent information passed into the network and information processed out from the network in the form of vectors or arrays. These vectors typically contain floating-point numbers. The number of elements within the vector is equal to the number of input neurons.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/input-and-output-neurons.png" alt="A diagram of a three layered neural network with the input and output layers highlighted." style="width:600px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 9: A neural network with the input and output layers highlighted.</p>

### Hidden Neurons

Hidden neurons sit in the middle of a network, surrounded by other neurons, they receive input from input neurons or other hidden neurons, and they output to output neurons or hidden neurons. They are never connected to the data or produce output themselves. This is their defining characteristic.  

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/hidden-neurons.png" alt="A diagram of a three layered neural network with the hidden layer highlighted." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 10: A neural network with the hidden layer highlighted.</p>

### Bias Neurons

Below is an example of a network including bias neurons. The diagram shows a neural network designed to compute XOR arguments. Each layer, excluding the input layer, has a bias neuron attached. Towards the end of this page, we will build and calculate the output of this network, and build an example in scikit-learn.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/XOR-NN.png" alt="An example of a neural network which implements bias neurons, an three layered XOR implementation." style="width:350px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 11: An exmple of a network designed to compute XOR arguments.</p>

The addition of bias into a network helps the network learn by allowing the programmer to shift the activation function curve to the left or right, the fine-tuning of this parameter can affect the success of learning. The graph below shows how an activation function can be shifted. Without bias, we can only control the shape of the curve - through the choice of the activation function.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/lateral-shift-bias.jpg" alt="A graph showing multiple lines maintaining shape but moving left or right, representing control over activation functions via bias." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 12: A graph showing lateral movment of an activation function due to bias manipulation.</p>

### Context Neurons

Context neurons exist in some specific types of neural networks and are not present in all network types. For example, recurrent neural networks (RNN) use context neurons as a form of memory to hold on to information from past calculations. They attempt to mimic context created by biological phenomena within human brains. An analogy helps make their purpose clearer; if you’re crossing the street and you hear a car horn you will likely stop and look towards the noise, looking for danger. However, if you were at a sports event and hear a horn from an over-enthusiastic supporter you ignore it. You’ve learned loud noises are important when crossing the street, but can be meaningless in other contexts. Context neurons provide a trivial form of memory in the context of their layer / purpose.

Below is an example of a recurrent neural network architecture using context neurons:

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/context-neuron.png" alt="An example of a recurrent neural network which utilises context neurons to implement a type of memory." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 13: An example of a Recurrent Neural Network.</p>

## Training a Multi-layered Neural Network

To further our understanding of how a neural network learns as it is training we're going to build a small multi-layered network designed to compute XOR arguments. Logic gates are as low as you can go in the building blocks of computers before you hit physics. Lets learn a little about them before we get onto the practical stuff.

### What are Logic Gates?

Logic gates are the basic building block of digital systems, taking one or more binary inputs and creating a single binary output based on internal logic, transistors perform these actions in computer electronics. Logic gates can be connected to create more complex multi-input/output circuits which perform operations such as addition, subtraction, multiplication, inversion etc. These are the building blocks of computer systems.

If you're interested in learning more about logic gatees Khan Academcy has a <a href="https://www.khanacademy.org/computing/computers-and-internet/xcae6f4a7ff015e7d:computers/xcae6f4a7ff015e7d:logic-gates-and-circuits/a/logic-gates" target="_blank">useful article</a>.

### Logical Operations with Neural Networks

Neurons in a network can represent the logical operators (AND, OR, NOT). We will explore this to gain a more concrete understanding of how neural networks act and learn through weights and biases.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/and-or-not.png" alt="Diagram of AND, OR and NOT logic gates implemented as single layer neural networks." style="width:600px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 14: AND, OR and NOT single-layer neural networks.</p>

The output of the network is calculated by summing the input, weights and biases. For simplicity in our example, we will use a step activation function, outputs will be conformed to a boolean output, positive or negative. However, in actual implementations the network is predicting outputs, so a step function would not be suitable, a function which compresses values to a range between 0 and 1, or -1 and 1 would be more appropriate.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/step-activation-function.jpg" alt="Diagram of AND, OR and NOT logic gates implemented as single layer neural networks." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 15: A step activation function.</p>

Lets calculate AND as an example. AND operations require both inputs to be the same to return TRUE (positive), otherwise, it returns FLASE (negative).

( 1 * 1 ) + ( 1 * 1 ) + ( -1.5 ) =  0.5

This output is greater than or equal to 0.5, so the output is true. The output is changed to false if one of the outputs differs from the other:

( 1 * 1 ) + ( 1 * 0 ) + ( -1.5 ) =  -0.5

Note the second input has changed.

### Truth Tables

Truth tables are a method of displaying the output possibilities of logic gates. Below are the truth tables for the logical operations AND, OR and NOT. Note that not simply inverts input values, hence why there are only two values present.

| AND         | OR          | NOT         |
|:------------|:------------|:------------|
| 0 AND 0 = 0 | 0 OR 0 = 0  | NOT 0 = 1   |
| 1 AND 0 = 0 | 1 OR 0 = 1  | NOT 1 = 0   |
| 0 AND 1 = 0 | 0 OR 1 = 1  |             |
| 1 AND 1 = 1 | 1 OR 1 = 0  |             |


### Computing XOR with Neural Networks

Exclusive OR or XOR provides a logical structure which specifics that only one of the inputs is true, not both. It is made up of the basic three logical structures mentioned previously. XOR gives the following truth table:

| XOR         | 
|:------------|
| 0 XOR 0 = 0 |
| 1 XOR 0 = 1 |
| 0 XOR 1 = 1 |
| 1 XOR 1 = 0 |

Now that we have the truth table, lets build an XOR neural network in theory (as opposed to building it with code), reinforcing our understanding of the basic functionality of neurons, how they interact and how a network operates. Then we'll code an XOR neural network.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/XOR-NN-vertical.png" alt="An annotated XOR neural network showing input, bias and weight values of each neuron." style="width:300px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 16: A design of a neural network designed to compute XOR.</p>

Lets break down the network and begin to calculate the output.

We start with the first two unlabelled hidden nodes shown in green - the two nodes directly connected to the inputs x0 and x1.

( 0 * 1 ) + ( 1 * 1 ) - 0.5 = 0.5 (True)

( 0 * 1 ) + ( 1 * 1 ) - 1.5 = -0.5 (False)

Next we calculate the third hidden node:

( 0 * -1 ) + 0.5 = 0.5 (True)

Finally, we calculate the output O:

( 1 * 1 ) + ( 1 * 1 ) -1.5 = 0.5 (True)


### Building a Neural Network to Compute XOR

Lets put this into practice by building a neural network to predict XOR using scikit-learn. We can utilise scikit-learns multi-layer perceptron. We'll also visualise the decision boundaries to help us understand how the algorithm comes to its decisions during training.

#### Setup

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

#### Train XOR Neural Network

Scikit-learn multi-layer perceptron (MLP) 

`hidden_layer_sizes` represents the number of hidden layers within the multi-layer perceptron. In our code below we have 5 layers, one more than our illustrative diagram.

`activation` specifies the activation function.

`max_iter` is the maximum number of training iterations.

`random_state` represents a state to starting point for weight and bias initialisation, a value here allows reproducible results across runs, it doesn't actually set the weight or bias to the value entered.

```python
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

#### Results

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/training-algorithms/xor-decision-boundry.jpg" alt="An annotated XOR neural network showing input, bias and weight values of each neuron." style="width:350px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 17: XOR decision boundries visualised.</p>

```text
MLP's XOR probabilities:
[class0, class1]
[[0.90824236 0.09175764]
 [0.08283202 0.91716798]
 [0.04569506 0.95430494]
 [0.95730544 0.04269456]]
```

## Tuning in Brief

Training a model is only half of the process to successfully build and utilize a machine learning model. We haven't discussed the tuning of parameters and the evaluation of the model in detail on this page. These two important subjects make up the next section. First, we continue the training section. We wrap up training with a discussion of common algorithms and challenges and solutions.

## Summary

On this page, we have learned the basic function of neural networks. We learned about activation functions; we have several choices based on the type of data we have and the purpose of our network. We learned about internal aspects of neural networks; weights and bias, values which alter the output of neurons.

All of these aspects come together to power the mathematical techniques hidden to the user and allow associations to be created between values. This is where we gain meaning from the data. An illustrative example: if we use a network suited to categorisation tasks, the mathematical techniques working behind the scenes including the activation function, network architecture and other functions (ML libraries may employ other functions to achieve tasks) to measure the distance between data points, using that distance and the relationships between points, the network can decide which categories each data point lies.

The next page discusses neural network activation functions in more detail.

---

<p style="text-align: center;">Feedback and questions are welcome!</p>

<p style="text-align: center;">Get in touch <a href="mailto:securitykiwi@protonmail.com">securitykiwi [ at ] protonmail.com</a>.</p>

---

## References

<ol>

<li>Grobler, J. (2020) <i>Classifier comparison</i>. Scikit-Learn Documentation. <a href="http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html" target="_blank">http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html</a></li>

<li>Kumar, A. (2020) <i>Iris Species Classification — Machine Learning Model</i> (Image credit). <a href="" target="_blank">https://medium.com/analytics-vidhya/iris-species-classification-machine-learning-model-8d7fa4e48f81</a></li>

<li>Sangosanya, W. (1997) <i>Logic Gates</i>. University of Surrey, UK. <a href="http://www.ee.surrey.ac.uk/Projects/CAL/digital-logic/gatesfunc/index.html" target="_blank">http://www.ee.surrey.ac.uk/Projects/CAL/digital-logic/gatesfunc/index.html</a></li>

<li>Scikit-Learn (2020a) <i>sklearn.linear_model.Perceptron</i>. Scikit-Learn. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html" target="_blank">https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html</a></li>

<li>Scikit-Learn (2020b) <i>User Guide: 1.1. Linear Models - Perceptron</i>. Scikit-Learn Documentation. <a href="https://scikit-learn.org/stable/modules/linear_model.html#perceptron" target="_blank">https://scikit-learn.org/stable/modules/linear_model.html#perceptron</a></li>

<li>Stanford (2020) <i>CS231n Convolutional Neural Networks for visual Recognition</i>. <a href="https://cs231n.github.io/neural-networks-1/
" target="_blank">https://cs231n.github.io/neural-networks-1/
</a></li>

<li>Wikipedia (2020) <i>Logic Gate</i>. <a href="https://en.wikipedia.org/wiki/Logic_gate" target="_blank">https://en.wikipedia.org/wiki/Logic_gate</a></li>

</ol>
