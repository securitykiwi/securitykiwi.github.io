---
title: Neural Network Activation Functions
author: 
---

<style>p {text-align: justify;}</style>

# Neural Network Activation Functions

Activation functions are a mathmatical function which determine the output of an individual neuron based on its input(s). Depending on the activation function used the neuron will activate ('fire') at intervals based on input and create different output. For example, a step function creates a threshold where the neuron fires in two positions (on and off); whereas a sigmod activation function will compress the input to values between 0 and 1 (0.0, 0.1, 0.2 ... 1.0). These two vastly different approaches are suited to different tasks, we discuss these and a number of others below. 

This page is designed to give you a foundational understanding of activation functions and serve as a reference when you encounter them in tutorials or during your own research. Not all activation functions are listed here, only the most widely used.

## Linear Activation Function

The linear activation function simply returns the neuron input value as its output value, it isn't maniuplated between any range. This type of activation function is often used where regression is to be applied, and the network is to learn to output numeric values. For example, modelling linear regression, a statistical technique for determining the relationship between values. Most activation functions are non-linear.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/linear-activation-function.jpg" alt="Linear activation function graph." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 0: Linear activation function.</p>

## Step Activation Function

The step activation function only output two values based on input, 0 or 1. If the input value is below 0.5 the returned value is 0, if the input value is 0.5 or above the returned value is 1. This allows the mapping of boolean values, true and false.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/step-activation-function.jpg" alt="Step activation function graph." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 0: Step activation function.</p>

## Sigmoid Activation Function

The Sigmoid activation function, also known as logistic activaito function, compresses input values to values between 0 and 1.0. Notice on the graph below the S shaped curve lies between the values 0.0 and 1.0. Sigmoid is often used when prediciton is required as probabilities fall between 0.0 and 1.0 (e.g. 0.5 = 50%).

<div style="text-align:center;"><img src="/assets/images/training-algorithms/sigmoid-activation-function.jpg" alt="Sigmoid activation funciton grpah." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 0: Sigmoid function.</p>

Sigmoid can suffer from the vanishing gradient problem, causing training to stall, the softmax function is a better fit when this issue arises.

## Hyperbolic Tangent Activation Function 

The Hyperbolic Tangent activation function, also called the tanh activation function conforms input signals to values between -1.0 and 1.0. It is similar to the Sigmoid activation function, with the additional negative value range. This is useful if your dataset includes negative and positive values as they will be mapped to appropraite negative numbers rather than forced around 0.

This additional range proves useful when the input values to the network are negative. In such circumstances the Sigmoid function reduces such negative values to near 0, which results in the networks parameters being updated less regularly and negatively affecting training. The Tanh function does not suffer this same flaw, as negative values are not forced into a positive range.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/hyperbolic-tangent-activation-function.jpg" alt="Hyperbloic tangent activation function graph." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 0: Hyperbolic tangent activation function.</p>

Tanh function can also suffer from the vanishing gradient problem, as it is essentially, a wider Sigmoid function (expanding from 0.0 to 1.0 to -1.0 to 1.0). 

## Rectified Linear Units (ReLU) Activation Function

The Rectified Linear Units (ReLU) activation function, also known as Rectified Linear (ReL) activation function, does not compress values to a small range like the Sigmoid or Tanh activation functions, which compress values to 0 to 1, or -1 to 1.0 respectively. ReLU outputs the input value or 0 if the input is 0 or less. This allows it to retain the information contained within the inpit, versus compressing and potentially loosing neuance between data. For this reason ReLU is frequently used in deep neural networks.

<div style="text-align:center;"><img src="/assets/images/training-algorithms/relu-activation-function.jpg" alt="Rectified lear unit (ReLU) activation function graph." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Figure 0: ReLU activation function.</p>

ReLU functions are capable of outputting a true zero value, whereas Sigmoid and Tanh functions produce near zero values. This small detail, along with the information it retains in its output, makes ReLU much more flexible and widely applicable. Particularly in situations where this true zero value will be of benefit, such as when an autoencoder is required as in the examples we have touched on in other pages.


## Other Common Activation Functions

### Softmax Activation Function

The Softmax activation function, or normalised exponential function, creates a probability distribution of an input vector. Softmax is often used as the final layer in deep learning architecutres, the purpose of which is to predict which class an input values lies within, softmax forms the inputs which have passed throgh many layers into probabilities.

### Leaky ReLU Activation Function

A leaky ReLU provides more flexability over a normal ReLU by changing the range from 0 to infinity in a normal ReLU to -infinity to infinity in a leaky ReLU. The 'leak' increases the range of the function, changing the behaviour from manipulating all negative values to 0 and mapping them to appropraite negaitve values.

### Swish Activation Function

The Swish activation funcion is similar to the ReLU activation funciton, and was designed to replace it for deep learning applications - it successfully out performs ReLU according to its creators. Swish does not compress values between a range, like ReLU it has a 0 to infinity range. Again, like ReLU, Swish zeros negative values. The reason Swish out performs ReLU isn't entirely clear, Swish has a smoother gradient, unlike ReLUs abrupt change (see figure 5), this may allow minor optimisations which build during training.

## Summary

This page has briefly discussed a number of the most widely used activation functions. This information is not immidiately useful as other chapters have been, this knoweldge will be useful as you come to apply neural networks in tutorials or you're looking to experiment. You will remember a small detail of a specific funciton, be able to come back and find a function which suits your needs or use it as a starting point for Googling.

The next page discusses optimization algorithms.

---

## References

<ol>

<li>Ramachandran, P., Zoph, B., and Le, Q. (2017) <i>Searching for Activation Functions</i>. arXiv.org. <a href="https://arxiv.org/abs/1710.05941v2" target="_blank">https://arxiv.org/abs/1710.05941v2</a></li>

<li>Wikipedia (2020a) <i>Acivation Functions</i>. <a href="https://en.wikipedia.org/wiki/Activation_function" target="_blank">https://en.wikipedia.org/wiki/Activation_function</a></li>

<li>Wikipedia (2020b) <i>Logistic Function</i>. <a href="https://en.wikipedia.org/wiki/Logistic_function" target="_blank">https://en.wikipedia.org/wiki/Logistic_function</a></li>

<li>Wikipedia (2020c) <i>Hyperbolic Tangent Function</i>. <a href="https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent" target="_blank">https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent</a></li>

<li>Wikipedia (2020d) <i>Rectified Linear Unit (ReLU)</i>. <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)" target="_blank">https://en.wikipedia.org/wiki/Rectifier_(neural_networks)</a></li>

<li>Wikipedia (2020e) <i>Softmax Function</i>. <a href="https://en.wikipedia.org/wiki/Softmax_function" target="_blank">https://en.wikipedia.org/wiki/Softmax_function</a></li>

</ol>
