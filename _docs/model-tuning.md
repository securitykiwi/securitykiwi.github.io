---
title: Model Tuning
author: 
---

<style>p {text-align: justify;}</style>

After we have <a href="/model-evaluation-and-tuning/model-evaluation/" target="_blank">evaluated</a> our model during <a href="/model-evaluation-and-tuning/model-experimentation" target="_blank">experimentation</a> we can consider tuning. Tuning is the process of improving the output of our algorithm by altering various parameters and re-evaluating the output after changes have been made.

On this page, we will discuss specific tuning parameters and go through a practical example to get an idea of how tuning can effect algorithms. This page is designed to provide a foundation and prepare you for our projects later on, it is not expected that you will gain a complete understanding of the code for example. Below we use TensorFlow and Keras for the first time and train a neural network on the MNIST hand-written digits dataset which we used in the final part of <a href="/docs/training-regression-models/" target="_blank">Training Regression Models</a>.

#### Contents

* [Tuning Parameters](#tuning-parameters)
* [Tuning Models](#tuning-models)
    * [Regression Model](#regression-model)
    * [Classification Model](#classification-model)
* [Summary](#summary)

## Our Prior Learning

In previous chapters we have discussed <a href="/datasets-and-data-collection/dataset-preparation" target="_blank">Dataset Preperation</a>. Ensuring data is 'clean' and suitable for the technique is the first step to ensuring tuning goes well and results in high performance. Converting non-numerical values, removing duplicate entries and normalization are common process performed to ensure performance. 

We discussed <a href="/training-models/optimization-algorithms" target="_blank">optimization algorithms</a> in the training section. They can also form part of our tuning process. Different optimization algorithms can be chosen to provide better results.

## Regularization

<i>Regularization</i> improves the efficiency of models and solves specific problems such as overfitting by reducing the complexity (the number or size of variables) of the model by adding a penalty. Regularization has two common forms:

<i>Lasso regression</i> (least absolute shrinking and selection operator (Lasso / LASSO)), also called L1 regularization, is a process which adds a penalty to the loss function (the ordinary least square) which reduces the complexity of data by shrinking certain values to zero, essentially removing those features. This type of regularization is useful for reducing the features (values) of large datasets, increasing efficiency.

<i>Ridge regression</i>, also called L2 regularization, is a process which adds a different type of penalty to the loss function, shrinking weights to near zero - but not absolute zero, allowing input features to be preserved.

## Tuning Parameters

There are a large number of tuning parameters depending on the task, specific implementation and software library used. As we cannot cover them all, nor would a large list and explanation be particularly exciting to read. We are going to build two models and tune them for a more practical approach. As we move on to projects, we will describe tuning in detail and further detail on tuning can be found in the documentation of the library you plan to use.

Despite the large number of tuning parameters, not all need to be altered during experimentation. The need to tune which parameters will arise from considering the output from evaluation metrics, as well as trial and error. Only one or two parameters should be changed at once, so you can compare the difference and see the effect more clearly. Choosing the hyper-parameters which you will focus on for specific experimentation will help you be methodical in your process.

First, we discuss a number of tuning parameters to get an idea of what hyper-parameters are.

#### Terminology Reminder

**Model parameters** are internal model variables usually not set by the programmer used and updated by the model during training. For example, weights which are altered as a neural network learns.

**Hyper-parameters** are external variables to the model and may be part of the manual tuning process of the model by the programmer. For example, the number of iterations an algorithm makes over a dataset can be tuned to increase performance (time to completion). 

There are two sub-types of hyper-parameter:

**Model hyper-parameters** which alter internal model variables, such as the number of hidden layers in a neural network.

**Algorithm hyper-parameters** which alter the speed and quality of learning, such as the learning rate.

### Learning Rate

The learning rate is the rate at which your model is trained, it is typically found in neural networks. Adjusting the learning rate has opposing results; a high rate results in a slow but meticulous training and a low rate results in fast training with the caveat of the possibility of the optimal weights being missed. The learning rate is decreased over each interaction resulting in the network beginning with a broad search space and narrowing its search as it finds optimal weights.

### Momentum

Momentum is a parameter that allows weight changes to continue progress even when the gradient should reverse its direction. Imagine a bowling ball, representing a weight, being thrown down a hilly bowling lane. The bowling lane represents a line on a graph with peaks and troughs. If the ball loses momentum, when it approaches a peak it will roll back and not progress forwards. Beyond this peak is a more optimal point which we would not have found because the algorithm, our ball, never would have reached it. We discussed momentum in <a href="/training-models/optimization-algorithms/#momentum" target="_blank">Optimization Models</a>, with the analogy of a blind-folded climber.

### Batch Size

Batch size is the number of training elements which must be calculated before weights are updated. Training algorithms using batch size sum all of the gradients for the batch before weight are updated, this process is called online training, which we discussed in detail in the <a href="/docs/introduction-to-training-models/" target="_blank">Introduction to Training Models</a>. Typically, a batch size equal to 10% of the size of the initial data set is all that is required.

### Number of Iterations / Epochs

The number of iterations also referred to as <i>epochs</i>, is the number of times a machine learning algorithm processes over the dataset before it converges at an optimal value. Too few iterations and the model may fail to converge and not find an optimal outcome, depending on the software library you are using you may get an error indicating this, for example, Scikit-learn will return an error stating the issue. Too many iterations may also negatively affect your model.

### Number of Hidden Layers

Specific to neural networks, the number of hidden layers has a profound effect on how neural networks are trained. Keeping your network as simple (as few layers) as possible is an important consideration to ensure it runs quickly and does not have negative side effects such as overfitting. However, too few layers will also have a negative effect, a balance needs to be found.

### Many more

As previously mentioned, there are many hyper-parameters not listed here. You can view the hyper-parameters you have control over based on the algorithm you are using, along with best practices, from the documentation of the software library you are using.

## Automating Tuning

Finding the best hyper-parameter values can be automated. This is a powerful concept allowing you to gain far greater accuracy and try many more variations of tuning combinations. <i>Grid Search</i> is a technique which allows you to specify the values for hyper-parameters and an algorithm will try many combinations to try to find the best outcome. However, automation does have the downside of increasing the time to complete yet another computationally expensive task. We automate our tuning in the example below.

## Tuning a Model

We will work with the MNIST dataset of hand-drawn digits, which we used in the training section to <a href="/training-models/training-regression-models" target="_blank">train a logistic regression model</a>. In this sub-section, we will use Keras and TensorFlow, rather than scikit-learn, for the first time. We will introduce them both on a dedicated page later on. The purpose of this section is to understand how changes in hyper-parameters affect the network, do not be discouraged if you do not understand all of the code. We are interested only in the concepts and output.

Below we follow an annotated TensorFlow tutorial from the documentation on <a href="https://www.tensorflow.org/tutorials/keras/keras_tuner" target="_blank">Keras Tuner</a>, a library specifically designed for tuning models in TensorFlow and Keras. We do this to expose ourselves to the best practices observed in the TensorFlow documentation, which you will find yourself diving into in the near future no doubt.

One major difference to the way we have worked in short the past is the creation of separate functions for specific purposes, which can be later called by the program. We have reserved this for projects so we don't over complicate things for new leaners. Generally in programming splitting actions up into functions is a good idea, it increases modularity, ease of maintenance and makes the program easier to understand if done well. If you're unfamiliar with the python syntax or otherwise feel you may benefit, do check out the <a href="/introduction/python-refresher" target="_blank">Python Refresher</a> from the introduction section.

#### Setup

We will be using Keras Tuner, another library specifically for tuning models in TensorFlow and Keras. You will need to install it in your notebook.

```python
# Install Keras Tuner
!pip install -q -U keras-tuner
```

Import the following dependencies.

```python
# Import dependencies
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
import IPython
```

#### Get the Data

Using Kera's built-in method we get the dataset. Similar to scikit-learn the MNIST dataset Keras provides a simple method to get the dataset.

```python
# Get the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
```

#### Preprocess the Data

We normalize the pixel values, ensuring they fall between 0 and 1. This prevents large differences in values negatively effecting our models learning.

```python
# Normalize pixel values
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0
```

#### Create the model

We follow TensorFlow and Keras model building approach, by defining a method which will return a complied Keras model. The function takes the parameter `hp` (hyper-parameter) which defines the range of values that the model can derive hyper-parameter values from. This may seem daunting, how do you know what code to write? what functions to call? This is built from an example on the TensorFlow <a href="https://www.tensorflow.org/tutorials/keras/keras_tuner" target="_blank">documentation</a>. We don't need to reinvent the wheel, nor do you when you create your own models.

Much of this is readable near enough to English to understand what is going on. However, you can look up individual functions in documentation to understand what a particular section is doing. You will find yourself doing this a lot as you research previous work.

```python
# New function containing the model structure
def model_builder(hp):
  # Set the model type to sequential, allowing a series of layers
  model = keras.Sequential()
  # Flatten the images so they are uniform
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  
  # Tune the number of units in the first Dense layer, optimal value between 32-512
  hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
  # Set the activation function to relu, a typically high performing function
  model.add(keras.layers.Dense(units = hp_units, activation = 'relu'))
  # Create 10 layers
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer, optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
   
  # Compile the model
  model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics = ['accuracy'])
  
  # Return the compiled model when the function is used
  return model
```

As you learn you will also pick up knowledge on what functions are appropriate for different tasks. We used the `sequential` function here to make the model of type sequential, which is appropriate when you want layers which accept one input and one output. We use the Adam optimization function, we quickly find this is a recommended algorithm when searching this specific library documentation and previous work.

#### Perform Tuning

Now we perform the tuning. We use the <a href="https://keras-team.github.io/keras-tuner/documentation/tuners/" target="_blank">Hyperband</a> tuner built into Keras. Predefined functions are used to set values, for example, the `val_accuracy` function assigned to the objective, tells the tuner which value it is to optimize (i.e. the objective is to create high accuracy). We set the number of epochs and the factor to reduce the number of epochs by, bringing the training to a halt.

```python
# Create a Keras tuner
tuner = kt.Hyperband(model_builder,
                     # Our objective is to gain high accuracy
                     objective = 'val_accuracy', 
                     # Set our max interations to 10
                     max_epochs = 10,
                     # Reduction factor for the number of epochs
                     factor = 3,
                     # State the directory
                     directory = 'my_dir',
                     # Name hte project
                     project_name = 'intro_to_kt')   
```

The hyperband tuner trains a number of models for a few iterations each and continues only with the highest performing half of the models for a next round, culminating in a high performing model.

Next, we create a function to clear the training output which is used in the next step to clean up after running.

```python
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)
```

Next, we run the tuner search, where we run the tuner we created above on our data to find the best values according to the settings we chose.

```python
tuner.search(img_train, label_train, epochs = 10, validation_data = (img_test, label_test), callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
 
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
```

Finally, after finding the optimal hyper-parameter values we retrain the model using these values.

```python
# Retrain the model with optimal values
model = tuner.hypermodel.build(best_hps)
model.fit(img_train, label_train, epochs = 10, validation_data = (img_test, label_test))
```


## Summary

On this page, we gain foundational knowledge of tuning. First, we briefly reminded ourselves of prior sections which covered dataset preparation and algorithm optimization, two areas which also significantly impact the performance of a machine learning system. Without these it doesn't matter how much tuning you perform, the system will not reach the level of performance it is truly capable of.

We were introduced to regularization, you may have recognised L1 and L2 for other reading or exploring algorithms. Regularization reduces the complexity of a machine learning algorithm by creating a penalty which reduces the size and number of weights. 

We discussed a number of hyper-parameters you may decide to control, this list was by no means exhaustive and served as an intro to the concept of what type of things we as the systems programmers have control over. We will further gain experience in tuning during projects in the sections and pages ahead.

Finally, we created a classifier in TensorFlow using Keras, to classify the hand-written digits of the MNIST dataset. We expanded on the TensorFlow documentation and explained more about how the system works and what different methods actually do.

The next section introduces system design. This forms part of the main objective of this course, to teach a project-oriented approach which will enable you to more confidently and successfully approach your own machine learning projects later on.

---

## References

<ol>
    
<li>TensorFlow (2020) <i>Introduction to the Keras Tuner</i>. TensorFlow Documentation. <a href="https://www.tensorflow.org/tutorials/keras/keras_tuner" target="_blank">https://www.tensorflow.org/tutorials/keras/keras_tuner</a></li>

<li>Keras (2020a) <i>Keras Tuner documentation</i>. <a href="https://keras-team.github.io/keras-tuner/" target="_blank">https://keras-team.github.io/keras-tuner/</a></li>

<li>Keras (2020b) <i>Tuners</i>. Keras Documentation. <a href="https://keras-team.github.io/keras-tuner/documentation/tuners/" target="_blank">https://keras-team.github.io/keras-tuner/documentation/tuners/</a></li>

<li>Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., and Talwalkar, A. (2018) <i>Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization</i>. arXiv.org. <a href="https://arxiv.org/abs/1603.06560" target="_blank">https://arxiv.org/abs/1603.06560</a></li>

<li>TensorFlow (2020a) <i>Introduction to the Keras Tuner</i>. TensorFlow Documentation. <a href="https://www.tensorflow.org/tutorials/keras/keras_tuner" target="_blank">https://www.tensorflow.org/tutorials/keras/keras_tuner</a></li>

<li>TensorFlow (2020b) <i>Hyperparameter tuning with Keras Tuner</i>. TensorFlow Documentation. <a href="https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html" target="_blank">https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html</a></li>

</ol>
