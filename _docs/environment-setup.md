---
layout: doc
title: Environment Setup
author: 
---

<style>p {text-align: justify;}</style>

We will primarily use <a href="https://jupyter.org/">Jupyter Notebooks</a> to put into practice what we learn throughout this course. Notebooks allow us to work with and run Python code easily, rather than learn a fully-fledged Integrated Development Environment (IDE). This page describes how to install Anaconda and Jupyter Notebooks.

Before we continue, each section containing code on this website also has a link to working code in a Google CoLab live environment. This is functionally the same as Jupyter Notebooks, just in-browser. You can decide if you want to install Anaconda and Jupyter Notebooks. I would recommend it, Colab is good for convenience, however.

Whichever you choose, Jupyter or Colab, do explore the code, consider it and perhaps copy it out again into a new cell. Actively learn what it is doing, rather than just running the code in the cell.

### Anaconda

Anaconda bundles Jupyter Notebooks (formerly iPython Notebooks), the Python language, <a href="https://scikit-learn.org/stable/">scikit-learn</a>, <a href="https://www.tensorflow.org/">TensorFlow</a>, <a href="https://matplotlib.org/">matplotlib</a> (and a number of other scientific programs) together for ease of use. We will use each of these throught this course.

1. Download the <a href="https://www.anaconda.com/products/individual" target="_blank">Anaconda</a> installer.
2. Launch Anaconda Navigator.
3. Launch Juypter Notebook from inside the Anaconda.

### Juypter Notebooks in Brief

![](/assets/images/jupyter-main.png)

Juypter Notebook opens in your web browser, running on a server on your local machine. The page that opens is a file browser of sorts, you can navigate and create new notebooks here. New notebooks are created with the plus (+) icon on the top right.

![](/assets/images/jupyter-notebook.png)

Notebooks organise code into cells, so you can group related code together or write notes and comments above/below code. You can set the type (code/markdown (text)) of cell in a drop-down menu. Code in a cell is run by pressing the play button and stopped by pressing stop. If the kernel 'dies' - ceases to function due to an error - you can simply restart it. The status of the kernel is indicated by the circle below the logout button on the right.

### Asynchronous Runtime in Notebooks

Juypter Notebooks runs each cell in a single event loop, this means while one cell is running, no other cell can run. You receive a runtime error; "This event loop is already running". We can apply a small 'patch' into a cell to allow you to run multiple event loops within that notebook. Don't get carried away though, this is to avoid the mild inconvenience of a runtime error, nothing more.

```python
# Asyncio allows multiple event loops to run in Jupyter Notebooks.
# Fixes "RuntimeError: This event loop is already running".
import nest_asyncio
nest_asyncio.apply()
```