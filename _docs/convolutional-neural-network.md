---
title: Convolutional Neural Network
author: 
---

<style>p {text-align: justify;}</style>

Convolutional Neural Networks (CNN) are a type of neural network architecture inspired by the architecture a specific part of the mammalian brain - the visual cortex. This area of the brain is optimized to process the three-dimensional data of the real-world; height, width, depth. Neurobiologists don’t appreciate such comparisons, as the similarity to real biological systems is vague, neural networks are simplified models of biological architecture. However, in the computing world, these tools which have been inspired by nature have surpassed existing tools performance and their similarity to biological neural systems fuels the excitment from readers.

## An Extra Dimension

A CNN's ability to process three-dimensional data allows information to be represented in an extra dimension, essentially allowing finer granularity of the representation and a greater range of relationships. Consider a list, a one-dimensional data store with rows, versus a table, a two-dimensional store with rows and columns. A list is useful, we can check items from a list as we pick up our groceries, but we can't gain much insight from analysing it. We need to add and organise information into columns. Even simple information such as grocery items and quantity are tabulated data. Taking the analogy beyond a normal shopping list, we can tabulate item nutrition; enegry, saturated fat, sugars, salt etc. Tabulating data allows us to go far beyond the functionality of a simple list - we can establish relationships between items. We can cluster high-fat items or low sugar items for example.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/list-vs-table.jpg" alt="." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">List vs Table. The additional axis (dimension) allows greater insight into the same data.</p>

The addition of a third dimension similarly increases the power of information representation. CNN's have found a place in text analysis for this reason. Their ability to plot words in <i>high-dimensional space</i>, which can reveal meaning which is otherwise unseen. In this example, high-dimensional space can be seen as adding dimensions to words. An analogy can illuminate this: a person can be seen to have dimensions; date of birth, place of birth, hometown, mother, father, siblings etc. If a data point representing a person and their associated information is plotted into high-dimensional space, clusters can be formed and insights derived. Families could be clustered or ages could be clustered and insight derived from those clusters.

<div style="text-align:center;"><iframe src="https://giphy.com/embed/DhPUGysxpsjg6C8e9D" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen style="border-radius: 10px;"></iframe></div>

<p style="text-align: center; font-style: italic;">High-dimensional space (Google Developers).</p>

The gif above shows a visualization of data clustered into high-dimensional space. The MNIST dataset, a corpus of handwritten single digits was analysed to computationally distinguish written numbers. In the image, you can see numbers are clearly clustered together by a machine learning algorithm. You can see occasional errors where the formation of a number has been identified incorrectly, for example, the number 4 for mistaken for the number 9.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/google-developers-high-dimensional-space.png" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">The MNIST dataset in high-dimensional sapce (Google Developers).</p>

A <a href="https://www.youtube.com/watch?v=wvsE8jm1GzE" target="_blank">video from Google Developers</a> explains the concept of high-dimensional space (Go ahead, I'll wait while you watch). The additional dimension allows the machine learning algorithm (the CNN) to learn which numbers are close to others and thus which are likely to be the same. This isn't so different to how other machine learning algoirthms operate, measuring the distance between data points to determine in which category a data point belongs.

## Convolutional Neural Networks in Security

Researchers (<a href="#jeon-et-al">Jeon et al</a>) used the 3-dimensions of a convolutional neural network to learn the behaviour of Internet of Things (IoT) malware by encoding behavioural data recorded by sensors in images. Data was gathered from three areas; memory, 'integrated behavioural features' such as system calls, processes, network information etc, and behavior frequency, where the previously collected data was analysed for frequency. The sensor data was converted into data the CNN can understand, integers, and the data was rescaled to ensure the best possible outcome (see Handing Non-numerical Data and Feature Scaling <a href="/docs/dataset-preparation/" target="_blank">Dataset Preperation</a>). The final stage in the process was 'channelization', where the data is encoded into pixels of images in the seperate Red, Green, Blue (RGB) channels and combined in a single RBG image.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/jeon-et-al-convolutional-neural-network-encoded-info.png" alt="." style="width:400px;"/></div>

<p style="text-align: center; font-style: italic;">Channelization' stage in Jeon et al's system.</p>

These complete RGB images now contain a great deal of sensor data representing the behaviour of the IoT systems they were collected from. Different patterns of behaviour create signitures which can be analysed by the CNN to determine which systems contain benign, non-malicous behaviour and which contain suspicious behaviour which is likely malware.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/jeon-et-al-benign-encoded-data.png" alt="." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">Benign behavioural visual signature (Jeon et al).</p>

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/jeon-et-al-malware-encoded-data.png" alt="." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">Malicious behavioural visual signiture (Jeon et al).</p>

Using this method Jeon et al achieved the highest accuracy of the techniques they compared their own method against, 99.2% accuracy, and the lowest false positive rate of 0.63%. Jeon et al point out the weakness of the approach, which uses dynamic analysis alone to study the malicious behaviour and they do not attempt to mitigate the ability of malware to avoid analysis.

This example is a useful showcase on how data can be encoded into different forms appropriate for different machine learning systems. Jeon et al's research article is _open access_ if you would like to learn more - <a href="https://ieeexplore.ieee.org/document/9097224" target="_blank">Dynamic Analysis for IoT Malware Detection with Convolutional Neural Network model</a>.

## Technical Details of Convolutional Neural Networks

Convolutional neural networks are similar to other neural networks; they are made up of individual neurons with learnable weights and biases, and they calculate the output of functions based on the purpose of that layer of the network. However, they have a number of technical features which differenciate them from other techniques. Below, we discuss the internals of CNN's to gain an understanding of how they work.

<div style="text-align:center;"><img src="/assets/images/convolutional-neural-network-diagram.png" alt="A diagram of a convolutional neural network." style="width:700px;"/></div>

<p style="text-align: center; font-style: italic;">A convolutional neural network architecture.</p>

### Image Optimized

Convolutional neural networks are optimized to process images. Regular neural networks cannot scale well to the typical sizes of images. Say an image is 300 pixels wide and 300 pixels high, to process this image our regular network would have 270,000 individual weights which need to be updated (300 x 300 x 3 = 270,000 (3 because three color channels - RGB)). The fully-connected layers are a disadvantage here and create an enormous cost to using regular neural networks for image processing. CNN's use a technique we explore below to reduce the complexity of image data called convolutions, where filters pass over the image and extract features from that area to pass on to the next layer.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/what-a-computer-sees.png" alt="." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">Left: the image humans percieve. Right: the image computers percieve.</p>

Convolutional neural networks 'see' the image as a vector of values, representing the data at each pixel (as seen above). If you have ever used a <a href="https://en.wikipedia.org/wiki/Hex_editor" target="_blank">hex editor</a> to 'view' an image you'll have seen similar output.


### Filters

The image below shows a convolutional neural network's convolution layer with an input image size of 32x32 (much like the images in the <a href="https://www.cs.toronto.edu/~kriz/cifar.html" target="_blank">CIFAR-10 dataset</a>). Images are processed individually, sections of an image are scanned by a filter, the convolution layer, the filter moves accross the image multiplying the filters values with the original values of each pixel to produce a single output for the filter position. This creates a grid called a feature map, or activation map, of values representing the output for each filter position.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/cnn-filter-feature-map-v2.jpg" alt="." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">Left: Input image. Right: Feature map.</p>

The image below shows the result of filter calculations. The image on the left is transformed into the feature map on the right. In this example, you can see the outline of the a shape similar to the letter "H" from the input image as data in the grid, while the "blank" area is represented by zero values. Note, algorithms may not see this as blank, the colour white still has a value to an algorithm.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/image-representation-convolutional-neural-network-cnn.jpg" alt="." style="width:450px;"/></div>

<p style="text-align: center; font-style: italic;">The symbol from the left (H) represented in the example feature map on the right.</p>


### Strides

Typically, a CNN model looks similar to figure . In this example the network is designed to predict the contents of an image; features of the image are extracted by filters (also called kernels) being applied during the convolution layers. We can think of these filters viewing a small part of an image (or whatever is contained within the matrix) in much the same way you would use a camera to take a panorama, or multi-frame photograph. This is clarified in figure 2.7. The frame (the area in yellow) passes across the field of view (the area in green), instead of taking photographs the filter is gathering features to pass onto the next layer in the network. We can see the filter incrementally moving across the matrix row by row. The number of pixels which are skipped as the filter moves across the matrix is called a stride. The stride can be used to adjust the sampling of a feature set; a small stride results in a dense feature set with only a large portion of the matrix sampled, a large stride reduces density and samples a smaller proportion of the matrix. Large strides will result in some information being lost during the convolution layers as that information is skipped. We aim to reduce the sample as much as we can before we encounter negative effects, down sampling in this way is a common practice which makes our network less complex and easier to train. Negative effects from strides can be overcome using a technique called pooling. 

### Pooling

Pooling layers mitigate the loss of information from strides. Added after the convolution layer and activation layer (e.g. ReLU), the pooling layers operation is similar to a filter being applied, typically much smaller, in 2x2 pixel grids. This small grid gathers a new set of data points for pooled feature maps. In our 2x2 example, it will reduce the feature map by a factor of 2.

There are two algorithms commonly applied to pooling layers; Max and Average. The max algorithm takes the maximum response from the area around the filter, the average algorithm takes the average response from the area around the filter. This allows some of the information lost in the stride to be captured and passed to the next layer.


## Summary

On this page we have learned about the additional dimension used by convolutional neural networks, a useful way to understand CNNs as beginners. We learned the term 'high-dimensional space' and how extra dimensions can allow new analysis and insights to be discovered. We went into detail on how convolutional neural networks function; passing over each pixel of data, striding over data to avoid the need to sample every pixel. Implementing pooling to avoid any negative effects of skipping data during each stride.


Further reading:

* [Stanford CS231n](https://cs231n.github.io/)

N.B. code examples will be added to this page in a future update to allow a more intuative understanding of aspects such as pooling. Check back in a few weeks.

---

## References

<ul>

<li id="jeon-et-al">Jeon, J., Park J. and Jeong, Y. (2020) <i>Dynamic Analysis for IoT Malware Detection with Convolutional Neural Network model</i>. IEEE. <a href="https://ieeexplore.ieee.org/document/9097224" target="_blank">https://ieeexplore.ieee.org/document/9097224</a> </li>

<li>Google Developers (2016) <i>A.I. Experiments: Visualizing High-Dimensional Space</i>. YouTube. <a href="https://www.youtube.com/watch?v=wvsE8jm1GzE" target="_blank">https://www.youtube.com/watch?v=wvsE8jm1GzE</a> </li>

<li>Stanford (2020a) <i>Deep Visual-Semantic Alignments for Generating Image Descriptions</i>. <a href="https://cs.stanford.edu/people/karpathy/deepimagesent/" target="_blank">https://cs.stanford.edu/people/karpathy/deepimagesent/</a> </li>

<li>Stanford (2020b) <i>CS231n: Convolutional Neural Networks for Visual Recognition</i>. <a href="https://cs231n.github.io/" target="_blank">https://cs231n.github.io/</a> </li>

<li>Stanford (2020c) <i>Convolutional Neural Networks (CNNs / ConvNets)</i>. <a href="https://cs231n.github.io/convolutional-networks/" target="_blank">https://cs231n.github.io/convolutional-networks/</a> </li>

</ul>
    