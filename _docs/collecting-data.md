---
title: Collecting Data
author:
---

<style>p {text-align: justify;}</style>

Once we have considered what the data is for and we have created at least a rough set of requirements, we can collect data. Data can be gathered using numerous techniques and tools; for example, software sensors can be placed within a network environment or data can be gathered from individual machines.

Analysis tools can provide data, such as <a href="https://www.wireshark.org/" target="_blank">WireShark</a>, a popular packet inspection tool can record and save network information. Capturing traffic over a long period can provide a sufficient dataset to use with machine learning techniques. There are many different software applications, or custom solutions, which can capture various information to create useful datasets.

We can gather data using <i>programming libraries</i> (also called <i>packages</i>, <i>software libraries</i> or simply <i>libraries</i>) to interface with computer hardware and operating systems. Programming libraries are files of code you can add to your own files, written by others which perform specific tasks. For example, provide interfaces for the use of the current time, gather statistics from hardware, and much more. They save us the time and effort required to solve these problems on our own. We use python libraries below.

## A Trivial Example

The following three-line script illustrates how the Python library psutil (python system and process utilities) can get a local machines CPU utilization percentage.

```python
import psutil
current_cpu = psutil.cpu_percent()
print(current_cpu)
```

Importing `psutil` allows us to get the current CPU utilization percentage by using the `.cpu_percent()` method and assign the value to the variable `current_cpu`, we then print the value. The complexity of the task is hidden - or abstracted away - by the use of the library. As the programmer, we can use three lines to do what would have taken a reasonable effort to complete otherwise.

This trivial example shows how we can gather system informaiton statistics, in this case they can be combined with other system statistics and give an indication a local machine has been comrpmoised (increased resource usage). A machine learning program can learn usage patterns from data captured over some time and create alerts when usage becomes unusual.

Much like Apple's "There's an App for that!", there will be a library that does what you what. Forming Google searches around your ideas and searching StackExchange and similar Q&A websites is a large part of figuring out how to collect data yourself.

Once you have collected data, such as CPU utilization you will need to save it in a format which can be used later on, CSV (comma separated value) and JSON (JavaScript Object Notation) are common data-interchange formats which are widely used and for which significant documentation and assistance exists online.

## Research Similar Works

Collecting data that is appropraite for your task is a deceptatively difficult task and has implications for the quality of the outcome of your algorithm. _Gatbage in, garbage out_, as they say. Academic researchers spend a great deal of time writing about their methodology, considerations, weaknesses, strengths etc of their research. Particular focus should be paid to research articles which discuss new datasets, or which are surveys of several datasets. These are gold mines of insight. The page <a href="docs/how-to-conduct-research/" target="_blank">How to Conduct Research</a> contains useful information for those looking to improve their research skills.

## Validate Your Data

Once we have gathered data, we should carefully consider it against requirements to perform a deep analysis of its quality and ensure it is representative and approrpaite. You can read more about this in the <a href="/datasets-and-data-collection/considering-data/" target="_blank">previous page</a> where we discuss requirements, if you haven't already.

We discuss <a href="/datasets-and-data-collection/challenges-with-datasets/" target="_blank">Challenges with Datasets</a> in a page ahead.

## Tool Kits

As well as exploring ideas and testing hypotheses, academic researchers create resources and tools to help themselves and other researchers conduct research more efficiently, predictably or any number of other goals. A popular way to acheive this is to create a tool kit, a collection of resources, discussions, architectures and possibly code.

<a href="#id2t">ID2T</a> is one such tool kit desinged to help researchers create datasets for machine learning-driven intrusion detection systems. The Intrusion Detection Dataset Toolkit (ID2T) provides instruction on how to insert synthetic traffic into datasets specifically for testing intrusion detection systems. During the testing of such systems we, as the system tester, need to know when an attack is occuring, so we can know if what the system detects is an attack. These instances become our labels in our supervised approach.

Similar tool kits can be found searching academic databases, such as <a href="https://scholar.google.com/" target="_blank">Google Scholar</a>, <a href="https://ieeexplore.ieee.org" target="_blank">IEEE Xplore</a> and <a href="https://dl.acm.org/" target="_blank">ACM Digital Library</a>. Take a look at the <a href="/docs/how-to-conduct-research/" target="_blank">How to Conduct Research</a> page for more.

## Seeking Help Online

If you are a seasoned programmer or tinkerer, you won't need me to tell you to seek out help online. However, if you are new to programming I want to draw attention to a number of the communities online who can help you if you cannot solve a problem. Seasoned programmers, developers, software engineers - whatever their title - are always Googling.

* <a href="https://stackexchange.com/" target="_blank">StackExchange</a> 
    * <a href="https://datascience.stackexchange.com/" target="_blank">StackExchange Data Science</a>
    * <a href="https://math.stackexchange.com/" target="_blank">StackExchange Mathematics</a>
* <a href="https://stackoverflow.com/" target="_blank">Stack Overflow</a>
* <a href="https://www.quora.com/" target="_blank">Quora</a> 

If you have any other suggestions, or non-English-speaking options drop me an <a href="mailto@securitykiwi@protonmail.com" target="_blank">email</a>.

## Summary

This page introduced us to gathering data and creating our own datasets. This process involves numerous considerations around the problem we are trying to solve to ensure we collect the right data. Once we have decided what data we need to gather, we move onto the most challenging undertaking, creating a program to do what we want. To novice programmers this can be a discouraging stage, it is important to not to give up, as we can gain help online. We discussed several websites to gain help, mostly through Google searches to see if existing answers exist, if they do not, we can use Q&A websites or forums to gain specific information.

Once we have gathered our data we must validate it against a series of requirments. These requirements can be build from researching pervious work, as well as considering the systems purpose and the data we are collecting. Once validated, we can feel confident using the dataset.

The next page discusses existing datasets and data sources.

---

## References

<ol>
    
<li> billiejoex, Rodola, G. (2020) <i>psutil</i>. <a href="https://pypi.org/project/psutil/" target="_blank">https://pypi.org/project/psutil/</a></li>

<li id="id2t">Cordero, C., Vasilomanolakis, E., Milanov, N., Koch, C., Hausheer, D., and Mühlhäuser, M. (2015) <i>ID2T: A DIY dataset creation toolkit for Intrusion Detection Systems</i>. IEEE. <a href="https://ieeexplore.ieee.org/document/8317091" target="_blank">https://ieeexplore.ieee.org/document/8317091</a></li>

<li>McKinney, W. (2017) <i>Python for Data Analysis</i>. O'Reilly Media. <a href="https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662" target="_blank">https://www.amazon.com/Python-Data-Analysis-Wes-Mckinney/dp/1491957662</a></li>
    
<li>StackExchange (2020) <i>StackExchange: Sites</i>. <a href="https://stackexchange.com/sites" target="_blank">https://stackexchange.com/sites</a></li>

<li>WireShark (2020) <i>WireShark</i>. <a href="https://www.wireshark.org/" target="_blank">https://www.wireshark.org/</a></li>    

</ol>
