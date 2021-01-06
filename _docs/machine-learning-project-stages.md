---
title: Machine Learning Project Stages
author: 
tags: [setup]
---

<style>p {text-align: justify;}</style>

Machine learning projects go through at least five stages; planning, data wrangling, training, evaluation, and tuning, with the optional stage of deploying (including scaling). This course is based around these stages to ensure you learn the processes to be successful in designing and building any machine learning project idea you have - not just the examples in this course. This page describes each project stage. We will learn more about the topic of these stages and implement them in later sections.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/machine-learning-project-stages.png" alt="A visualization of six stages in machine learning listed below." style="width:250px;"/></div>

<p style="text-align: center; font-style: italic;">Six stages machine learning projects go through.</p>

* **Planning** - considerations, how to design a machine learning system, etc.
* **Data wrangling** - how to view a dataset, how to prepare data for your algorithm(s).
* **Training** - how algorithms learn, how to train them and how to overcome challenges.
* **Evaluating** - how to evaluate your system performance and compare models.
* **Tuning** - how to tune a system to improve and automate improvement.
* **Deploying** - considerations for deploying your system, scaling a system for large deployments.

## Planning

Machine learning can be a complex subject with a number of stages to consider data and check assumptions before you decide how you will approach building a system. Approaching a machine learning project as a business project, by creating requirements and mapping out work will help you avoid issues and successfully complete a project. We will build these elements into our projects through the course to make them second nature.

In this sub-section we briefly discuss useful planning tasks and tools; requirements and work breakdown structures. Feel free to research and add other planning elements to your project to suit your needs. The increased structure around your project may help you achieve success, or you may view it as a hindrance. At a minimum I advise something akin to requirements and a work breakdown structure, to help you understand your project in greater depth.

### Dataset Requirements

Creating a series of requirements for a dataset is a useful step to check assumptions and help you think through how you will actually use the data, whether you are creating a dataset from scratch or using an existing one. A full example of dataset requirements are laid out in the up-coming section <a href="/datasets-and-data-collection/" target="_blank">Datasets & Data Collection</a>. We won't repeat ourselves here.

### Work Breakdown Structure

A <i>work breakdown strucutre</i> (WBS) is a hierarchical decomposition of a projects deliverables (the project output). It will help you figure out what work needs to be done and in what order. It is primarily a visual hierarchy, however, you can tabulate the data to include extra detail. Creating a WBS is recommended for any major project you attempt, the process of creating it will help you discover in what order things need to be completed and consider issues which may arise.

We will complete WBS for each of the major projects, marked with the word "project" in parenthesis. Below is an example of a WBS from a student research project, the numbers map to sections in a table explaining the steps from the WBS in more detail.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/WBS-dissertation-kirs-bolton.jpg" alt="A work breakdown structure of a student research project." style="width:800px;"/></div>

<p style="text-align: center; font-style: italic;">A work breakdown structure of a student research project.</p>

### Useful Tools

<a href="https://trello.com" target="_blank">Trello</a> is a useful free tool for keeping track of aspects related to projects in a to-do list format. Creating three columns; to-do, in-progress and complete, allows you to track elements through their lifecycle. As you work on items you move them from one column to the next. The to-do list format can be useful for people who find checking off lists encouraging, as you eventually build up a list of things which you have done.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/trello-example.png" alt="." style="width:600px;"/></div>

<p style="text-align: center; font-style: italic;">An example of a three-column Trello layout.</p>

### Design the System

Let's look at an example we will delve into in the <a href="/designing-systems/" target="_blank">Designing Systems</a> section; predicting the stage of an attack (e.g. recon, exploitation, exfiltration). Analysing a single data point may not hold the information required to deduce the stage of an attack. For example, a malicious IP address alone does not describe what stage an attack may be in. However, gathering multiple data points based on the known behaviour of attackers can allow system designers to gather enough information to reasonably expect to create a system with some success.

Data points may include _network-based indications_ such as malicious IP addresses, known malicious SSL certificates, malicious domain names, domain flux, encryption detection, and _host-based indications_ such as system calls and resource usage. Each of these holds little value alone, but when they are detected together, or in sequence, they may hold value and correlate to an attack stage. <a href="#ghafir">Ghafir et al</a> created such as system and achieved promising results, achieving 84.8% accuracy, with future designs able to incorporate more host-based methods and encryption detection to improve accuracy.

Machine learning systems often do not consist of a single algorithm, these are referred to as <i>ensamble system</i> or <i>ensamble learning</i>. The process of pretraining is a popular method to elevate the challenges of certain systems. For example, unsupervised methods are prone to <i>overfitting</i> where the algorithm is sensitive to outliers and noise in the input data, producing poor results. <a href="#ding">Ding et al</a> used pretraining to improve their system using an autoencoder to process information which was passed into a Deep Belief Network. We will discuss design elements in greater detail in a dedicated chapter, this section serves to expose you to some basic ideas.

We discuss these interesting aspects and go through examples to learn how to design machine learning systems later in this course.

## Data Wrangling

Data is of enormous importance in machine learning. The quality, quantity and representativeness (does a dataset match the data your system will work on when it's live) of data needs to be considered and understood before you can prepare a dataset for your algorithm. Those terms; quality, quantity and representativeness are deceptively simple, each has layers which we will discuss and put into practice in later chapters. 

We borrow heavily from the field of data science to prepare and process our data before it can be ingested by a machine learning estimator. We consider specific use-cases and the type of algorithm we manipulate and clean our data so it can produce high-quality insights. Unclean data can produce skewed results, preventing our system from being useful. Aspects of machine learning are often referred to as a "black box" - you don't know how it comes to the decisions it makes, and small changes in the data can have dramatic effects unseen to you. It's important to grasp this concept. Machine learning is inherently experimental, all implementations require tuning and evaluation to obtain optimal results.

We discuss <a href="/datasets-and-data-collection/challenges-with-datasets/" target="_blank">Challenges in Datasets</a>, perform data wrangling on a small sample data set in <a href="/datasets-and-data-collection/exploring-datasets/" target="_blank">Exploring Datasets</a> and learn useful wrangling techniques in <a href="/datasets-and-data-collection/dataset-preperation/" target="_blank">Data Preperation</a>.

## Training

Training and the algorithms ability to "learn" is the stand-out feature of any machine learning system. Understanding how this occurs, the differences between training algorithms and how an algorithm creates better and better output ("learning") will help you create successful systems. We won't delve into details here, that will be expanded on in <a href="/training-models/" target="_blank">Training Models</a>.

Machine learning algorithms are exposed to training datasets, which are subsets of larger datasets representative of data they will analyse. For example, a ML system designed to detect network intrusions will be trained on a dataset which is as similar as possible to the real network it will seek threats in; the same background traffic, protocols, ideally the same day/night cycle, and more. This ensures it learns patterns which are applicable to its final use environment.

Discussions on training may use mathematical equations to express various concepts, and essentially, show the tool which you will use. A reminder if you haven't touched calculus since highschool - take difficult parts in your stride, you do not need to be able to instantly read the equation to understand the content around it and use the tool. We keep complex math out of the course. The foundations we learn and the understanding we develop are enough to create machine learning systems using software libraries such as Scikit-learn and TensorFlow.

## Evaluation and Tuning

Machine learning is an experimental process, you will need to evaluate and tune models you create, and ideally compare one method against another. 

**Evaluation** involves analysing your implementation to measure its performance (e.g. accuracy). We can then use these evaluation metrics to tune and improve our systems to obtain optimal results - tuning.

**Tuning** involves the use of the evaluation metrics, with the optional but recommended step of using automation, to iterate through various parameters until a set of variables is found which produces the most optimal evaluation measure.

We will go over this subject in the section <a href="/model-evaluation-and-tuning/" targe="_blank">Model Evaluation & Tuning</a>, where we implement this stage in every major project we complete. Currently there is one project - <a href="/building-a-spear-phishing-detector/project-introduction/" target="_blank">Building a Spear Phishing Detector</a>.

## Deployment

You have designed a machine learning system to solve a problem, evaluated and tuned it, and now you want to deploy it and use it at scale. This final stage may not appear in all projects, however, if you wish to use algorithms in production at scale you can utilise purpose-built and robust systems like TensorFlow Serve. We will go into detail on this stage in a section which is coming soon.

## Summary

These six stages of a machine learning project will guide you through successful implementations of your own later on. Understanding this high-level overview of the stages a project goes through will help you gain perspective and the reasons why we do various things in the chapters ahead. This course is designed around these stages, presenting foundational material in its first iteration. As time goes by more intermediate and advanced content will be added which represent various stages and additions to stages we already cover.

The next page discusses how machine learning is being used in the security field today.

---

## References

<ol>
    <li id="ding">
    Ding, Y., Chen, S., and Xu, J. (2016) <i>Application of Deep Belief Networks for opcode based malware detection.</i> IEEE. <a href="https://ieeexplore.ieee.org/document/7727705">https://ieeexplore.ieee.org/document/7727705</a>
</li>
<li id="ghafir">
    Ghafir, I., Hammoudeh, M., Prenosil, V., Han, L., Hegarty, R., Rabie, K., and Aparicio-Navarro, F. (2018) <i>Detection of advanced persistent threats using machine-learning correlation analysis.</i> Elsevier. <a href="https://www.sciencedirect.com/science/article/pii/S0167739X18307532">https://www.sciencedirect.com/science/article/pii/S0167739X18307532</a>
</li>
    
<li>Trello (2020) <i>Trello</i> <a href="https://trello.com/" target="_blank">https://trello.com/</a></li>
    
<li>Wikipedia (2020) <i>Work Breakdown Structure</i> <a href="https://en.wikipedia.org/wiki/Work_breakdown_structure" target="_blank">https://en.wikipedia.org/wiki/Work_breakdown_structure</a> </li>
</ol>
