---
title: Machine Learning Examples in Security
author:
---

<style>p {text-align: justify;} ol {text-align: left;}</style>

Machine learning has become an invaluable tool in many industries, from advertising sales to <a href="https://cloud.google.com/blog/products/gcp/how-a-japanese-cucumber-farmer-is-using-deep-learning-and-tensorflow" target="_blank">farming cucumbers</a>, and cyber security is no different. Detecting external and internal threats, predicting the stage of an attack (e.g. recon, delivery, exploitation, etc), predicting data breaches, analysing social media for intelligence to cyber weather forecasting. This page discusses examples where machine learning has been used to solve problems in security by researchers. Similar systems are used by commercial cyber security companies and no doubt form part of business R&D for future products and features.

My aim is to add projects to the course for a number of the areas below, please consider becoming a Patron to <a href="" target="_blank">support content development</a> and gain access to screencast walkthroughs, Q&As and more.

#### Contents

* [Threat Detection](#threat-detection)
    * [Intrusion Detection](#intrusion-detection)
        * [External Network Intrusion](#external-network-intrusion)
        * [Insider Threat Detection](#insider-threat-detection)
    * [Malware Detection](#malware-detection)
        * [Detecting Malicious Executables](#detecting-malicious-executables)
* [Threat Intelligence](#threat-intelligence)
    * [Threat Prediction](#threat-projection)
        * [Predicting Cyber Weather](#predicting-cyber-weather)
    * [Threat Projection](#threat-projection)
        * [Projecting Attack Stage](#projecting-attack-stage)
    * [Data Breach Detection](#data-breach-detection)
* [Summary](#summary)

## Threat Detection

### Intrusion Detection

Detecting intrusions within a network environment is a task suited to machine learning due to the vast amounts of network data which need to be analysed. Humans require time and skill to dive deeply into network and packet capture data and stand little chance at keeping up with live data feeds. Automated signature-based systems rely on attacks matching a narrow digital signature, which changes with different tools, techniques and behaviour; whereas _machine learning systems can learn and identify previously unseen behaviour_.

### External Network Intrusion

Yin et al applied recurrent neural networks (RNN), a type of neural network with feed-back loop designed to pass some information backwards to retain information in a form of memory, to the problem of intrusion detection. Only relatively recently have deep learning algorithms such as RNN been utilised to their full potential as hardware resources have become more powerful and more accessible to researchers budgets.

The researcher's system design is relatively simple. They <i>process the training data into numerical values</i> > <i>normalize those values</i> > <i>feed the processed features into the RNN</i> for prediction. Yin et al use the NSL-KDD dataset, which contains 41 features, including a number of network (protocol_type, service) and host (logged_in, is_guest_login) features, overall achieving an accuracy of 99.53% and test accuracy of 81.29%. While achieving high accuracy alone isn't the only metric of importance when designing and testing systems - they must be evaluated and conform to other requirements. The table below shows an interesting element we will discuss in later chapters and experiment with ourselves; <i>tuning parameters</i> such as the <i>learning rate</i> and the design of systems in the <i>numer of nodes</i>.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/yin-rnn-table.jpg" alt="Image of the results of Yin et al recurrent neural network with different numbers of nodes and learning rate." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">Accuracy and training time (seconds) of RNN model experimentation, Yin et al.</p>

### Insider Threat Detection

Internal threats (as opposed to external attackers) is an active area of research. Le, et al tackled this problem by comparing three machine learning techniques; <i>logistic regression</i>, <i>random forests</i> and <i>artificial neural networks</i>, against the CERT insider threat dataset. A process pipeline consisting of <i>data ingress</i> > <i>preprocessing for frequency, statistics and user informaton</i> > <i>machine learning algorithm</i> > <i>alert</i> forms the core design of their system.

The work demonstrates such techniques show promise in automating and improving insider threat detection. Results for each algorithm on different sections of data from within the dataset report each has strengths and weaknesses, and while none stand out as a single best technique the researchers point out the value of the type of data which is processed. Their experimentation shows the shortcomings of focusing on instance-based data versus user-based data. While user-based data may be more time consuming to collect and process, it appears to provide far better features for the machine learning algorithms leading to significantly higher malicious behaviour detection for all three algorithms.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/le-tables-results.jpg" alt="Image of the results tables from Le et al's research." style="width:700px;"/></div>

<p style="text-align: center; font-style: italic;">Tables of results from three algorithms, Le et al.</p>

This research represents a concept of importance which is often overlooked. The difference between certain algorithms may be minimal, but the complexity of the systems is often different. In these cases, it is important to remember not to choose the most interesting, fancy neural network, but the simplest tool for the job. This may sound obvious but it is easy to forget and get caught up wishing to use and learn more about a certain technique - a fine goal for a non-production task. In production projects, the choice of an algorithm over another may come down to a few percentage points. In these cases, it entirely depends on the specifics of the project, a 2% drop inaccuracy may be acceptable for the increased ease of use, faster training times and decreased maintenance costs of a simpler algorithm. However, in a large scale project, 2% may be a significant and noticeable jump in the task as it is seen by the end-user.

### Malware Detection

Machine learning systems which are designed to be suited for a narrow application are often applicable to much wider problem sets. Deep Belief Networks (DBN) and Convolution Neural Networks (CNN) are examples of such systems. CNNs are designed to mimic the visual cortex neurons of mammals, processing visual information (e.g. facial recognition). However, research has discovered a number of other areas in which they excel, Natural Language Processing (NLP) for example (e.g. sentiment analysis). In a CNNs case, this is because the 3D nature of physical environments - the data which the visual cortex is designed to process - allows the network programmer to use the third dimension to gain what amounts to a higher resolution of data.

We will explore this in the CNN entry in the <a href="/algorithms-and-techniques/" target="_blank">Algorithms & Techniques</a> section.

#### Detecting Malicious Executables

Ding et al utilized a Deep Belief Network to detect malware, while comparing it against three other ML techniques for the same task; <i>Support Vector Machine (SVM)</i>, <i>Decision Tree</i>, and <i>k_Nearest Neighbour (KNN)</i>. Their aim was to detect malicious executable files by analysing machine operations code, <i>opcode</i><i> (the low-level instructions a computer uses to process data). Ding et al implement pre-training, where a machine learning algorithm is used as an <i>autoencoder</i> to preprocess data which is passed onto the main analysis algorithm. Pre-training can alleviate issues in some unsupervised techniques, allowing the main algorithm to produce better results. In testing Ding et al found the deep belief network and decision tree algorithms perform between 1.5% and 0.5% better than the SVM and KNN. The DBN achieved the best results, although only marginally better than the decision tree, however, DBN can often be more flexible.

## Threat Intelligence

### Threat Prediction

The field of threat prediction is an active area of research with a number of potentially game-changing prediction tasks; predicting burst attacks (DDoS, brute force), predicting malware within a network and even predicting data breaches (discussed in its own section below). The field offers surprising results, often predicting events with a good level of accuracy or providing hours of warning ahead of an event, allowing security teams to focus on a specific threat or them time to take preventative measures.

#### Predicting Cyber Weather

Park et al present a mechanism they describe as a cyber weather forecasting system, FORE (FOrcasing using REgression analysis). FORE uses a technique Park et al created in earlier work called ADUR (Anomaly Detection Using Randomness Check). ADUR monitors network traffic randomness to predict malware worms, aiming to detect worms at an early stage of propagation so the system can alert users or network administrators as early as possible - the same idea of tornado warnings. FORE is the forecasting model which uses data collected by ADUR. The FORE system design has three parts: times series data (data over time) input, linear regression analysis to make a prediction and reliability analysis to reduce false positives.

In testing Park et al's system achieves its goal, it is able to detect a malicious worm when 0.03% of hosts have been infected, whereas ADUR alone detected a worm at 1% infection. This equated to 306 infected hosts at 0.03% and 10,306 at 1%. Park et al use-case focused on research predicting worm propagation across the internet, however, the system is applicable to large corporate networks.

### Threat Projection

Threat projection is a sub-domain of situational awareness, the phrase "situational awareness" originated in the military and refers to the “Perception of the elements in the environment within a volume of time and space, the comprehension of their meaning and the projection of the status in the near future.” (Endsley, 1988). So the purpose of _cyber situational awareness_ is to discover the near-future status of a computer networks threat landscape, this is a function of threat projection - to _figure out what a threat will do in the near future_.

#### Projecting Attack Stage

Network attacks are often said to conform to stages which the attacker goes through in order to achieve their goal. For example, they may conduct reconnaissance looking for weaknesses within the network, weaponise an exploit they have found during the recon stage, moving onto delivery to send the weaponised bundle, etc. There are many models which aim to describe the stages of attack such as the <a href="https://www.lockheedmartin.com/en-us/capabilities/cyber/cyber-kill-chain.html" target="_blank">Lockhead Martin Cyber Kill Chain</a> and stages similar to those below.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/stages-of-a-cyber-attack.png" alt="." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">An interpretation of the stages of a cyber attack.</p>

Ghafir, et al set out to create a system which can detect threat actors, predict the stage an attack is currently in and the likely future stage and alert the security team. Knowing the likely attack stage allows greater visibility of an attackers behaviour and allows security teams to focus their efforts. Their system, MLAPT (Machine Learning Advanced Persistent Threat) is formed of modules which capture a number of indications of compromise (IoC) to predict the attack stage. IoC include known bad IP addresses, malicious URLs etc, detected in the network. MLAPT is split into modules for ease of development and future extension. Modules include: a tor traffic detector, malicious domain name detector, malicious IP detector, malicious SSL certificate detector and a scanning detector, among others. Individually, detectors aren't particularly useful and create false positives, combined the system becomes more capable. Ghafir et al's system feeds the output from detectors through a correlation system which decides if the combination of alerts meets the threshold, then onto a series of separate machine learning algorithms and the most accurate model is chosen. The results show promise, achieving up to 84.8% accuracy. Further improvements could be made, MLAPT current does not utilize many host-based methods to detect malicious behaviour on machines (rather than at a network level). MLAPT also doesn't scan for encrypted content, a very popular behaviour among network intruders and malicious software.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/ghafir-table.jpg" alt="." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">Classification algorithms and the accuracy of the models.</p>

### Data Breach Detection

Liu et al approach the problem of breach detection with surprising accuracy. Their system monitors externally observable properties of a network in two categories; symptoms of mismanagement and malicious activity over time. Symptoms of mismanagement include misconfigurations such as DNS, and malicious activity includes spam and scanning activity. These features are trained on a <i>random foreset classifier</i> which is tested against three real-world incident databases and achieves a prediction accuracy of 90%. Their approach is different from that of many machine learning approaches due to the number of features they use, over 248. Liu et al's paper, '[Cloudy with a Chance of Breach: Forecasting Cyber Security Incidents](#references)' is open, so you can read it from the link in the references.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/liu-table.jpg" alt="." style="width:500px;"/></div>

<p style="text-align: center; font-style: italic;">Accuracy of forecasting against each dataset.</p>

## Summary

There are many applications of machine learning in cyber security, we have only discussed a small number. I encourage you to seek out research from academia and commercial security companies (white papers). Be critical in your reading and remain sceptical of focus on accuracy and complex neural networks. When you come to create your own projects, unnecessary complexity is your enemy.

The page <a href="#" target="_blank">How to Conduct Research</a> (coming soon) in the final section Resources provides useful advise on how to analyse research quality.

The next page discusses how to approach learning, a useful guide to understand learning with the goal of achieving success in your learning and future projects.

---

<p style="text-align: center;">Feedback is welcome!</p>

<p style="text-align: center;">Get in touch <a href="mailto:securitykiwi@protonmail.com">securitykiwi [ at ] protonmail.com</a>.</p>

---

## References

<ol>
    <li>
        Yin, C., Zhu, Y., Fei, J., and He, X. (2017) <i>A deep learning approach for intrusion detection using recurrent neural networks.</i> IEEE. <a href="https://ieeexplore.ieee.org/document/8066291" target="_blank">https://ieeexplore.ieee.org/document/8066291</a>.
    </li>

<li>
    Le, D., Zincir-Heywood, A. (2019) <i>Machine learning based INsider Threat Modelling and Detection.</i> IEEE. <a href="https://ieeexplore.ieee.org/document/8717892" target="_blank">https://ieeexplore.ieee.org/document/8717892</a>
</li>

<li>
    Ding, Y., Chen, S., and Xu, J. (2016) <i>Application of Deep Belief Networks for opcode based malware detection.</i> IEEE. <a href="https://ieeexplore.ieee.org/document/7727705" target="_blank">https://ieeexplore.ieee.org/document/7727705</a>
</li>

<li>
    Ghafir, I., Hammoudeh, M., Prenosil, V., Han, L., Hegarty, R., Rabie, K., and Aparicio-Navarro, F. (2018) <i>Detection of advanced persistent threats using machine-learning correlation analysis.</i> Elsevier. <a href="https://www.sciencedirect.com/science/article/pii/S0167739X18307532" target="_blank">https://www.sciencedirect.com/science/article/pii/S0167739X18307532</a>
</li>

<li>Park, H., Sung-Oh, D., Lee, H., and Hoh, P. (2012) <i>Cyber Weather Forecasting: Forecasting Unknown Internet Worms Using Randomness Analysis.</i> Springer. <a href="https://doi.org/10.1007/978-3-642-30436-1_31" target="_blank">https://doi.org/10.1007/978-3-642-30436-1_31</a>
</li>

<li>Liu, Y., Sarabi, A., Zhang, J., Naghizadeh, P., Karir, M., Baily, M., and Liu, M. (2015) <i>Cloudy with a Change of Breach: Forecasting Cyber Security Incidents.</i> USENIX. <a href="https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/liu" target="_blank">https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/liu</a> </li>
    
<li>Sato, K. (2016) <i>How a Japanese cucumber farmer is using deep learning and TensorFlow</i> <a href="https://cloud.google.com/blog/products/gcp/how-a-japanese-cucumber-farmer-is-using-deep-learning-and-tensorflow" target="_blank">https://cloud.google.com/blog/products/gcp/how-a-japanese-cucumber-farmer-is-using-deep-learning-and-tensorflow</a></li>
    
</ol>