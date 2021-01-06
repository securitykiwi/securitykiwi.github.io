---
title: Considering Data
author: 
---

<style>p {text-align: justify;}</style>

We must first consider the purpose of a project. Will it be used to monitor live data from a system or network? Will it be part of a tool to assess possible malware files? Are we simply analysing a dataset? The project's purpose, along with data-specific considerations such as; adequate documentation, specific features, format, size, duration, among others, will allow us to create a series of <i>requirements</i> to ensure a dataset is suitable for its intended purpose. This stage is often not adequately explored, or ignored entirely by books and can lead to difficultly when users come to use datasets and move beyond examples.

Requirements originated in project management theory and are used in software development to work efficiency, reduce mistakes and ultimately ensure the final product does what it was designed for. We can use requirements for our dataset to prevent us from wasting time using a dataset which is unsuitable. Considering our dataset in detail will also help us consider other issues and the specific elements we need to change or understand in our design.

## Dataset Requirements

Requirements are individual functional needs which when satisfied ensure a project or design meets its intended aim. In our case, they will ensure a dataset is suitable for its purpose and _help us understand the weaknesses of the dataset_ which we may have to take action to mitigate.

Below is a list of requirements for a dataset which will be used for threat detection within a network. These requirements were created by thinking through a threat detection system and conducting research around existing network threat detection machine learning research. _You don't have to read entire research papers_ - introductions, conclusions and 'proposed design' sections contain valuable information on project design, datasets, data processes and issues the authors took into consideration.

In particular, the research paper which first presented a dataset is a useful source of why the creators chose certain features to include in the dataset, which can be used to reverse engineer requirements. For example, a dataset creator insisted real background traffic must be must and synthetic attack data can be layered on top, solving the problem of ensuring representative data and knowing when attacks actually occur in the data. In that example, you've picked up the requirement of real traffic background data, and that attack data need not be 'real' (i.e. directly recorded from a network). You would research these to understand the strengths and weaknesses and decide whether it makes sense as a requirement for your purpose.

## Example Requirements for a Dataset

1.	**Real traffic:** Datasets should consist of real traffic, without artificial data inserted post-capture, with the exception of attack traffic. Simulated background traffic can lead to undesired behaviours within models. Simulated attack traffic is desirable so start and end times can be determined with certainty.
2.	**Threat traffic:** Traffic should include the behaviour of threat actors. These behaviours should be visible within traffic so they can be labelled (for supervised methods).
3.	**Features:** Network datasets can include two types of features, network or host-based. Network features consist of elements such as flow duration, number of packets, IP addresses, port numbers and the like. Host features consist of behavioural indications, such as failed login attempts, malicious file detection and similar characteristics.
4.	**Labelling:** For supervised machine learning algorithms dataset records should be labelled as malicious or none-malicious.
5.	**Duration:** Datasets should span at least several weeks. The data should include cyclostationary effects of day/night attack patterns, however, the duration is particularly important when considering APTs as their attacks are conducted over long periods.
6.	**Documentation:** A detailed description, and ideally an analysis of, the dataset is required to understand its limitations.
7.	**Format:** Datasets are provided in several formats, typically CSV, pcap (tcpdump) or flow (netflow). Dataset formats are dictated by many factors, for example, a long data capture duration dramatically affects the size of pcap formats and thus flow formats are typically used for large datasets.
8.	**Size:** Size of the number of records is a factor dictated by deep learning algorithms. They require large datasets which can be split into appropriately sized training and test sets.

### Notes from the Example

Network and host-based indicators are important considerations. Network indicators are captured from the network only, host-based indicators are captured from individual machines. These can be combined to form a higher definition picture of events. However, their use will depend on the purpose of any system you design. A system we discuss later on by Ghafir et al uses a combination of indicators to detect the stage of a threat actors attack, thus giving valueable insight to a security team.

Requirements involving network security will always need to consider duration. The dataset needs to not only capture examples of the type of traffic you wish to analyse, but should be as close to live data as possbile. In the requirements above the day/night cycle of traffic is referenced. Networks likely have a great deal of traffic during the day versus reduced traffic at night, with additional changes such as daily or weekyl backups during non-peak hours. If your training data does not include these neuances, your system may mistake backups or other events as anomolies.

The format of the data will need to be parsed by a custom script as part of any system you design. These are usually relatively simple solutions. For example, converting one file type into another, reading the file contents into a DataFrame (a feature of Panda's).

## Using Requirements

Once we have a list of requirements we can evaluate existing datasets or any we create ourselves, against these requirements. The actual process is simple, we map out each requirement against each dataset. If we have several datasets to evaluate a table can help visualise the best choice, a list can suffice for a single dataset. 

Below is an example table evaluating two DARPA datasets against our requirements. You may need to scroll right to see all of the columns.

| Dataset     | Real Traffic | Intrusions   | Features | Labelled | Duration | Docs | Format | Size |
|:------------|:-------------|:-------------|:---------|:---------|:---------|:-----|:-------|:-----|
| DARPA '98   |     ✗        |      ✓       |Host & Network|     ✓    |     ✓    |   ✓  |  pcap  |  ✓   |
| DARPA '99   |     ✗        |      ✓       |Host & Network|     ✓    |     ✓    |   ✓  |  pcap  |  ✓   |

Once we have mapped out the requirements in this table format, we can quickly get an idea if our proposed datasets meet our needs. While the initial thinking and creating a table containing a larger number of entries can be time-consuming, the actual use of the table is simple. If none of our datasets conforms to our needs, we consider the closest match(es) one last time to ensure they are truly not suitable. If this remains the case, we now know we must continue our search for better datasets or create our own dataset. We discuss collecting your own data on the next page.

## Summary

This page introduced the concept of using requirements to ensure any datasets we use are appropriate for our intended purposes. While creating requirements doesn't have the same appeal as creating neural networks or getting hands-on with code, _the requirements process is invaluable_ to ensure the data is representative, of high quality and suitable for your application - it should not be skipped.

The next pages discuss collecting data to create your own datasets.

---

## References

<ol>   

<li>Burek, P. (2008) <i>Creating clear project requirements: differentiating "what" from "how".</i> PMI: Project Management Institute. <a href="https://www.pmi.org/learning/library/clear-project-requirements-joint-application-design-6928" target="_blank">https://www.pmi.org/learning/library/clear-project-requirements-joint-application-design-6928</a></li>

<li>Radigan, D. (2020) <i>Project Management: Requirements.</i> Atlassian. <a href="https://www.atlassian.com/agile/product-management/requirements" target="_blank">https://www.atlassian.com/agile/product-management/requirements</a></li>

<li>Wikipedia (2020) <i>Requirements Engineering.</i> <a href="https://en.wikipedia.org/wiki/Requirements_engineering" target="_blank">https://en.wikipedia.org/wiki/Requirements_engineering</a></li>
    
</ol>
