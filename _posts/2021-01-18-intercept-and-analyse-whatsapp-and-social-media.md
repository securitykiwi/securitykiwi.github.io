---
title: Intercepting and Analysing WhatsApp and Social Media
author: Kris
categories: [Q&A, OSINT]
---

This post is part of a question & answer sent to securitykiwi [at] protonmail [dot] com. Users <a href="/contact" target="_blank">send</a> in questions relating to machine learning and I answer them via email and collate the answers for <a href="https://www.patreon.com/securitykiwi" target="_blank">Patreon supporters</a> to benefit from community questions. 

### Question

'How to intercept (capture & analyse) whatsapp and social media messages automatically?'

### Answer

Social media (e.g. twitter, instagram, facebook, etc) have APIs which can be used to scrape (e.g. <a href="https://developer.twitter.com/en/docs/twitter-api" target="_blank">twitter API</a>, <a href="https://developers.facebook.com/docs/instagram-basic-display-api" target="_blank">instagram basic display API</a>) and making posts, profiles, images, videos etc posted available to any programs you create. Many of these social media platforms have open source intelligence programs built for them by people on GitHub (and other places). For example, twitter has <a href="https://github.com/twintproject/twint" target="_blank">TWINT</a> (Twitter Intelligence), incidentally I created a short guide to <a href="https://github.com/krisbolton/twint-in-jupyter" target="_blank">getting started with TWINT</a> in Jupyter Notebooks. Searching the platform and the word scraper will likely be your best bet here. I haven't used it, but searching "whatsapp scraper" brings up <a href="https://github.com/JMGama/WhatsApp-Scraping" target="_blank">this scraper</a> on GitHub.

This is how to get the data. Because this information is human-readable you can do analysis by hand, typically in any program which can read CSV files. However, you can gain more control over analysis and use machine learning techniques by using python and libraries such as <a href="https://pandas.pydata.org/" target="_blank">Pandas</a>.

The techniques and libraries you'll use are related to data science, an area intertwined with machine learning. Data science uses machine learning to analyse data, e.g. clustering, categorising and otherwise manipulating data you fetch from the different APIs using those open source scrapers. Books like <a href="https://bedford-computing.co.uk/learning/wp-content/uploads/2015/10/Python-for-Data-Analysis.pdf" target="_blank">Python for Data Analysis</a> and <a href="https://jakevdp.github.io/PythonDataScienceHandbook/" target="_blank">Python Data Science Handbook</a> will be using references. You don't need to read them all. I suggest reading the introduction chapters if you're not familiar with the techniques. The security kiwi chapter on <a href="https://security.kiwi/docs/considering-data/" target="_blank">datasets & data</a> sources will be useful reading. The libraries we discuss in this chatpter, <a href="https://scikit-learn.org/stable/" target="_blank">Scikit-learn</a>, a python machine learning library, as well as the library Pandas. Pandas is a great tool for data science, one particular feature 
Other books on Open Source Intelligence may be useful to you as well, e.g.: <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html" target="_blank">DataFrame</a> allows you to organise data into rows and columns - a table - and provides functions to manipulate tabulated data.

Other books on Open Source Intelligence may be useful to you as well, e.g.: <a href="https://www.elsevier.com/books/automating-open-source-intelligence/layton/978-0-12-802916-9" target="_blank">Automating Open Source Intelligence</a>. Any books like this that are expensive, look at the content page and then search the techniques individually. With this sort of technical thing, there usually isn't one reference, the techniques will have existed for a long time and be discussed online in detail. You just needed to know the names of the techniques or programs to search for them.

A machine learning technique which comes to mind beyond clustering etc is <a href="https://en.wikipedia.org/wiki/Sentiment_analysis" target="_blank">Sentiment Analysis</a>. A technique where the system attempts to interpret the sentiment - the opinion - of a written message. Typically, this is measured in three to five levels: good, neural bad, or very good, good, neural, bad, very bad. Each equating to the strength of the opinion expressed. An example of this is use was reading twitter sentiment during the London 2012 olympics and showing colour-coded results on the <a href="http://sentistrength.wlv.ac.uk" target="_blank">London eye</a>. A security and much more interesting example has been its use to <a href="https://arxiv.org/abs/1804.05276" target="_blank">predict cyber events my measuring hacker sentiment</a>.

I hope this helps.

Kris

### Do You Have a Question?

If you have a question related to machine learning please do <a href="/contact">send</a> it in.
