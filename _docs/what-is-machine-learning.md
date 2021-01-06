---
title: What is Machine Learning?
author: 
tags: [Basics]
---

<style>p {text-align: justify;}</style>

Early artificial intelligence (AI) initiatives focused on setting rules to simulate intelligence in machines, machine learning seeks to develop and utilize AI behaviour by providing systems with access to data from which they can “learn” – _to act without being explicitly programmed to act in a certain way_.

News that IBM’s <a href="https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)" target="_blank">Deep Blue</a>, a chess-playing computer which was demonstrated to the world in 1997 by defeating Chess Grandmaster Garry Kasparov after three games, met global headlines. A computer defeated a human in what was perceived to be a game requiring human intellect to win. Deep Blue did not have human-level intellect, of course, it relied on explicit programming and computational speed to win. In contrast, recent global headline-grabbing AI, Google’s AlphaGo, leveraged neural networks and reinforcement learning to defeat human players of the Chinese board game Go. AlphaGo is entirely different to Deep Blue, it is not explicitly programmed - it learns.

Deep Mind, an Oxford-based AI startup acquired by Google in 2014, leveraged various machine learning techniques to teach a system called <a href="https://deepmind.com/research/case-studies/alphago-the-story-so-far" target="_blank">AlphaGo</a> what the game of Go looked like. The system was exposed to a large number of amateur games before it was pitted against different versions of itself to learn from its own mistakes and incrementally improve - the ML technique known as <i>reinforcement learning</i>. In 2016 AlphaGo played Lee Sedol, a 9th Dan and one of the worlds best players in a series of live-streamed and televised matches from Seoul, South Korea. Sedol lost four out of five games, winning the fourth to be the only human to beat AlphaGo. The battle was captured in <a href="https://www.alphagomovie.com/" target="_blank">film</a> by director Greg Kohs. Below is an example of a game from the Chinese Future of Go event.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/alphago-tang-weixing-16-dec-2016.jpg" alt="Image of the board game Go, between AlphaGo and Tang Weixing, 2016." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">AlphaGo vs. Tang Weixing. AlphaGo won by resignation. Credit: Wikipedia</p>

Much like Deep Blue, AlphaGo can only do one thing well, play Go. It cannot play Chess, Bingo or any other game. Similarly, other machine learning algorithms can only perform single tasks - the purpose they were created for. Machine learning is a type of “narrow”, or “weak” AI, it can only work on tasks with a narrow scope. Despite the narrow focus the algorithms abilities, _the value of machine learning comes from its ability to predict and recognise patterns_, allowing it to work on unfamiliar data once it has been trained. The programmer doesn’t have to write new code or think of all possible outcomes.

Pseudocode might help further illustrate the idea:

#### Explicit program

```python
If email contains "pills";
If email contains "p!lls"; 
If email contains "pillz";
...
Then mark spam;
```

#### Learning program

```python
Try to classify emails;
Change self to reduce errors;
Repeat;
```

Going beyond the constraints of single-task, narrow AI, is "hard AI", "broad AI" or "general AI". With the ability to perform multiple tasks, understanding context to successfully apply what it has learned previously to entirely new situations. The term AI typically conjures the concept of self-aware computers in the minds of most people, this would be the most sophisticated form of general intelligence in an AI. Realistically, this type of broadly applicable AI will not be possible for decades. For now, an impressive step forwards might take the form of a machine learning algorithm which could play chess and Mario Kart.

## A Brief History of Artificial Intelligence

Artificial intelligence research goes through cycles, as interest peaks due to successful research and implementations, and wane as the limitations of new techniques are discovered over time.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/ai-history-timeline.png" alt="." style="width:650px;"/></div>

<p style="text-align: center; font-style: italic;">Brief History of AI.</p>

The first AI boom occurred during the 1950s with the <i>perceptron</i> and as <i>search algorithms</i> became popular. New methodologies like <i>depth-first</i>, <i>breadth-first</i> and <i>search tree</i> algorithms began to allow problems to be solved which weren’t easily solved previously. The peaks and troughs in interest are not controlled by the whim of researchers, they are often attributed to technical limitations. During the 1950s, new hardware and chip architecture allowed more calculations or provided more memory, which enabled operations which simply weren’t possible to run on previous hardware. It's easy to forget what we consider trivial operations weren’t possible on pre-1950 hardware and the gigabyte system memory we are now used to didn’t become the norm until the 2000s. 

Periods where AI research wasn’t fruitful or fashionable have become known as “AI winters”. The popularity of search algorithm research waned, as new discoveries decreased and hardware limitations restricted advancement. It wasn’t until the 1980s that a new method pushed AI research back into the mainstream for the second AI boom. <i>Knowledge representation</i>, provided new methods such as <i>expert systems</i>, <i>semantic nets</i> and new <i>ontologies</i> which allowed information to be represented in ways computers could better understand and work with.

{% include image.html img="images/openai-ai-and-compute-graph.png" style="wide" lightbox="true" alt="Alt for image" %}

<p style="text-align: center; font-style: italic;">AI compute power 1957 - 2020. <br/> The large jump is attributed to the increased use of GPUs. <br/> Credit: Open AI.</p>

The current AI boom is the result of <i>deep learning</i> and was accelerated by numerous advances in machine learning and the increased use of GPUs. Advances in, and the cost of hardware and cloud computing led to favourable conditions for research spurring AI research into an unprecedented period of growth. Unforeseen and unlikely technologies have affected hardware power and price over recent years. The demand from consumers for affordable, powerful hardware for virtual reality gaming and simultaneously the rise of cryptocurrency mining has injected cash and encouraged GPU manufacturers to cram more transistors onto their silicone to create faster hardware which in turn benefited AI research. It is difficult to suggest a single point for the current boom. It is the culmination of a number of research successes around 2006, <i>deep belief networks</i> and record-breaking deep learning algorithms for example, which lead to increased investment and interest in AI research.  

Machine learning and deep learning are not the final AI boom. No one knows for sure what the next AI boom will be, when or if the next winter will start, or what technique(s) will lead to <i>general artificial intelligence</i>. The current boom means new research discoveries which are ripe for application in unexplored fields, as the private sector looks to beat the competition and the public sector seeks to utilise their massive data sets. Machine learning libraries offer those with programming knowledge the ability to implement what was once only possible to a few hundred academics globally. 

## Deep Learning

Deep Learning is a technique for implementing machine learning, a <i>deep neural network</i> is a popular example of a specific technique within Deep Learning. Deep Neural Networks are a neural network with multiple layers. They have existed since 1943 when Pitts introduced a multi-layered perceptron, however, it wasn’t until 1984 that there was an efficient way to train these complex networks due to hardware constraints. Deep Learning is the area of Machine Learning which has most contributed to its rise to prominence in recent years.

<div style="text-align:center;"><img src="https://securitykiwi.b-cdn.net/images/recurrent-neural-network.png" alt="Image of the board game Go, between AlphaGo and Tang Weixing, 2016." style="width:550px;"/></div>

<p style="text-align: center; font-style: italic;">An example of a Recurrent Neural Network (RNN).</p>

With continued advances in CPUs and more importantly, Graphical Processing Units (GPUs); powerful hardware capable of significant compute power, as well as the price reducing and access to this faster hardware allowing researchers to create new techniques and exploit this computing power. As researchers try to solve more complex problems, they often look to nature to inspire their creations. Some deep neural network algorithms are set up to loosely mimic the structures of specific parts of animal brains, with artificial neurons controlling the frequency of firing based on input and a weight. <i>Convolutional Neural Network</i> (CNN), a type of deep neural network architecture, is designed to mimic the processes of the visual cortex for example. CNNs allows the processing of three-dimensional data, much like the visual cortex.

More recently, researchers Geoffrey Hinton et al, from the University of Toronto and the National University of Singapore, kick-started academic research into machine learning again in 2006, contributing to the beginning of what has become the Third Rise of AI research. The article, ‘<a href="https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf" target="_blank">A fast learning algorithm for deep belief nets</a>’, proposed a new algorithm which made the training of deep belief networks faster and avoided underfitting and overfitting problems which had plagued researchers. Artificial Intelligence research did not stop, of course, it was less fruitful than expected and thus fewer researchers were interested and less money was available for funding. When advances are made, more researchers choose to get into the field to explore the new techniques and typically more money becomes available as it becomes possible to prove the benefits of new techniques.

This course aims to provide you with the resources to take advantage of these new discoveries, academic research and apply techniques for fun and to build your own projects.


<div><i>You can watch <a href="https://www.youtube.com/watch?v=WXuK6gekU1Y" target="_blank">AlphaGo - The Movie</a> on YouTube for free.</i></div>

---

## References

<ol>

<li>Bostrom, N. (2014) <i>Superintelligence: Paths, Dangers, Strategies</i>. Oxford University Press, Oxford. <a href="https://en.wikipedia.org/wiki/Superintelligence:_Paths,_Dangers,_Strategies" target="_blank">https://en.wikipedia.org/wiki/Superintelligence:_Paths,_Dangers,_Strategies</a> </li>
    
<li>DeepMind. (2020) <i>AlphaGo: The Story so Far.</i> <a href="https://deepmind.com/research/case-studies/alphago-the-story-so-far" target="_blank">https://deepmind.com/research/case-studies/alphago-the-story-so-far</a> </li> 

<li>Hinton, G., Osindero, S., and Teh, W. (2006) <i>A fast learning algorithm for deep belief nets.</i> [PDF] <a href="https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf" target="_blank">https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf</a> </li>

<li>Kohs, G et al. (2017) <i>AlphaGo (Movie)</i>. <a href="https://www.alphagomovie.com" target="_blank">https://www.alphagomovie.com</a> </li>

<li>Amodei, D., and Hernandez, D. (2018) <i>Open AI: AI and Compute</i>. <a href="https://openai.com/blog/ai-and-compute/" target="_blank">https://openai.com/blog/ai-and-compute/</a> </li>

<li>Wikipedia. (2020a) <i>Deep Blue (chess computer)</i>. <a href="https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)" target="_blank">https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)</a> </li>
    
<li>Wikipedia. (2020b) <i>AlphaGo.</i> <a href="https://en.wikipedia.org/wiki/AlphaGo" target="_blank">https://en.wikipedia.org/wiki/AlphaGo</a> </li>    
    
<li>Wikipedia. (2020c) <i>Future of Go Summit.</i> <a href="https://en.wikipedia.org/wiki/Future_of_Go_Summit" target="_blank">https://en.wikipedia.org/wiki/Future_of_Go_Summit</a> </li>   

<li>Wikipedia. (2020d) <i>Artificial general intelligence.</i> <a href="https://en.wikipedia.org/wiki/Artificial_general_intelligence" target="_blank">https://en.wikipedia.org/wiki/Artificial_general_intelligence</a> </li> 
    
<li>Wikipedia. (2020e) <i>Knowledge Representation and Reasoning.</i> <a href="https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning" target="_blank">https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning</a> </li>     

</ol>
