# Can lies be faked?

This repository was created for my BSc Thesis (or Monograph, or Final Year Project, or "Trabalho de Conclusão de Curso", depending on how you might call it): "Video-based Deception Detection with Deep Learning". You can read the whole Thesis [here](https://github.com/MahatKC/SlowFastDeceptionDetection/blob/master/Video-Based%20Deception%20Detection%20with%20Deep%20Learning.pdf) (in Portuguese).

However, the project was developed further and a paper has been written based on the Thesis: ["Can lies be faked? Comparing low-stakes and high-stakes deception video datasets from a Machine Learning perspective"](https://www.sciencedirect.com/science/article/pii/S0957417424005505), published in Expert Systems with Applications.

This repository is organized as follows: **Code** folder has the code used to run experiments and has instructions on how to organize the datasets so experiments run properly; **Results** folder has the CSV files with the results of each of the runs performed during the hyperparameter search phase, as well as TXT files with detailed results from 5-Fold testing.

----

## Paper Abstract

Despite the great impact of lies in human societies and a meager 54% human accuracy for Deception Detection (DD), Machine Learning systems that perform automated DD are still not viable for proper application in real-life settings due to data scarcity. Few publicly available DD datasets exist and the creation of new datasets is hindered by the conceptual distinction between low-stakes and high-stakes lies. Theoretically, the two kinds of lies are so distinct that a dataset of one kind could not be used for applications for the other kind. Even though it is easier to acquire data on low-stakes deception since it can be simulated (faked) in controlled settings, these lies do not hold the same significance or depth as genuine high-stakes lies, which are much harder to obtain and hold the practical interest of automated DD systems. To investigate whether this distinction holds true from a practical perspective, we design several experiments comparing a high-stakes DD dataset and a low-stakes DD dataset evaluating their results on a Deep Learning classifier working exclusively from video data. In our experiments, a network trained in low-stakes lies had better accuracy classifying high-stakes deception than low-stakes, although using low-stakes lies as an augmentation strategy for the high-stakes dataset decreased its accuracy.

----

## Thesis Abstract

**Deception Detection is a task in which humans show an outstanding difficulty, reaching an accuracy of only 54%** according to the literature. Despite that, this task becomes significantly relevant in contexts such as trials, interviews and criminal investigations, in which the impact of mistakenly classifying a discourse as deceptive or truthful may be catastrophic. In this sense, lies and their identification have spiked the interest of researchers for centuries, with the creation of devices destined to aid in deception detection and, more recently, with the development of Machine Learning and Deep Learning systems capable of properly classifying them. Based on these systems, this work analyzes literature on the subject and implements a **Deep Neural Network based on the SlowFast architecture using the Real-Life Trial dataset reaching an accuracy of 66.36%**. Special consideration is given to ethical issues and limitations concerning the use of Machine Learning and Deep Learning systems that perform deception detection, with the **recommendation that these systems not be used in real-life situations given the existing limitations for their satisfactory development**.

----
