# Neural-Gas-MCMC
Sampling a compressed space for human like cognitive performances simulation

# Table of content
* [General Informations](#gen-inf)
* [About the project](#about)
* [References](#ref)

## General Informations
A free project, made for the final exam of "Informatique and Programmation" (Computer Science and Programmation), took in the first semester of 2020
and overseen by Alexandre Bluet, at the university Lyon II.

## About the project
Inspired from the paper of Dasgupta, Schulz & Gershman (2017), this project aimed to reproduce human like performances and bias in reasoning, using 
Markov Chain Monte Carlo (MCMC).

Instead producing myself the space of hypothesis to be sampled, I've add a Neural Gas, which is a Self Organizing Map algorithme.
This clustering algorithm, produces a map of the data using "neurones" to represent clusters.
Neurones are more or less close to each other, depending on the distance between the data that they represent.
By clustering datasets found on the internet, I intended to reproduce this performances, without explicitly setting the hypothesis state.

The idea was that the classical paradigm Encoding-Storage-Retrieval, could be written as a Compression-Decompression coupling, where the cluster space
would represent the memory, and the MCMC the Retrieval Phase.


## References
* Dasgupta, I., Schulz, E., & Gershman, S. J. (2017). Where do hypotheses come from?. Cognitive psychology, 96, 1-25.
