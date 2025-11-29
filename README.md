# Dendritic NN Block for Edge Impulse

This repository implements a neural network block for Edge Impulse which leverages dendritic optimization. To see details on how to compile and push this block to Edge Impulse foloow the instructions from the [original repository](https://github.com/edgeimpulse/example-custom-ml-block-pytorch/tree/master). This block was created for the 2025 Edge Impulse Hackathon.  Our submission video going over the details of this project can be found [FILL IN LINK TO VIDEO HERE]().

## What is Dendritic Optimization?

The original artificial neuron was invented in 1943 based on neuroscience that from the 1860s.  Since then backpropagation was introduced and there has been significant advancement in hardware, optimizers, data curaiton, and architectures while the core building block has remaiend fundamentally the same.  Interestingly, for 70 of the last 80 years, neuroscience contineud to support this original design.  However moden neuroscience now understands that the perceptron missing the most critically important peice of biological intelligence, the decision making of the dendrites within the neuron itself.  Dendritic optimization is a way to leverage these ideas to augment the neurons of artificial neural networks with dendrite nodes, empowering data scientists to acheive smarter, smaller, and cheaper models on the same datasets.  For futher details about this research a selection of papers can be found [here](https://github.com/PerforatedAI/PerforatedAI/tree/main/Papers).

## This Project

This project first explored the improvements dendritic optimization could acheive on the model in the [keyword spotting tutorial](https://docs.edgeimpulse.com/tutorials/end-to-end/keyword-spotting), and then created a public Edge Impulse block to enable anyone to leverage this capability on their own Edge Impulse projects. 

## Our Experiments

For details on our experiments please view the [W&B report](https://wandb.ai/perforated-ai/Dendritic%20Edge%20Impulse%20Audio%20-%20Combo/reports/Edge-Impulse-Keyword-Spotting--VmlldzoxNTIxNjE5Ng?accessToken=3lm4jm5f9npsu45vs180ybo6150ed4gnhos9rrkk6seqb4bmf458me28seynu0xb) of the 800 trials we ran sweeping over hyperparameters for this application.

## This repository

This repository replaces the original PyTorch script from the original PyTorch example block with our custom script.  It updates the hyperparameter settings to enable users to experiment with all of the hyperparameters we swept over.  It also compiles the final dendritic models in ONNX format to be used ine exactly the same way as the old block.  This is a plug and play Impulse Block allowing users to use dendritic optimization on any Edge Impulse project which uses audio data.  As an open source project it enables users to make required adjustments to work with additional data formants.  Additionally, by working together with the Edge Impulse team this block shows a starting point to extend the default Edge Impulse NN classifier block with dendritic optimization, empowering all Edge Impulse users to acheive improved outcomes on any proejct, by clocking a single checkbox.
