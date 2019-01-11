# RBM
Restricted Boltzmann Machines as Keras Layer

## Introduction
Boltzmann machines are unsupervised, energy-based probabilistic models (or generators). This means that they associate an energy for each configuration of the variables that one wants to model. <br/>
Intuitively, learning in these models corresponds to associating more likely configurations to lower energy states. <br/>
In these states there are units that we call visible, denoted by <i><b>v</i></b>, and hidden units denoted by <i><b>h</i></b>.  <br/>
A general model of Boltzmnn Machine is shown below.
<br/>
<img src="https://matheusfacure.github.io/img/tutorial/rbm/Boltzmann_Machine.png">
<br/>
In fact, Boltzmann machines are so complicated that they have yet to prove practical utility. So we will have to restrict them in some way. Restricted Boltzmann Machines fulfill this role. <br/>
They are Boltzmann Machines on the condition that there are no direct connections between the visible units nor between the hidden ones.
<br/>
<img src="https://matheusfacure.github.io/img/tutorial/rbm/rbm.png">
<br/>
## Code
The code was impplemented using Python 3, and had the follow dependences:<br/>
<ul>
  <li>Tensorflow</li>
  <li>Keras</li>
  <li>Numpy</li>
</ul>

## Results
One way to evaluate the RBM is visually, by showing the <i><b>W</i></b> parameters as images. <br/>
If the training is successful, the weights should contain useful information for modeling the MNIST base digits.
<br/>
<img src="https://matheusfacure.github.io/img/tutorial/rbm/resultRBM.png">
<br/>
Above, not all weights are easily interpreted. Note how the weights highlighted in red contain black lines at the top or bottom. 
Black pixels mean negative values in w and can be interpreted as a filter that prevents the passage of information. 
These black lines then capture information that the digits do not exceed line height. 
Thus, the MBR places little probability in visible states with positive pixels in places higher or lower than those lines. 
The filter highlighted in yellow is probably useful for detecting sloping traces on the right, such as the "7".
