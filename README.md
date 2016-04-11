(Adapted from [Edwin Chen](https://github.com/echen/restricted-boltzmann-machines) and  the [Wikipedia page on RBMs](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine))

# How to Use

First, initialize an RBM with the desired number of visible and hidden units.

    rbm = RBM(num_visible = 6, num_hidden = 2)
    
Next, train the machine:

    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]) # A 6x6 matrix where each row is a training example and each column is a visible unit.
    r.train(training_data, max_epochs = 5000) # Don't run the training for more than 5000 epochs.
    
Finally, run wild!

    # Given a new set of visible units, we can see what hidden units are activated.
    visible_data = np.array([[0,0,0,1,1,0]]) # A matrix with a single row that contains the states of the visible units. (We can also include more rows.)
    r.run_visible(visible_data) # See what hidden units are activated.

    # Given a set of hidden units, we can see what visible units are activated.
    hidden_data = np.array([[1,0]]) # A matrix with a single row that contains the states of the hidden units. (We can also include more rows.)
    r.run_hidden(hidden_data) # See what visible units are activated.
    
    # We can let the network run freely (aka, daydream).
    r.daydream(100) # Daydream for 100 steps on a single initialization.

# Introduction

Suppose you are modelling and artificial intelligence agent which competes with different types of enemies, each one implementing different stochastical strategies. We want the agent to detect the different strategies developed by its opponents in order to adapt its behaviour to them, without necessarily having previous information about which strategy is being implemented except the patterns of movement displayed by each opponent.

Restricted Boltzmann Machines are **stochastic neural networks** (*neural network* meaning we have neuron-like units whose binary activations depend on the neighbors they're connected to; *stochastic* meaning these activations have a probabilistic element) consisting of:
* One layer of **visible units** (users' movie preferences whose states we know and set);
* One layer of **hidden units** (the latent factors we try to learn); and 

Furthermore, each visible unit is connected to all the hidden units (this connection is undirected, so each hidden unit is also connected to all the visible units). To make learning easier, we restrict the network so that no visible unit is connected to any other visible unit and no hidden unit is connected to any other hidden unit. Thus, the two layers of hidden and visible units are connected through a matrix of weights W = (w_{i,j}) (size m×n) associated with the connection between hidden unit h_j and visible unit v_i, as well as bias weights (offsets) a_i for the visible units and b_j for the hidden units. Given these, the energy of a configuration (pair of boolean vectors) (v,h) is defined as

![RBM-energy](https://upload.wikimedia.org/math/b/e/0/be0bf0da1b4d822d3852c56a504b4f72.png)

This energy function is analogous to that of a Hopfield network. As in general Boltzmann machines, probability distributions over hidden and/or visible vectors are defined in terms of the energy function:

![RBM-probability](https://upload.wikimedia.org/math/f/3/b/f3b3eef66258cece4d18aa06344eb569.png)

where Z is a partition function defined as the sum of e^{-E(v,h)} over all possible configurations.

Since the RBM has the shape of a bipartite graph, with no intra-layer connections, the hidden unit activations are mutually independent given the visible unit activations and conversely, the visible unit activations are mutually independent given the hidden unit activations.[7] That is, for m visible units and n hidden units, the conditional probability of a configuration of the visible units v, given a configuration of the hidden units h, is

![RBM-individual-probability](https://upload.wikimedia.org/math/a/8/4/a8437d5b9b8dca11a35cd9bb1bc6cefc.png) 

and 

![RBM-individual-probability](https://upload.wikimedia.org/math/0/b/3/0b3b7ace86df7502dcb6b1eba976b67c.png)

For example, suppose we are modelling a videogame agent in a world with different kinds of characters (e.g. enemies, monsters, civilians...) which present different movement patterns depending on their strategies to attack or avoid enemy attacks, e.g. random movement or random oscillatory movement. Imaging we want to model an agent which is able to display different responses deppending on the strategy implemented by the characters it encounters. However, as the different patterns have a compontent of random behaviour, they are not trivial to differentiate directly.

![Agent detection example](https://github.com/MiguelAguilera/restricted-boltzmann-machines/blob/master/example-agents.png)

In order to detect different patterns of movement and infer which strategies are behind each pattern we will implement a Restricted Boltzmann Machine, in wich we will record a set of movements m(t), for t=1,...,L, where m(t)=1 represents the agent moving up and m(t)=0 represents the agent moving down.

For simplicity in the learnign phase, we will ignore the offsets of visible and hidden neurons a_i and b_j and substitute them for an extra *bias unit*, whose state is always on and it will be connected to all hidden and visible units (having and equivalent effect to adding the offsets):

![RBM Example](https://github.com/MiguelAguilera/restricted-boltzmann-machines/blob/master/RBM-example.png)

# State Activation

Restricted Boltzmann Machines, and neural networks in general, work by updating the states of some neurons given the states of others, so let's talk about how the states of individual units change. Assuming we know the connection weights in our RBM (we'll explain how to learn these below), to update the state of unit $i$:

* Compute the **activation energy** $a_i = \sum_j w_{ij} x_j$ of unit $i$, where the sum runs over all units $j$ that unit $i$ is connected to, $w_{ij}$ is the weight of the connection between $i$ and $j$, and $x_j$ is the 0 or 1 state of unit $j$. In other words, all of unit $i$'s neighbors send it a message, and we compute the sum of all these messages.
* Let $p_i = \sigma(a_i)$, where $\sigma(x) = 1/(1 + exp(-x))$ is the logistic function. Note that $p_i$ is close to 1 for large positive activation energies, and $p_i$ is close to 0 for negative activation energies.
* We then turn unit $i$ on with probability $p_i$, and turn it off with probability $1 - p_i$.

For example, let's suppose our two hidden units really do correspond to different strategies for the opponents movement.

* If opponent *A* has a particular pattern of movement (e.g. m = [0,1,0,1,1,0,1,1,1,0,0,...]), we could provide those values to the visible units, and then ask our RBM which of the hidden units activate. Similar patterns will activate the same hidden units, whereas different patterns will activate different hidden units. Thus, the RBM allows us to *generate* models of people in the messy, real world.
* Conversely, we could set the RBM with a particular configuration of the hidden units and ask which visible units turn on, generating possible patterns that *match the model* learned by the RBM.

# Training

So how do we learn the connection weights in our network? Suppose we have a bunch of training examples, where each training example is a binary vector with L elements corresponding to a opponent pattern of movement m(t). Learning the different patterns means to maximize the product of probabilities assigned to some training set V (here each v corresponds to a particular occurrence of a m(t) pattern):

![training-function](https://upload.wikimedia.org/math/f/c/c/fccaba865768c42939126335d12032c6.png)

The algorithm most often used to train RBMs, that is, to optimize the weight vector W, is the contrastive divergence (CD) algorithm due to Hinton, originally developed to train PoE (product of experts) models.[13][14] The algorithm performs Gibbs sampling and is used inside a gradient descent procedure (similar to the way backpropagation is used inside such a procedure when training feedforward neural nets) to compute weight update.

The basic, single-step contrastive divergence (CD-1) procedure for a single sample can be summarized as follows:

    - Take a training sample v, compute the probabilities of the hidden units and sample a hidden activation vector h from this probability distribution.
    - Compute the outer product of v and h and call this the positive gradient.
    - From h, sample a reconstruction v' of the visible units, then resample the hidden activations h' from this. (Gibbs sampling step)
    - Compute the outer product of v' and h' and call this the negative gradient.
    - Let the weight update to w_{i,j} be the positive gradient minus the negative gradient, times some learning rate: \Delta w_{i,j} = \epsilon (vh - v'h'}).

Continue until the network converges (i.e., the error between the training examples and their reconstructions falls below some threshold) or we reach some maximum number of epochs.

Why does this update rule make sense? Note that 

* In the first phase, the positive gradient measures the association between the v and h unit that we *want* the network to learn from our training examples;
* In the "reconstruction" phase, where the RBM generates the states of visible units based on its hypotheses about the hidden units alone, the negative gradient measures the association that the network *itself* generates (or "daydreams" about) when no units are fixed to training data. 

So by adding positive - negative graidents to each edge weight, we're helping the network's daydreams better match the reality of our training examples.

(You may hear this update rule called **contrastive divergence**, which is basically a funky term for "approximate gradient descent".)

# Examples

We can use the code in this repository (forked from [Edwin Chen](https://github.com/echen/restricted-boltzmann-machines)) in Python (the code is heavily commented, so take a look if you're still a little fuzzy on how everything works) to understand how RBMs work through an example.

First, using the functions in generate_patterns.py we can generate different patterns of movement: random movement (just binary white noise) and random oscillatory functions (in which p(m(t)=1) = sin(2*pi*t/T), where T is the period of the oscillations). We choose to have 3 types of movement patterns we want to differentiate, 1 random and 2 oscillatory with periods (T_A=5 and T_B=9).

We generate 200 samples for each kind of movement, and use sequences of m(t) with lenght L=30. We train the RBM introducing m(t) into the visible units.

After training, we generate 8 new samples of each type of movement and feed them into the RBM, obtaining the following values for the hidden units:

Sinusoidal A
 [ 1.  0.]
 [ 1.  0.]
 [ 1.  0.]
 [ 1.  0.]
 [ 1.  0.]
 [ 1.  0.]
 [ 1.  0.]
 [ 1.  0.]
Sinusoidal B
 [ 0.  1.]
 [ 0.  1.]
 [ 0.  1.]
 [ 0.  1.]
 [ 0.  1.]
 [ 0.  1.]
 [ 0.  1.]
 [ 0.  1.]
Random
 [ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]
 [ 0.  0.]
 [ 1.  0.]
 [ 0.  0.]

We see that the combination [1,0] is activated for oscillatory patterns with T_A=5, the combiantion [0,1] for oscillatory patterns with T_B=9, and the combination [0,0] for random patterns. Note that one pattern of random movement is misclassified as oscillatory movement. This is because that particular pattern happens to be very similar to an oscillatory pattern (m=[0 0 1 0 0 1 1 1 1 0 1 1 0 0 1 0 1 0 0 0 1 1 0 1 1 1 1 1 0 0]). Having a larger value of L and generating more sequences in the trainig phase should minimize the errors in pattern detection.


# Modifications

I tried to keep the connection-learning algorithm I described above pretty simple, so here are some modifications that often appear in practice:

* Above, $Negative(e_{ij})$ was determined by taking the product of the $i$th and $j$th units after reconstructing the visible units *once* and then updating the hidden units again. We could also take the product after some larger number of reconstructions (i.e., repeat updating the visible units, then the hidden units, then the visible units again, and so on); this is slower, but describes the network's daydreams more accurately.
* Instead of using $Positive(e_{ij})=x_i * x_j$, where $x_i$ and $x_j$ are binary 0 or 1 *states*, we could also let $x_i$ and/or $x_j$ be activation *probabilities*. Similarly for $Negative(e_{ij})$.
* We could penalize larger edge weights, in order to get a sparser or more regularized model.
* When updating edge weights, we could use a momentum factor: we would add to each edge a weighted sum of the current step as described above (i.e., $L * (Positive(e_{ij}) - Negative(e_{ij})$) and the step previously taken.
* Instead of using only one training example in each epoch, we could use *batches* of examples in each epoch, and only update the network's weights after passing through all the examples in the batch. This can speed up the learning by taking advantage of fast matrix-multiplication algorithms.

# Further

If you're interested in learning more about Restricted Boltzmann Machines, here are some good links.

* [A Practical guide to training restricted Boltzmann machines](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf), by Geoffrey Hinton.
* A talk by Andrew Ng on [Unsupervised Feature Learning and Deep Learning](http://www.youtube.com/watch?v=ZmNOAtZIgIk).
* [Restricted Boltzmann Machines for Collaborative Filtering](http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf). I found this paper hard to read, but it's an interesting application to the Netflix Prize.
* [Geometry of the Restricted Boltzmann Machine](http://arxiv.org/abs/0908.4425). A very readable introduction to RBMs, "starting with the observation that its Zariski closure is a Hadamard power of the first secant variety of the Segre variety of projective lines". (I kid, I kid.)
