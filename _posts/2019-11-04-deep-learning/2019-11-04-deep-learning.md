Table of Contents:
* [Basics](#basics)
* [Regularization Techniques](#regularization-techniques)
* [Normalization Techniques](#normalization-techniques)
* [Optimization Techniques](#optimization-techniques)
* [Vanishing/Exploding Gradient Problem](#vanishingexploding-gradient)
* [Overfitting](#overfitting)
* [Distributed Learning](#distributed-learning)
* [Convolutional Neural Network](#cnn)
* [Generative Adversarial Network](#gan)
* [Q Learning](#q-learning)

Exam topics:
* Why batch normalization is costly if not normalizing on a mini-batch but the entire dataset?
**Ans:** Forward costly, backward is even more costly.
* If second-order Newton approximation is more accurate, why not use it?
**Ans:** (a) calculate Hessian is costly (b) it's not mathematically correct to do stochastic with Newton, i.e. the stochastic Hessian's inverse is no longer an unbiased estimation! b.c. expectation is NOT close on inverse operation. The only solution is to use a large batch size to approximate it, but that's a tradeoff.
* Sync/Async PyTorch code. See [Distributed Learning](#distributed-learning)

---
## Basics

### PyTorch vs TensorFlow
* Define-by-run vs. Define-and-run
* Dynamic computational graph vs. Static computational graph. e.g. for RNN, PyTorch can detach hidden unit dynamically to allow truncated BPTT.
* Better distributed training vs. Better deployment practice

### Likelihood vs Probability
$$
L(\theta \mid x) = f(x\mid \theta)
$$

Likelihood is defined w.r.t. parameter space, i.e. data $x$
is given, we optimize the parameter $\theta$. While Probability is for given model parameter $\theta$, given the density distribution of $x$.

### Maximum Likelihood Estimation
Given a set of observations $x$ (ground-truth), we need to estimate a model with parameter $\theta$, s.t. the observations are most likely to happen in that way.
Two classic [examples](https://zhuanlan.zhihu.com/p/36824006), coins and beans. Out of 100 coins we get 52 positive sides and 48 negative sides, what the optimal model $\theta$ (in this case, the probability of a coin being positive) s.t. the experiment is most likely to happen? We have the total probability $p=\theta^{52}(1-\theta)^{48}$. $p$ has to have the max likelihood to happen (otherwise the result won't end up like this, in a parallel universe), so we take the extremum by taking derivative of $p$. And we take $log$ before doing that to convert $\prod$ to $\Sigma$.

### i.i.d.
independent & identically distributed

### Two Fundamental Theorems in Probability Theory
* Law of Large Numbers (LLN): the result of performing the same experiment a large number of times will approach the expected value. More trials, get closer.
* Central Limit Theorem (CLT): the result distribution will approach a normal distribution when a large number of trails are done

### Softmax
$$
y_{i} = \frac{e^{x_i}}{\Sigma_{k=1}^{K}e^{x_k}}, \forall\ i\in \{1,2,...,K\}
$$
$x$ is K-dimension input, $y$ is K-dimension output.

Advantages of softmax over other operation (such as argmax):
* Normalize the data by giving probability distribution in range $[0,1]$ (other operations have this too)
* **Differentiable**. The softmax gives at least a minimal amount of probability to **all** elements in the output vector, and so is nicely differentiable, hence the term "soft" in softmax. Conversely, "hardmax" (i.e. argmax) will only preserve the max element, it's not differentiable
* Sensitive to magnitude. It's a **nonlinear** exponential transformation s.t. high scores will become more prominent after.
* Simple computation when used together with Cross-Entropy error.

### Negative Log Likelihood (or Cross-Entropy)
Why "log"? convert the maximum likelihood from product $\prod$ into sum $\Sigma$.

Why "negative"? Optimizers usually minimize an objective function, so maximize $(f)$ == minimize $(-f)$

### Activation functions
* sigmoid: "S" shape, $\frac{e^x}{e^x+1}$
* ReLU (Rectified Linear Unit, suppress negative space): max(x, 0)
* tanh

### Problem Type and Loss Functions
See [this](https://zhuanlan.zhihu.com/p/35709485) and [this](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/)

#### Classification problem
* Task is to predict a discrete class label.
* Output is a one-hot vector representing the result for a multi-label classification.
* To measure the error between two one-hot vectors (prediction and label), cross-entropy error is used. $L = -\Sigma y_i \cdot log(p_i)$, $y_i$ is the one-hot boolean value (0 if doesn't match, 1 if match, also called indicator function), $p_i$ is the probability. Before taking the log, it's $\prod p_i^{y_i}$. Understand from the coin problem. C-E error measures the deviation of the current data distribution from the target distribution (truth).

#### Regression problem
* Task is to predict a continuous quantity.
* Output is a scalar value representing the predicted quantity.
* To measure the error between two quantities (prediction and observation), Mean Squared Error (MSE, or root RMSE), $L = \frac{1}{M} \Sigma_{i=1}^{M} \|\hat{y}_i - y_i\|^{2}$ where $M$ is mini-batch size.

#### Validation/Testing
* For classification problem, classification error (i.e. accuracy) is used, $\frac{\text{No. of correct items}}{\text{No. of total items}}$
* For regression problem, MSE or RMSE can just be used.

---
## Regularization Techniques
Regularization is used to **reduce network complexity** and **avoid overfitting**. A common technique is to add penalty term (or regularization term) in the loss function, such as $L_1, L_2$ penalty. Other techniques such as dropout (or even batch normalization) can also be considered as a form of regularization.

### $L_2$ penalty
$L_2$ penalty is the sum of squared weights. Large weights will have higher loss, smaller weights are rewarded:
$$L_2: L(\theta) = \frac{1}{N}\Sigma_{i=1}^{N}\rho(f(x^i;\theta), y^i) + \lambda {\lVert \theta \rVert}_2^2$$

The first term is the loss, the second term is the penalty term to prevent parameter $\theta$ from being too large. $\lambda$ is used to adjust the balance. During backpropagation, since ${\lVert \theta \rVert}_2^2 = \Sigma (W_i)^2\rightarrow \frac{\partial {\lVert \theta \rVert}_2^2}{\partial W_i}=2W_i\rightarrow \nabla_{W_i}L(\theta)=\frac{\partial \rho}{\partial W_i} + \frac{\partial {\lVert \theta \rVert}_2^2}{\partial W_i}$, i.e. the gradient values increase s.t. the weight values is decreased more than normal GD. That's why this is called "weight decay". The weights will be updated depending on its magnitude. This restricts the weights from being very large.

A obvious question would be: why large parameter $\theta$ $\Leftrightarrow$ model overfitting? One feature of overfitting is that the parameter weights have large values (an intuitive 2D example explained [here](https://www.cnblogs.com/alexanderkun/p/6922428.html), higher weights-->higher model complexity because the model needs to use a complex curvature to fit the data thus usually needs large weights).

### Dropout
Dropout can also help as regularization by randomly deactivating some hidden units. This reduces the model complexity. Hinton thinks overfitting comes from co-dependency between hidden units, therefore this random deactivation breaks the co-dependency.

Dropout will generate a random set of Bernoulli masks $R=\{R^1,...,R^L\}$ for all layers. Each $R$ could represent a different model architecture. Since the deactivation is Bernoulli random, there are exponentially large number of different dropout models even for one data sample, which is not practical for training. Instead, we pick **one mask $R$ for one data sample**. So in mini-batch SGD, $M$ data samples actually have $M$ different sets of $R$s

Now the question is, if we used different models for each data sample during training, what should be THE model we used for testing? Two approaches:
* Heuristic algorithm (flawed but effective): scale the layer output by probability $p$ to simulate the dropout layer, which is called "weight scaling". Think in this way: if $p=0.7$, 30% of the hidden units were dropped during training, now if we run inference with the current model (w/o dropout), the output will be larger than we want (larger by a factor of $p$); then we scale each layer output will a factor of $p$, it will be good.

> Note: "heuristic" means a plausible method but may not be optimal/most precise

* Monte Carlo simulation (more mathematically correct): for each data sample, repeat the training for $N$ times and average. Since each training will generate a random mask $R$, this simulation the model behavior when $N$ is sufficiently large. But can be slow.

---
## Normalization Techniques
Normalization is used to standardize the input so as to **accelerate the convergence** of the optimization problem (i.e. more efficient learning). Generally, normalization has the following benefits:
* It can regularizes the model
* It can generalizes the network behavior on different input
* It can enable higher learning rate
* It can address the vanishing/exploding gradient problem

### Xavier Parameter Initialization
The idea is to have the input and output has the same variance. Randomly initialize the weights $W$, and divide by $\sqrt{d_{l-1}}$ (d is the number of hidden unit of the input layer). Why this can work?

See proof [here](https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization):
For linear neuron
$$Y = W_1X_1 + W_2X_2 + \dotsb + W_n X_n$$
We can have
$$\text{Var}(Y) = \text{Var}(W_1X_1 + W_2X_2 + \dotsb + W_n X_n) = n\text{Var}(W_i)\text{Var}(X_i)\\$$
If we want $\text{Var}(Y)=\text{Var}(X_i)$, we should let $n\text{Var}(W_i)=1\rightarrow \text{Var}(W_i)=
\frac{1}{n}$
Recall some basics that
$$Var(aX+b) = a^2Var(X)$$
So we should initialize as
$$W_i=\frac{\text{Random Variable}}{\sqrt{n}}$$

### Batch Normalization
Batch normalization address the vanishing gradient problem (caused by saturation) and reduce the internal covariate shift between layers. Now every layer is normalized to have the same expectation and variance before activation. In training, as the parameters of the preceding layers change, the distribution of inputs to the current layer changes accordingly, such that the current layer needs to constantly readjust to new distributions. This problem is especially severe for deep networks, because small changes in shallower hidden layers will be amplified as they propagate within the network, resulting in significant shift in deeper hidden layers.

Take a mini-batch, normalize based on mini-batch mean and variance --> scale and shift (i.e. linear transform). During BP, the weight will be multiplied during the parameter updating.  BN rescales the distribution to N~(0,1) for easier learning, such that the weights are mostly in unsaturated range.

By definition
$$H_i^l=\sigma(\frac{Z_i^l-E_X[Z_i^l]}{\sqrt{Var_X[Z_i^l]}})$$
The $E_X[Z_i^l]$ and $Var_X[Z_i^l]$ should be the statistics for the entire dataset, which means, to normalize one hidden unit in a layer, we need to cache the values for all data samples. This is problematic and the reasons are twofold:
* Forward step: each hidden unit needs a cache of length N (N is size of the training dataset). Memory constraint.
* Backward step: this is even WORSE! When backpropagate and update the weights $W^l$, the partial derivative will again rely on the entire dataset. The calculation can be very slow.

So, we need to approximate the $E_X[Z_i^l]$ and $Var_X[Z_i^l]$ i.i.d. with only a mini-batch of data, i.e. previously we have
$$E_X[Z_i^l]=\frac{1}{N} \Sigma_{j=1}^{N}Z_i^{l,j}$$
now we have

$$E_X[Z_i^l]=\frac{1}{M} \Sigma_{j=1}^{M}Z_i^{l,j}$$
where $Z_i^{l,j}:= Z_i^l$ for jth data sample.

Note that for batch normalization to work, the mini-batch size should be too small since the mean and variance statistics should approximate the whole dataset.

### Layer Normalization
Batch normalization only works with mini-batch SGD (and is expensive), while layer normalization is a per data sample method (cheap). They typically perform the same but layer normalization is easier to implement and less computationally intensive.

Layer normalization will not learn the distribution of the dataset, but just normalize the layer with its hidden units.
$$E_X[Z^l]=\frac{1}{H} \Sigma_{i=1}^{H}Z_i^l$$

For each data sample, the layer normalization will adjust its output before activation. This does not rely on other data samples.

These two normalization techniques are working on different statistics. For comparison:
| | Batch Normalization | Layer Normalization |
---|---|---
Concept | batch samples, for one unit | single sample, across all units
Cost | expensive | cheap
Type | approximate normalization | exact normalization
Mini-batch requirement | yes | can work for batch size=1 (e.g. online training)

In addition, data augmentation will involve some normalization as well.

---
## Optimization Techniques
* RMSprop
* ADAM

## Vanishing/Exploding Gradient
Gradient values could become extremely small/large during training, why?
* **Back propagation essence**. Chain rule in back propagation leads to different learning speeds (i.e. parameter updating speed) at different layers. Usually the layers close to input get updated very slowly, and the ones close to output get good updates. This is inherited behavior of BP.
* Choice of **activation functions**. If the derivative of activation function is less than 1, the gradients will be vanishing exponentially; otherwise, exploding exponentially. Magnitude of derivatives of sigmoid function caps at 0.25 (vanishing), tanh caps at 1.0 (vanishing), ReLU is a little better that for positive input is exactly 1.0 (no effect) but for negative input is 0 (still vanishing).
* **Saturation** of the activation functions. This is also related to activation function property. For sigmoid and tanh, the derivative plot is supported at 0, so when the input is very large/small, the derivatives will be close to 0, which leads to less or no learning. ReLU doesn't have this problem.
* Weights magnitude. Similar to the derivative magnitude of activation functions, weight magnitude also has the same effect. If the weight values are very small, it's hard to efficiently propagate the gradient backward to lower layers. Batch normalization rescales the weights so the learning speed at different layers can be synchronized.

How to solve?
* Choose appropriate activation functions.
* Batch Normalization.
* ResNet. Use shortcut to backpropagate "1+".
* LSTM.

## Overfitting
Overfitting means good model performance on training dataset but poor on validation/test dataset. Solutions include:
* **L1/L2 Regularization** [modify loss function]. Restrict weight values.
* **Dropout** [modify model architecture]. Reduce model complexity.
* **Data augmentation** [modify dataset]. Enrich data diversity.
* **Transfer learning** [modify dataset]. Apply a more generalized model learned on a larger dataset.
* **Ensemble learning** [modify model]. Merge different models together, each trained on randomly initialized parameters & random sequence of data together. Have generalized/flexible performance.
* **Early stopping** [modify training process]. Divide the traning dataset into a real training set and a validation set. Stop when the loss on validation set starts to increase. More reasonable approaches can define a generalization loss as the stopping criteria.

## Distributed Learning
* Data parallel: each node has an entire copy of the model, but trains on different data
* Model parallel: each node has only part of the model. Used for very large model that can't fit into one node. Less likely in practice.

### Synchronous SGD
For each step, one node will work on one part of the data (and this data is done in mini-batch on each node), do forward and backward training, and report the gradient. A barrier is set to wait for all nodes to complete the current step. Then the gradients are averaged and the averaged gradient is used to update the model copy on each node. So, during the entire training, each node has exactly the same copy of the model.

Analysis:
* The effective mini-batch size is N x M where N is number of nodes, M is mini-batch size. So, the LR can be higher and the convergence is faster.
* Communication cost among nodes increase as the model becomes larger, so the parallel efficiency can't achieve 100%. The communication cost is O(logN). Think a binary tree structure where the root is any arbitrary node. We first concentrate everything to the root, calculate the averaged gradient, and broadcast the gradient value. From the binary tree structure we know it has height O(logN), and two times of propagation/broadcast is O(logN).
* Barrier synchronization will be constrained by the slowest node.

Code:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

model = ResNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(num_epochs):
    # Update learning rate
    scheduler.step()

    # Train
    model.train()
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        # Calculate gradient for each node
        optimizer.zero_grad() # reset gradient
        results = model(images) # forward
        loss = criterion(results, labels) # compute loss
        loss.backward() # calculate gradients

        # Synchronize gradients among nodes
        for param in model.parameters():
            tensor = param.grad.data.cpu()
            dist.all_reduce(tensor, op=dist.reduce_op.SUM)
            tensor /= float(size) # average the reduced (i.e. summed) gradients
            param.grad.data = tensor.cuda()

        # Update weight parameters for each node
        optimizer.step()
```

### Asynchronous SGD
Asynchronous will have one node as the parameter server to collect gradient from worker servers and send back the updated model.

Note that the worker server send the gradients `param.grad.data` to parameter server, but should receive a model `param.data` back. The asynchronous is done by non-blocking of `send` and `recv`.

Analysis:
* Some nodes are allowed to go significantly faster than others, but every time a node get model from the parameter server, it will be the latest model
* Communication cost grows linearly with the number of workers N. There is only one parameter server and no broadcast exists (b.c. its async), the cost is O(N)
* Asynchronous SGD is biased, and bias grows with N

Code:
```python
for epoch in range(num_epochs):
    if rank == 0: # parameter server
        # receive gradients from worker
        optimizer.zero_grad()
        for param in model.parameters():
            tensor = param.grad.data.cpu()
            request = dist.irecv(tensor, src=node)
            request.wait()

        # update model
        optimizer.step()    

        # send back the latest model
        for param in model.parameters():
            tensor = param.data.cpu()
            request = dist.isend(tensor, dst=node)
            request.wait()

    else: # worker servers (symmetric to parameter server)
        # training
        optimizer.zero_grad()
        results = model(images)
        loss = criterion(results, labels)
        loss.backward()

        # send gradients
        for param in model.parameters():
            tensor = param.grad.data.cpu()
            request = dist.isend(tensor, dst=0)
            request.wait()

        # receive updated model
        for param in model.parameters():
            tensor = param.data.cpu()
            request = dist.irecv(tensor, src=0)
            param.data = tensor.cuda()
```

## CNN
Why CNN can work with image recognition?
Magic of convolution operation:
* Invariant to translation/shift
* Abstract local features
* Shared weights. For any object in the image, the same filter (i.e. weights) is applied. As compares to fully connected network which applies a different weight to each pixel, this is more abstract and reduce the parameters needed. The "filter" is called "shared weight".

### ResNet
On how to handle the channel mismatch (after convolution, the channels will increase, so F(x) and x have different number of channels) and spatial mismatch (after convolution, F(x) may or may not be downsampled, the image dimension can mismatch) when we add back the identity x:
* **channel mismatch**: specify `out_channels` in conv1x1 so that the input image will be replicated to the same number of channels as the convolutioned output
* **spatial mismatch**: specify `stride` in conv1x1 same as the `stride` in conv3x3

## GAN
[intro](https://zhuanlan.zhihu.com/p/25071913) and [details](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html#use-wasserstein-distance-as-gan-loss-function)

## Q Learning
The gist in Q learning is that the value function at the current step = current reward + cumulate all future rewards, and what we want is to maximize this value function. Bellman equation ensures that the optimal solution exists and is unique.

### Tabular Q-Learning
Store action-quality value Q(x,a) in a table, and train and keep updating the table (one at a step), the table is expected to converge the optimal solution.

Compare tabular Q and deep Q:
* Tabular Q is proved to converge
* if x & a space are large or even continuous, it will cost a lot of memory or even impossible to learn for tabular Q
* tabular Q is extremely slow. For one step, tabular only update one entry, but in deep Q, it update the model $\theta$ which affects all $(x_k,a_k)$, much more efficient


How to select Action $A_t$?
