# Entropy-Gradient based Odd-Even classifier
This project implements an odd-even binary classifier using entropy-gradient updates rather than the usual gradient descent algorithm. A custom training loop is also used here instead of the built-in optimizer of TensorFlow. So, the whole project is implemented mainly using the NumPy library.

## Model Architecture

<p>
  <img src="https://github.com/user-attachments/assets/fcca29ad-f694-4b47-b428-d03e049415ad" alt="Model Architecture" width="400">
</p>

This model has a single perceptron implementation where each input to the output node is a dual-weight structure. Instead of the traditional single-weight input, the input from each node to the output node is a combined parameter of two different weight components. This better allows for the intended entropy-gradient-based updation.

### Knowledge calculation
The knowledge calculation for this is done using the formula:

$$
z_k = \sum_{j=1}^{n} (w_{1,j} + G_{1,j}) x^{(0)}_j + b^{(0)}
$$

Where $\( w \)$ and $\( G \)$ are the combined input parameters and $\( b \)$ is the bias value.

### Activation Function
The activation function used here is sigmoid, which is calculated by:

$$
D_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}
$$

Sigmoid function is used since it is suitable for binary classification.

### Entropy-Gradient
The gradient of the entropy functional with respect to $\( z \)$ at step $\( k \)$ is given by:  

$$
\frac{\partial H(z)}{\partial z} \Bigg|_k = -\frac{1}{\ln 2} z_k D_k (1 - D_k)
$$

This gradient is used to update the model parameters. Unlike traditional loss-based optimization, this entropy-based update rule ensures that the training process is guided by knowledge-based principles rather than traditional error minimization.

### Knowledge Constraint
In addition to the entropy calculation, to ensure stable learning, the change in knowledge $\( z \)$ between consecutive iterations is constrained as:  

$$  
z_{i+1} - z_i < \delta  
$$  

This prevents excessive fluctuations in $\( z \)$, ensuring smooth updates and numerical stability. Since the model optimizes parameters using entropy gradients instead of traditional loss functions, limiting $\( z \)$ helps maintain controlled learning dynamics. If an update exceeds $\( \delta \)$, it is scaled down proportionally to satisfy the constraint.

## Entropy-Gradient v/s Loss-Based
Differences in this approach from the traditional loss-based methods are:
* The entropy gradient differs from the normal cross entropy as it adjusts weights based
on the entropy functional, focusing on information gain (entropy) rather than loss
minimization.
* Traditional methods focus on reducing the loss, but the entropy-gradient method focuses
more on a stable growth of knowledge (z).
* the knowledge (z) is constrained for better training in the entropy gradient approach.
* In the traditional method, there might be large updates to the weights. This might lead to
unstable training. However, with the entropy-gradient approach, the updates are more uniform, so
the training is more stabilized.

## Model Results
This base model was able to produce an accuracy of around 71%.

![image](https://github.com/user-attachments/assets/182411fd-a19d-4bda-b0b1-a96a4a73a209)

From this graph, it can be deduced that:
* There is no overfitting: The training and testing accuracies remain close. So the model
doesn’t overfit.
* Stable learning: Since there are no sudden fluctuations, the training can be said to be stable
and smooth.
* Saturation point is around 35 epochs since the graph levels out afterward.

On testing some random values from the MNIST data set, the model produced the following results, which had an accuracy of 70%.

![image](https://github.com/user-attachments/assets/5e68206d-d78a-4c29-b75a-c24788dc0025)


