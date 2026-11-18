 # MNIST Neural Networks from First Principles
### Mathematical Foundations, Implementation, and Computational Analysis


## Overview


### What the project is

Using the MNIST handwritten digit dataset, I built a multilayer neural network to explore forward propagation, backpropagation, activation functions, loss computation, and optimization. The goal isn't simply to classify handwritten digits, but to understand how each mathematical operation contributes to the learning process.

### Motivation

The increasing accessibility of machine learning frameworks has significantly simplified the implementation of neural networks. However, these abstractions can obscure the underlying mathematical operations, particularly for individuals attempting to understand how neural networks learn and optimize their parameters.

### Methodology

The project serves as a tangible output while learning the fundamental mathematics used for Artificial Intelligence, by building it from the ground up using Pure Python implementation, without using any NumPy and PyTorch or TensorFlow libraries throughout the start. 

The project also encourages the use and need for low level implementations of several functions, such as a summation function made with pure python, or the crucial mathematical functions used in Machine Learning, such as the dot product, softmax and more.

## Neural Network Architecture

![Neural Network Architecture](assets/neural-network.jpeg)

### 4 Layered Neural Network

This design is suitable enough for a MNIST Handwritten Digit Classifier Network. The whole purpose of the project is to learn the core ideas or the fundamental mathematics of a neural network.

## Mathematical Principles

### Linear Algebra

In programming, linear algebra has been widely used especially with game development. What actually is the difference between linear algebra and the traditional and usual algebra we learned during high school and college? And why do we need this for Artificial Intelligence as well?

The core idea is that the algebra we know only solves one problem at a time. What makes linear algebra so special is that it allows multiple calculations to happen all at once. And since we are dealing with multivariable problems when it comes to game development or artificial intelligence, linear algebra is crucial in these cases.

#### Vectors

In neural networks, vectors are seen differently than how they are in math and physics. In math, they are seen as arrows, a quantity that has both magnitude and direction. In our case, we refer to it as containers for data. It is an array of numbers where it represents a specific attribute or feature. 

Take this equation for example

$$ \mathbf{x} =  
\begin{bmatrix} 
x_1 \\ 
x_2 \\ 
x_3 
\end{bmatrix} $$

We can say that the whole vector is an object. While $x_1$, $x_2$, $x_3$ is the features of the object. Suppose that object is a chair, we could say that $x_1$ is representing the legs of a chair, then $x_2$ is whether the chair has a backrest or not, and $x_3$ could be if the chair is portable or not.

#### Matrices

Like vectors, a matrix is an array of numbers and information that serves as containers for the data that we need in neural networks.

There are two use cases for the matrix:

1.) Containers of data - storing and representing a series of valuable data (an example would be a data set)
2.) Containers of weights - a weight represents the neuron's ability to influence relative to the received input (most common use case)

Consider this equation:

$$
\mathbf{A} = 
\begin{bmatrix}
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2
\end{bmatrix}
$$

Where $A$ can be a representation of several objects with their own features, take $x_1$ and $x_2$ for example; they are on the same column, which can also be represented as a vector. While $y$ and $z$ also operate on the same idea. They can conceptually mean that, matrix $A$ contains three objects of the same kind with two features for each. This is how matrices are used as data sets. 

As for how matrices is used commonly in neural networks, which are storing weights. It is a bit different than how it is used as data sets.

Consider this matrix $W$:

$$
\mathbf{W} =
\begin{bmatrix}
0.1 & 0.25 & 0.6 & 0.12 \\
0.23 & 0.43 & 0.23 & 0.11 \\
0.54 & 0.68 & 0.13 & 0.25
\end{bmatrix}
$$

On its own, it simply looks like a matrix with a bunch of numbers in it. Unlike data sets, where the values stored in a matrix is a vector whose use and what it represents are something that we inform beforehand. When storing weights, it is entirely different; it is a bunch of digits distributed randomly in each of their place in the matrix. To understand, knowing why weights are randomly initialized first is important.

### Neurons

Neural network is based on the biological neurons humans have, although artificial; it somewhat copies the "learning" process of a neuron. So we form a mathematical concept for these neurons in our program. 

Consider this matrix again: 

$$
\mathbf{W} =
\begin{bmatrix}
0.1 & 0.25 & 0.6 & 0.12 \\
0.23 & 0.43 & 0.23 & 0.11 \\
0.54 & 0.68 & 0.13 & 0.25
\end{bmatrix}
$$

Each column, is the weights of a neuron. And the digits inside that weight is what makes a neuron distinguish an input, at least when trained. In this case, or rather, in most case; the weights are initialized in random as a "raw" representation of an untrained neuron. 

#### The Mathematical Formula of a Neuron

$$
z = Wx + b
$$

##### This is the most fundamental formula in neural networks

So why is it initialized in random? What exactly are the weights?

#### Weights

This is what the matrix $W$ represents on the mathematical formula of a neuron. It is a matrix filled with digits distributed randomly at first. It is random because it crosses out symmetry on the weights. Imagine if all weight values are equal, then the results would be identical as well. This is will make sense later on when the idea of updating the weights is introduced. 

In this matrix: 

$$
\mathbf{W} =
\begin{bmatrix}
0.1 & 0.25 & 0.6 & 0.12 \\
0.23 & 0.43 & 0.23 & 0.11 \\
0.54 & 0.68 & 0.13 & 0.25
\end{bmatrix}
$$

Each digits in a row represents the weights of that neuron. 

#### Inputs

This is what $x$ is representing on the mathematical formula of a neuron. Most of the time, it is a vector; but there are cases where it is a matrix as well. But in this project, it is usually a vector containing the features and data of an input. 

Consider this vector:

$$
\mathbf{x} = 
\begin{bmatrix}
0.98 \\
0.45 \\
0.32 \\
0.12 
\end{bmatrix}
$$

This is an example of an input where it has four features. It represents the data of an input mathematically. 

#### Bias

For the bias where it represents the $b$ on the neuron formula, the primary reason is purely mathematical. Consider the mathematical formula for the neuron $z = Wx + b$. If we removed $b$, the origin for each would be at (0, 0), making it harder for the network to distinguish patterns and is limited only to that certain origin. By adding a bias, the boundaries can be placed anywhere on the graph, as it "shifts" the equation. Which is why the formula for a neuron is $z = Wx + b$. 

Consider this vector:

$$
\mathbf{b} = 
\begin{bmatrix}
0.2 \\
0.5 \\
0.1 
\end{bmatrix}
$$

One thing you will notice is that the number of rows in a bias is also the same number of rows a weight has. It is because, for each neuron, it has one bias value for each. And we have already established that the number of rows in a matrix containing weights, is the number of neurons there is. 

When combined, the equation eventually looks like this:

$$
z = Wx + b
$$

$$
z = 
\begin{bmatrix}
0.1 & 0.25 & 0.6 & 0.12 \\
0.23 & 0.43 & 0.23 & 0.11 \\
0.54 & 0.68 & 0.13 & 0.25
\end{bmatrix}
\begin{bmatrix}
0.98 \\
0.45 \\
0.32 \\
0.12 
\end{bmatrix} + 
\begin{bmatrix}
0.2 \\
0.5 \\
0.1 
\end{bmatrix}
$$

This is what a neuron operations looks like mathematically. 

You will also notice that the input vector is the only one that isn't symmetrically aligned to both weights and biases. It is the only one with 4 features while the weights and biases only has three. 

### Linear Transformations

#### Scalar to Vector Operations

Now we will introduce the first fundamental operations in linear algebra that is essential for neural networks and is the key to understanding why the mathematics inside a neural network works.

Consider this equation:

$$
c = 4 \qquad
\mathbf{v} = 
\begin{bmatrix}
0.3 \\
0.2 \\
0.1 
\end{bmatrix}
$$
$$
c(\mathbf{v}) = 
4
\begin{bmatrix}
0.3 \\
0.2 \\
0.1 
\end{bmatrix}
$$
$$
c(\mathbf{v}) = 
\begin{bmatrix}
4(0.3) \\
4(0.2) \\
4(0.1) 
\end{bmatrix}
$$

This is an example of a scalar to vector operations, the scalar multiplies every feature of a vector by itself.

But in neural networks, we aren't dealing with just a single digit, but a lot of them. So we do an operation one step further. 

#### Dot Product

We have tackled what scalar to vector operations looks like, but what if it’s a vector to vector, or even a vector to matrix operations? That’s what dot product is. Instead of one value affecting a vector, there are several at once. And knowing how it works and behaves is crucial to understanding how the entire neural network operates without abstraction.

Consider this equation: 

$$
\mathbf{W} = 
\begin{bmatrix}
0.1 & 0.25 & 0.6 & 0.12 \\
0.23 & 0.43 & 0.23 & 0.11 \\
0.54 & 0.68 & 0.13 & 0.25
\end{bmatrix}
\qquad
\mathbf{x} =
\begin{bmatrix}
2 \\
3 \\
1 \\
5
\end{bmatrix}
$$

Let:

$$
\mathbf{Wx} =
\begin{bmatrix}
0.1 & 0.25 & 0.6 & 0.12 \\
0.23 & 0.43 & 0.23 & 0.11 \\
0.54 & 0.68 & 0.13 & 0.25
\end{bmatrix}
\begin{bmatrix}
2 \\
3 \\
1 \\
5
\end{bmatrix}
$$

Unlike scalar to vector where it has one feature, and we only need to multiply that feature to the features inside a vector or a matrix; for a dot product, it is commonly used when a case like vector to vector or vector to matrix happens. 

Take $Wx$ as an example:

$$
\mathbf{Wx} =
\begin{bmatrix}
0.1 & 0.25 & 0.6 & 0.12 \\
0.23 & 0.43 & 0.23 & 0.11 \\
0.54 & 0.68 & 0.13 & 0.25
\end{bmatrix}
\begin{bmatrix}
2 \\
3 \\
1 \\
5
\end{bmatrix}
$$

Multiply the vector $x$ to each corresponding column per neuron in matrix $W$:

$$
\mathbf{Wx} =
\begin{bmatrix}
0.1(2) +& 0.25(3) +& 0.6(1) +& 0.12(5) \\
0.23(2) +& 0.43(3) +& 0.23(1) +& 0.11(5) \\
0.54(2) +& 0.68(3) +& 0.13(1) +& 0.25(5)
\end{bmatrix}
\begin{bmatrix}
2 \\
3 \\
1 \\
5
\end{bmatrix}
$$

and then sum up the product of each feature into one result per each neuron.
