# Machine Learning Specialization 2022

## Disclaimer
This is my personal notes on the course, meaning these are my interpretations of the course material and lectures, my solutions the course assignments - which might be suboptimal in certain cases. Any mistakes in either the notes or the solutions are mine and not the course creator's. Feel free to use these notes and solutions to bolster your understanding, but __do not use them to surpass the course prematurely, nor to get yourself out of a jam.__ _This will not help you and leave you with a frailed understanding of the material._

## Specialization Outline
The Specialization is divided into 3 courses:
* Part 1: Supervised Machine Learning: Regression and Classification
* Part 2: Advanced Learning Algorithms
* Part 3: Unsupervised Learning: Reccomenders, Reinforcement Learning

# Notes
## [Part 1: Supervised Machine Learning: Regression and Classification](1.%20Supervised%20Machine%20Learning/)
### <u>Week 1: Introduction to Machine Learning</u>
#### Overview
* __Machine Learning__ is "the science of getting computers to act without being explicitly programmed" (Arthur Samuel 1959) and is a subfield of _Artificial Intelligence_.
* There are many applications of machine learning in daily life, even without us noticing it.
  * Some include Web Search, programming Self-Driving Cars, Speech Recognition, Advertising, Healthcare, Agriculture, and much, much more.
  * Andrew described a few in his [TED talk](https://youtu.be/reUZRyXxUs4).
* AGI (Artificial General Intelligence) is the intelligence of a machine that could equal or surpass human intelligence, but has been overhyped. It might take a long time, or a very long time, to achieve, but it seems the best way to get closer is through learning algorithms.
* There is a massive demand for machine learning engineers, and the demand is most likely going to increase, thus it is a great time to learn it.

#### Supervised vs Unsupervised Learning
* __Supervised Learning__ is when you have a dataset with the correct answers, and you want to learn a function that maps from the input to the output.
  * Some examples include spam filtering, speech recognition, machine translations, online advertising, self-driving cars, and visual inspection.
  * 2 types of supervised learning:
    * __Regression__ is when the output is a continuous value (real number).
      * As an example, you could use regression to predict the price of a house.
    * __Classification__ is when the output is a discrete value (category).
      * As an example, you could use classification to predict whether a tumor is malignant or benign.
* __Unsupervised Learning__ is when you have a dataset without the correct answers, and you want to learn a function that maps from the input to the output.
  * We ask the algorithm to determine the structure of the data, and it will try to find patterns.
  * Types of unsupervised learning:
    * __Clustering__ is when you want to group similar data points together.
      * As an example, you could use clustering to group together similar news articles.
    * __Dimensionality Reduction__ is when you want to reduce the number of features in your dataset.
      * As an example, you could use dimensionality reduction to reduce the number of pixels in an image.
    * __Anomaly Detection__ is when you want to find unusual data points.
      * As an example, you could use anomaly detection to find unusual credit card transactions.
    * __Reinforcement Learning__ is when you want to train an agent to perform a task in an environment.
      * As an example, you could use reinforcement learning to train a robot to walk.

#### Regression Model
* Linear regression model with one variable is just fitting a straight line to the data.
  * Could help predict the price of a house based on its size.
* The model (f) outputs a prediction (y-hat) given some inputs (x) after it is trained.
  * The model, f, is a mathematical formula eg. $f_{w,b}(x) = w x + b$ or just $f(x) = w x + b$, which is a linear model.
  * w and b referred to as the parameters or weights of the model.
* The __Cost Function__ is a function that is used to measure the performance of the model.
  * Calculated with $\frac{1}{2m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})^2$ where $f(x^{(i)})$ is the prediction of the model for the i-th training example, and $y^{(i)}$ is the actual value of the i-th training example.
  * Also written as $J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$ and we want to minimize $J(w,b)$.

#### Train Model with Gradient Descent
* __Gradient Descent__ is one of the most important building blocks in Machine Learning. It is an algorithm that is used to minimize cost function.
  * The gradient descent algorithm is as follows:
    1. Initialize the parameters w and b.
    2. Calculate the cost function $J(w,b)$.
    3. Calculate the partial derivatives of $J(w,b)$ with respect to w and b.
    4. Update the parameters w and b with the partial derivatives, simultaneously.
    5. Repeat steps 2 to 4 until convergence.
  * __Convergence__ is when $J(w,b)$ stops decreasing.
  * The intuition behind gradient descent is that we want to find the minimum of the cost function, and we can do this by taking steps in the direction of the negative gradient.
* The __Learning Rate__ is a hyperparameter that controls how big the steps are in the direction of the negative gradient.
  * If the learning rate is too small, it will take a long time to converge.
  * If the learning rate is too big, it might not converge, or it might even diverge.

### <u>Week 2: Regression with Multiple Input Variables</u>

#### Multiple Linear Regression
* __Vectorization__ is when you perform operations on vectors and matrices instead of individual numbers.
  * This is much faster than performing operations on individual numbers.
  * Also uses specialized hardware to perform operations on vectors and matrices.
* An alternative Gradient Descent is the __Normal Equation__.
  * The normal equation is as follows: $\theta = (X^T X)^{-1} X^T y$ where $\theta$ is the vector of parameters, $X$ is the matrix of features, and $y$ is the vector of outputs.
  * The normal equation is much faster than gradient descent, but it is not scalable to large datasets.
  * The normal equation is also not suitable for large datasets because it requires the inverse of $X^T X$, which is computationally expensive.

#### Gradient Descent in practice
* __Feature Scaling__ is when you scale the features so that they are in the same range.
  * This makes gradient descent converge faster.
* To vertify that gradient descent is working, plot the graph of the cost function against the number of iterations.
  * If the cost function is decreasing, then gradient descent is working.
    * If it is decreasing too slowly, then you might need to increase the __learning rate__.
  * If the cost function is not decreasing, then gradient descent is not working, or the learning rate is too big.
    * Try decreasing the __learning rate__.
* Choosing the most appropriate features is known as __feature engineering__.


__In progress...__


# References
* [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
