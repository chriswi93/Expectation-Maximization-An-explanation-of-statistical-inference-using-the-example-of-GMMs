<h1>Parameter Inference of Gaussian Mixture Models using Expectation Maximization</h1>

Clustering forms a group of unsupervised learning algorithms that are designed for finding unknown patterns in data. It is a fundamental part of many researches and practitioners working with data. K-Means is one of the best known and easiest clustering methods used today. The algorithm uses hard assignment to assign a data point to exactly one cluster. However, the lack of in-between assignment often leads to issues regarding overlapping clusters. 

In this article the Expectation Maximization algorithm is explained and discussed in simple words as a fundamental principal of statistical inference. Afterwards an implementation of the concept is presented in Python using the example of univariate Gaussian Mixture Models. The article is written for researchers and practitioners with a fundamental understanding of Machine Learning and Statistics.

<h2>Expectation Maximization Clustering</h2>
EM Clustering is a method to adress the issue of hard assignment. It adds the statistical assumption that every data point <i>x<sub>i</sub></i> is randomly drawn from a distribution. In Gaussian Mixture Models the underlying assumption is a normal distribution. Therefore, every cluster <i>k<sub>i</sub></i> out of <i>K</i> clusters equals a normal distribution with the expected value &mu;<sub>k</sub> and variance &sigma;<sup>2</sup><sub>k</sub>. Therefore, we formally write:
<p align="center"><i>
  x<sub>i</sub> ~ N(&mu;<sub>k</sub>,&sigma;<sup>2</sup><sub>k</sub>)
</i></p>
<i>K</i> is a hyperparameter of the model. A <b>hyperparameter</b> is a constant that has to be defined before inferencing the model parameters. Usually a hyperparameter does not change during inference. However, a <b>model parameter</b> is not known in advance. It has to be estimated. In many cases model parameters are randomly initialized before parameter inference.

Every normal distribution is 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{W}(A,f)&space;=&space;(T,\bar{f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{W}(A,f)&space;=&space;(T,\bar{f})" title="\mathcal{W}(A,f) = (T,\bar{f})" /></a>
