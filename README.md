<h1>Parameter Inference of Gaussian Mixture Models using Expectation Maximization</h1>

Clustering forms a group of unsupervised learning algorithms that are designed for finding unknown patterns in data. It is a fundamental part for many researches and practitioners working with data. K-Means is one of the best known and easiest clustering methods used today. The algorithm uses hard assignment to assign a data point to exactly one cluster. However, the lack of in-between assignment often leads to issues regarding overlapping clusters. 

In this article the Expectation Maximization algorithm is explained and discussed in simple words as a fundamental principal of statistical inference. Afterwards an implementation of the concept is presented in Python using the example of univariate Gaussian Mixture Models. The article is written for researchers and practitioners with a fundamental understanding of Machine Learning and Statistics.

<h2>Model</h2>
EM Clustering is a method to adress the issue of hard assignment. It adds the statistical assumption that every data point <i>x<sub>i</sub></i> is randomly drawn from a distribution. In Gaussian Mixture Models the underlying assumption is a normal distribution. Therefore, every cluster <i>k<sub>i</sub></i> out of <i>K</i> clusters equals a normal distribution with mean &mu;<sub>k</sub>. For simplicity the variance &sigma;<sup>2</sup> is set to 1. Blei et al. (2016) formally write:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\mu_{k}&space;\sim&space;N(0,\sigma^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\mu_{k}&space;\sim&space;N(0,\sigma^2)" title="\mu_{k} \sim N(0,\sigma^2)" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;z_{i}&space;\sim&space;Categorial(\frac{1}{K},...,\frac{1}{K})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;z_{i}&space;\sim&space;Categorial(\frac{1}{K},...,\frac{1}{K})" title="z_{i} \sim Categorial(\frac{1}{K},...,\frac{1}{K})" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;x_{i}|z_{i},&space;\mu&space;\sim&space;N(z_i^T,\mu,1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;x_{i}|z_{i},&space;\mu&space;\sim&space;N(z_i^T,\mu,1)" title="x_{i}|z_{i}, \mu \sim N(z_i^T,\mu,1)" /></a>
</p>

<i>K</i> is a hyperparameter of the model and determines the number of clusters which is fixed. A <b>hyperparameter</b> is a constant that has to be defined before inferring the model parameters. Usually a hyperparameter does not change during training. However, a <b>model parameter</b> is not known before. It has to be estimated during inference. In many cases model parameters are randomly initialized. <b>x</b> is the observed data which depends on cluster assignment <i>z<sub>i</sub></i> and the mean &Mu;. <b>&Phi;</b> is a <i>K</i> dimensional vector of a categorial distribution. It encodes the prior probability assumption that a data point <i>x<sub>i</sub></i> was generated from a certain cluster <i>z<sub>i</sub></i>. This is also a hyperparameter. For simplicity it is set to <i>&Phi;<sub>k</sub> = 1/K</i> for <i>k &isin; K</i>. 

<h2>EM Clustering</h2>

The algorithm optimizes the probability that every <i>x<sub>i</sub></i> is assigned to cluster <i>z<sub>i</sub></i> with a overall high likelihood of the model parameters given the observed data <i>p(&Phi;|x)</i>. A very important condition of the Expectation Maximization algorithm is that the <b>probability density function (pdf)</b> of the a posteriori distribution is known and available in closed form. This is one of many aspects that differentiates Expectation Maximization from Variational Inference. The probability density function of the posterior distribution in univariate Gaussian Mixture Models is the probability density function of the univariate normal distribution: 
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;p(x)=\frac{1}{\sqrt{2\pi&space;\sigma^{2}}}e^{-\frac{(x-\mu)^2}{2&space;\sigma^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;p(x)=\frac{1}{\sqrt{2\pi&space;\sigma^{2}}}e^{-\frac{(x-\mu)^2}{2&space;\sigma^2}}" title="p(x)=\frac{1}{\sqrt{2\pi \sigma^{2}}}e^{-\frac{(x-\mu)^2}{2 \sigma^2}}" /></a>
</p>
Expectation Maximization computes a point estimate of the actual posterior distribution. However, the function that is optimized during inference is non-convex. The properties of a non-convex function let conclude that a found optimum is not guaranteed to be the global optimum. It learns a local optimal solution for the latent variables <b>z</b> and <b>&Mu;</b> by using the observed variable <b>x</b>. The objective of the EM algorithm is to find a maximum likelihood estimate for the parameters of the model. In other words the algorithm finds a model parameter configuration the observed data was generated from very likely. 

<h3>E-Step</h3>
In the first step the probability for each data point <i>x<sub>i</sub></i> and every possible cluster assignment is computed. 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;p(x_{i}|&space;\mu_{k},&space;\sigma^2_{k})&space;=&space;\frac{\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}{\sum_{k=1}^{K}\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;p(x_{i}|&space;\mu_{k},&space;\sigma^2_{k})&space;=&space;\frac{\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}{\sum_{k=1}^{K}\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}" title="p(x_{i}| \mu_{k}, \sigma^2_{k}) = \frac{\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}{\sum_{k=1}^{K}\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}" /></a>
</p>

In the numerator the prior expectation of the cluster assignment is multiplied by the density of the current selected cluster <i>z<sub>i</sub></i>. The denominator is the normalization factor that simply represents the sum of densities over all possible cluster assignments <i>z<sub>i</sub></i> &isin; {1,...,K}. Therefore, the outcome of the normalized densities is a probability value between 0 and 1. It represents the probability that a data point <i>x<sub>i</sub></i> is generated by cluster <i>z<sub>i</sub></i>. These probability values are computed for each data point and each possible cluster assignment. 

<h3>M-Step</h3>
In the next step the model parameters (&Mu; and &sigma;) are updated. The prior expectation of the cluster assignment is usually fixed, but could also be updated by computing the average of each column in the table above. <br>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\mu_{k}&space;=&space;\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})x_{i}}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\mu_{k}&space;=&space;\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})x_{i}}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}" title="\mu_{k} = \frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})x_{i}}{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})}" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\sigma_{k}&space;=&space;\sqrt{\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})(x_{i}-\mu_{k})^2}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_{k}&space;=&space;\sqrt{\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})(x_{i}-\mu_{k})^2}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}}" title="\sigma_{k} = \sqrt{\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})(x_{i}-\mu_{k})^2}{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})}}" /></a>
</p>

The updated value for &Mu; is the weighted average of all data points <i>x<sub>i</sub></i> that are assigned to cluster <i>k</i>. Similar the updated value for &sigma; is also computed by using the probabilities as weights.

<h2>Example</h2>
Let's assume we observe the data points x=[3,4.5]. We assume <i>K</i>=2 with a prior cluster assignment &Phi;=[0.5, 0.5].
The initial values for &Mu; and &sigma; are:

&Mu; = [2,5] 
&sigma; = [1,1]

The density for data point 1 assuming cluster 1 generated it:

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;N(3,\mu_{1},\sigma^2_{1})=N(3,2,1)=0,24" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;N(3,\mu_{1},\sigma^2_{1})=N(3,2,1)=0,24" title="N(3,\mu_{1},\sigma^2_{1})=N(3,2,1)=0,24" /></a>
</p>

The density for data point 1 assuming cluster 2 generated it:

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;N(3,\mu_{2},\sigma^2_{2})=N(3,5,1)=0,05" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;N(3,\mu_{2},\sigma^2_{2})=N(3,5,1)=0,05" title="N(3,\mu_{2},\sigma^2_{2})=N(3,5,1)=0.05" /></a>
</p>

Next step is to normalize the densities to compute the probability values (E-Step):

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;p(x_{1},\mu_{1},\sigma^2_{1})=\frac{0.5*0.24}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.83" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;p(x_{1},\mu_{1},\sigma^2_{1})=\frac{0.5*0.24}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.83" title="p(x_{1},\mu_{1},\sigma^2_{1})=\frac{0.5*0.24}{0.5*0.24 + 0.5*0.05}=0.83" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;p(x_{1},\mu_{2},\sigma^2_{2})=\frac{0.5*0.05}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.17" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;p(x_{1},\mu_{2},\sigma^2_{2})=\frac{0.5*0.05}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.17" title="p(x_{1},\mu_{2},\sigma^2_{2})=\frac{0.5*0.05}{0.5*0.24 + 0.5*0.05}=0.17" /></a>
</p>

We also compute it for data point 2:

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;p(x_{2},\mu_{1},\sigma^2_{1})=0.05" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;p(x_{2},\mu_{1},\sigma^2_{1})=0.05" title="p(x_{2},\mu_{1},\sigma^2_{1})=0.05" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;p(x_{2},\mu_{2},\sigma^2_{2})=0.95" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;p(x_{2},\mu_{2},\sigma^2_{2})=0.95" title="p(x_{2},\mu_{2},\sigma^2_{2})=0.95" /></a>
</p>

Last step is to perform the update operations for the model parameters (M-Step):

