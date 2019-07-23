<h1>Expectation Maximization - An explanation of statistical inference using the example of Gaussian Mixture Models</h1>

Clustering forms a group of unsupervised learning algorithms that are designed to find unknown patterns in data. It is one of the fundamental methods for many researches and practitioners working with data. The k-means clustering algorithm is one of the best known and simpliest clustering methods used today. The algorithm assigns each data point to exactly one cluster. This is called hard assignment. However, the lack of in-between assignment often leads to issues regarding overlapping clusters. Additionally, k-means ignores the variance of the clusters and therefore it can happen that the algorithm does not work well with different cluster sizes.

In this article the Expectation Maximization (EM) algorithm is explained and discussed in simple words as a fundamental principal of statistical inference. Afterwards an implementation of the concept is presented in Python using the example of univariate Gaussian Mixture Models (GMMs). The article is written for researchers and practitioners that have a basic understanding of statistics and machine learning.

<h2>Model</h2>
EM clustering is a method to adress the issue of hard assignment and data generedated by different cluster sizes. It adds the statistical assumption that every data point <i>x<sub>i</sub></i> is randomly drawn from a statistical distribution <i>k</i>. In Gaussian Mixture Models (GMMs) the underlying distribution is a normal distribution. Therefore, every cluster <i>k &isin; {1,...,K}</i> equals a normal distribution with mean &mu;<sub>k</sub> and standard deviation &sigma;<sub>k</sub>:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;z_{i}&space;\sim&space;Categorial(\frac{1}{K},...,\frac{1}{K})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;z_{i}&space;\sim&space;Categorial(\frac{1}{K},...,\frac{1}{K})" title="z_{i} \sim Categorial(\frac{1}{K},...,\frac{1}{K})" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;x_{i}|z_{i},&space;\mu&space;\sim&space;N(z_i^T,\mu,1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;x_{i}|z_{i},&space;\mu&space;\sim&space;N(z_i^T,\mu,1)" title="x_{i}|z_{i}, \mu \sim N(z_i^T,\mu,1)" /></a>
</p>

<i>K</i> is a hyperparameter of the model and determines the number of clusters which is fixed. A <b>hyperparameter</b> is a constant that has to be estimated before inferring the model parameters. Usually a hyperparameter does not change during training. However, a <b>model parameter</b> is not known before. It has to be estimated during inference. In many cases model parameters are randomly initialized. <br>
Another relevant variable is <i>x</i>. It represents the observed data which depends on the cluster assignment <i>z</i>, the mean &mu; and the standard deviation &sigma;. <b>&Phi;</b> is a <i>K</i> dimensional vector of a categorial distribution. It encodes the prior probability assumption that a data point <i>x<sub>i</sub></i> was generated from a certain cluster <i>z<sub>i</sub></i>. We do not have an inital assumption. Therefore, we set <i>&Phi;<sub>k</sub> = 1/K</i> for <i>k &isin; {1,...,K}</i>. 

<h2>EM Clustering</h2>

The goal of the algorithm is to maximize the overall likelihood <i>log(p(&Phi;|x))</i> that the data <i>x</i> is observed given the final cluster assignments <i>z<sub>i</sub></i> and its parameters (&mu; and &sigma). A requirement of the EM algorithm is that the <b>probability density function (pdf)</b> of the a posteriori distribution is known and available in closed form. The probability density function of the posterior distribution in univariate GMMs is the probability density function of the univariate normal distribution: 
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;p(x)=\frac{1}{\sqrt{2\pi&space;\sigma^{2}}}e^{-\frac{(x-\mu)^2}{2&space;\sigma^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;p(x)=\frac{1}{\sqrt{2\pi&space;\sigma^{2}}}e^{-\frac{(x-\mu)^2}{2&space;\sigma^2}}" title="p(x)=\frac{1}{\sqrt{2\pi \sigma^{2}}}e^{-\frac{(x-\mu)^2}{2 \sigma^2}}" /></a>
</p>
The EM algorithm computes a point estimate of the parameters of the actual posterior distribution. However, the function that is optimized during inference is non-convex. The properties of a non-convex function let conclude that a found optimum is not guaranteed to be the global optimum. Therefore, it only finds a local optimal solution for the latent variables (<i>z</i>, <i>&mu;</i>; <i>&sigma;</i>) by using the observed variables <i>x</i>.

<h3>E-Step</h3>
In the first step the probability for each data point <i>x<sub>i</sub></i> and every possible cluster assignment is computed. 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;p(x_{i}|&space;\mu_{k},&space;\sigma^2_{k})&space;=&space;\frac{\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}{\sum_{k=1}^{K}\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;p(x_{i}|&space;\mu_{k},&space;\sigma^2_{k})&space;=&space;\frac{\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}{\sum_{k=1}^{K}\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}" title="p(x_{i}| \mu_{k}, \sigma^2_{k}) = \frac{\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}{\sum_{k=1}^{K}\Phi_{k}N(x_{i},\mu_{k},\sigma^2_{k})}" /></a>
</p>

In the numerator the prior expectation of the cluster assignment is multiplied by the density of the current selected cluster. The denominator is the normalization factor that simply computes the sum of densities over all possible cluster assignments <i>k</i> &isin; {1,...,K}. Therefore, the outcome of the normalized densities is a probability value between 0 and 1. It represents the probability that a data point <i>x<sub>i</sub></i> is generated by cluster <i>k</i>. These probability values are computed for each data point and each possible cluster assignment. 

<h3>M-Step</h3>
In the next step the model parameters &mu; and &sigma; are updated. The prior expectation of the cluster assignment is usually fixed, but could also be updated by computing the average of each column in the table above. <br>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\mu_{k}&space;=&space;\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})x_{i}}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\mu_{k}&space;=&space;\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})x_{i}}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}" title="\mu_{k} = \frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})x_{i}}{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})}" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\sigma_{k}&space;=&space;\sqrt{\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})(x_{i}-\mu_{k})^2}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{k}&space;=&space;\sqrt{\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})(x_{i}-\mu_{k})^2}{\sum_{i=1}^{N}p(x_{i}|\mu_{k},&space;\sigma^2_{k})}}" title="\sigma_{k} = \sqrt{\frac{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})(x_{i}-\mu_{k})^2}{\sum_{i=1}^{N}p(x_{i}|\mu_{k}, \sigma^2_{k})}}" /></a>
</p>

The updated value for &mu; is the weighted average of all data points <i>x<sub>i</sub></i> that are assigned to cluster <i>k</i>. Similar the updated value for &sigma; is also computed by using the probabilities as weights.

<h2>Example</h2>
Let's assume we observe the data points x. We set <i>K</i>=2 with a prior cluster assignment &Phi;.
The initial values are:

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;x&space;=&space;[3,&space;4.5],\mu&space;=&space;[2,5],&space;\sigma&space;=&space;[1,1],&space;\phi&space;=&space;[0.5,&space;0.5]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;x&space;=&space;[3,&space;4.5],\mu&space;=&space;[2,5],&space;\sigma&space;=&space;[1,1],&space;\phi&space;=&space;[0.5,&space;0.5]" title="x = [3, 4.5],\mu = [2,5], \sigma = [1,1], \phi = [0.5, 0.5]" /></a>
</p>

<img src="model.png" />

The density for data point 1 (assuming cluster 1 generated it):

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;N(3,\mu_{1},\sigma^2_{1})=N(3,2,1)=0,24" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;N(3,\mu_{1},\sigma^2_{1})=N(3,2,1)=0,24" title="N(3,\mu_{1},\sigma^2_{1})=N(3,2,1)=0,24" /></a>
</p>

The density for data point 1 (assuming cluster 2 generated it):

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;N(3,\mu_{2},\sigma^2_{2})=N(3,5,1)=0,05" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;N(3,\mu_{2},\sigma^2_{2})=N(3,5,1)=0,05" title="N(3,\mu_{2},\sigma^2_{2})=N(3,5,1)=0.05" /></a>
</p>

Next step is to normalize the densities to compute the probability values (E-Step):

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;p(x_{1},\mu_{1},\sigma^2_{1})=\frac{0.5*0.24}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.83" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;p(x_{1},\mu_{1},\sigma^2_{1})=\frac{0.5*0.24}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.83" title="p(x_{1},\mu_{1},\sigma^2_{1})=\frac{0.5*0.24}{0.5*0.24 + 0.5*0.05}=0.83" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;p(x_{1},\mu_{2},\sigma^2_{2})=\frac{0.5*0.05}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.17" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;p(x_{1},\mu_{2},\sigma^2_{2})=\frac{0.5*0.05}{0.5*0.24&space;&plus;&space;0.5*0.05}=0.17" title="p(x_{1},\mu_{2},\sigma^2_{2})=\frac{0.5*0.05}{0.5*0.24 + 0.5*0.05}=0.17" /></a>
</p>

The probability that data point 1 was generated by cluster 1 is 83 percent whereas the probability that it was generated by cluster 2 is 17 percent. We also compute the probability values for data point 2:

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;p(x_{2},\mu_{1},\sigma^2_{1})=0.05" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;p(x_{2},\mu_{1},\sigma^2_{1})=0.05" title="p(x_{2},\mu_{1},\sigma^2_{1})=0.05" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;p(x_{2},\mu_{2},\sigma^2_{2})=0.95" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;p(x_{2},\mu_{2},\sigma^2_{2})=0.95" title="p(x_{2},\mu_{2},\sigma^2_{2})=0.95" /></a>
</p>

The probability that data point 2 was generated by cluster 1 is 5 percent whereas the probability that it was generated by cluster 2 is 95 percent.

Last step is to update the model parameters (M-Step). These are the new estimates after the first iteration:
<p>
  <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\mu_{1}&space;=&space;\frac{0.83*3&plus;0.05*4.5}{0.83&plus;0.0.05}=3.09" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\mu_{1}&space;=&space;\frac{0.83*3&plus;0.05*4.5}{0.83&plus;0.05}=3.09" title="\mu_{1} = \frac{0.83*3+0.05*4.5}{0.83+0.0.05}=3.09" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\mu_{2}&space;=&space;\frac{0.17*3&plus;0.95*4.5}{0.17&plus;0.95}=4.27" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\mu_{2}&space;=&space;\frac{0.17*3&plus;0.95*4.5}{0.17&plus;0.95}=4.27" title="\mu_{2} = \frac{0.17*3+0.95*4.5}{0.17+0.95}=4.27" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\sigma_{1}&space;=&space;\sqrt{\frac{0.83*(3-2)^2&plus;0.05*(4.5-2)^2}{0.83&plus;0.05}}=1.14" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{1}&space;=&space;\sqrt{\frac{0.83*(3-2)^2&plus;0.05*(4.5-2)^2}{0.83&plus;0.05}}=1.14" title="\sigma_{1} = \sqrt{\frac{0.83*(3-2)^2+0.05*(4.5-2)^2}{0.83+0.05}}=1.14" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\sigma_{2}&space;=&space;\sqrt{\frac{0.17*(3-5)^2&plus;0.95*(4.5-5)^2}{0.18&plus;0.95}}=0.82" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{2}&space;=&space;\sqrt{\frac{0.17*(3-5)^2&plus;0.95*(4.5-5)^2}{0.18&plus;0.95}}=0.82" title="\sigma_{2} = \sqrt{\frac{0.17*(3-5)^2+0.95*(4.5-5)^2}{0.18+0.95}}=0.82" /></a>
</p>

In practice both steps are repeated several times. It is guaranteed that the model parameters converge to a stationary point. Let's evaluate the convergence of the model. Therefore, we compute the log likelihood before and after the first iteration:

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;log(p(x|\mu,&space;\sigma^2))&space;=&space;-2.46" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;log(p(x|\mu,&space;\sigma^2))&space;=&space;-2.46" title="log(p(x|\mu, \sigma^2)) = -2.46" /></a>
</p>

<p>
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;log(p(x|\mu,&space;\sigma^2))&space;=&space;-1.81" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;log(p(x|\mu,&space;\sigma^2))&space;=&space;-1.81" title="log(p(x|\mu, \sigma^2)) = -1.81" /></a>
</p>

It turns out that the log likelihood is increasing. That means it is more likely that the estimated model after the first iteration has generated the observed data. Therefore, the algorithm works as expected and the model is getting better.

