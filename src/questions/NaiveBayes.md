## MLE for Multinomial Naive Bayes

Consider the following definition of MLE problem for multinomials. The input to the problem is a finite set $\mathcal Y$, and a weight $c_y\ge0$ for each $y\in\mathcal Y$. The output from the problem is the distribution $p^{*}$ that solves the following maximization problem.
$$
p^{*}=\arg \max _{p \in \mathcal{P}_{\mathcal{Y}}} \sum_{y \in \mathcal{Y}} c_{y} \log p_{y}
$$
### ( i ) Prove that, the vector $p^{*}$ has components

$$
p_{y}^{*}=\frac{c_{y}}{N}
$$
for $\forall y \in \mathcal{Y}$, where $N=\sum_{y \in \mathcal{Y}} c_{y} .$ (Hint: Use the theory of Lagrange multiplier)

#### Answer:

$$
\begin{aligned}
&\max \sum_{y \in \mathcal{Y}} c_{y} \log p_{y} \\
&\text { s.t. } \sum_{y \in \mathcal{Y}} p_{y}=1 \\
&p(y) \geq 0, \forall y \in \mathcal{Y}
\end{aligned}

\mathop\Rightarrow\limits^{\text{Lagrange}}

\begin{array}{c}
F=\sum_{y \in \mathcal{Y}} c_{y} \log p_{y}+\lambda\left(\sum_{y \in \mathcal{Y}} p_{y}-1\right)\\
\frac{\partial F}{\partial \lambda}=\sum_{y \in \mathcal{Y}} p_{y}-1\\
\frac{\partial F}{\partial p_{y}}=\frac{c_{y}}{p_{y}}+\lambda\\
\lambda=-\sum_{y \in \mathcal{Y}} c_{y}, p_{y}=\frac{c_{y}}{\sum_{y \in \mathcal{Y}} c_{y}}\\
p_{y}=\frac{c_{y}}{N}
\end{array}
$$

### ( ii ) Using the above consequence, prove that, the maximum-likelihood estimates for Naive Bayes model are as follows:

$$
p(y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)}{m}
$$
and
$$
p_{j}(x \mid y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y \wedge x_{j}^{(i)}=x\right)}{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)}
$$

#### Answer:

We now prove the result in theorem 1. Our first step is to re-write the log-likelihood function in a way that makes direct use of "counts" taken from the training data:
$$
\begin{aligned}
L(\underline{\theta})=& \sum_{i=1}^{m} \log q\left(y_{i}\right)+\sum_{i=1}^{m} \sum_{j=1}^{n} \log q_{j}\left(x_{i, j} \mid y_{i}\right) \\
=& \sum_{y \in \mathcal{Y}} \operatorname{count}(y) \log q(y) \\
&+\sum_{j=1}^{n} \sum_{y \in \mathcal{Y}} \sum_{x \in\{-1,+1\}} \text { count }_{j}(x \mid y) \log q_{j}(x \mid y)
\end{aligned}
$$
where as before
$$
\begin{gathered}
\operatorname{count}(y)=\sum_{i=1}^{m}\left[\left[y^{(i)}=y\right]\right] \\
\operatorname{count}_{j}(x \mid y)=\sum_{i=1}^{m}\left[\left[y_{i}=y \text { and } x_{j}^{(i)}=x\right]\right]
\end{gathered}
$$
Consider first maximization of this function with respect to the $q(y)$ parameters. It is easy to see that the term
$$
\sum_{j=1}^{d} \sum_{y \in \mathcal{Y}} \sum_{x \in\{-1,+1\}} \operatorname{count}_{j}(x \mid y) \log q_{j}(x \mid y)
$$
does not depend on the $q(y)$ parameters at all. Hence to pick the optimal $q(y)$ parameters, we need to simply maximize
$$
\sum_{y \in \mathcal{Y}} \operatorname{count}(y) \log q(y)
$$
subject to the constraints $q(y) \geq 0$ and $\sum_{y=1}^{k} q(y)=1$. But by the consequence of **( i )** , the values for $q(y)$ which maximize this expression under these constraints is simply
$$
q(y)=\frac{\operatorname{count}(y)}{\sum_{y=1}^{k} \operatorname{count}(y)}=\frac{\operatorname{count}(y)}{n}
$$
By a similar argument, we can maximize each term of the form
$$
\sum_{x \in\{-1,+1\}} \text { count }_{j}(x \mid y) \log q_{j}(x \mid y)
$$
Applying **( i )**, we can get
$$
q_j(x\mid y)=\frac{\text{count}_j(x\mid y)}{\sum_{x\in\{-1,1\}}\text{count}_j(x\mid y)}=\frac{\text{count}_j(x\mid y)}{\text{count}(y)}
$$
