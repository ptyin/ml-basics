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
\max & \sum_{y \in \mathcal{Y}} c_{y} \log p_{y} \\
\text { s.t. } & \sum_{y \in \mathcal{Y}} p_{y}=1 \\
&p_y \geq 0, & \forall y \in \mathcal{Y}
\end{aligned}
$$

Lagrangian problem is:
$$
\begin{aligned}
&\left\{
\begin{array}{l}
F=-\sum_{y \in \mathcal{Y}} c_{y} \log p_{y}+\lambda(\sum_{y \in \mathcal{Y}} p_{y}-1)-\sum_{y\in\mathcal{Y}} \mu_yp_y\\
\mu_y\ge0 & \forall y\in \mathcal Y
\end{array}
\right.
\\ \\
\Rightarrow & \left\{
\begin{array}{ll}
\frac{\partial F}{\partial p_{y}}=-\frac{c_{y}}{p_{y}}+\lambda -\mu_y=0 & \forall y\in \mathcal Y\\
\mu_yp_y=0 & \forall y\in\mathcal Y
\end{array}
\right.
\end{aligned}
$$

- For $\lambda$, 

$$
\begin{array}{l}
\because \mu_yp_y=0 \text{ and } c_y=\lambda p_y-\mu_yp_y \\
\therefore c_y=\lambda p_y \Leftrightarrow p_y=\frac {c_y}{\lambda} \\
\because \sum_{y\in\mathcal Y}p_y=1 \\
\therefore \sum_{y\in\mathcal Y}p_y=\frac1\lambda\sum_{y\in\mathcal Y}c_y=1 \\
\therefore \lambda=\sum_{y\in\mathcal Y}c_y
\end{array}
$$

- Therefore, $p_y=\frac{c_y}{\lambda}=\frac{c_y}{\sum_{y\in\mathcal Y}c_y}$

### ( ii ) Using the above consequence, prove that, the maximum-likelihood estimates for Naive Bayes model are as follows:

$$
p(y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)}{m}
$$
and
$$
p_{j}(x \mid y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y \wedge x_{j}^{(i)}=x\right)}{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)}
$$

#### Answer:

The first step is to re-write the log-likelihood function in a way that makes direct use of "counts" taken from the training data:
$$
\begin{aligned}
l(\Omega)=& \sum_{i=1}^{m} \log p\left(y^{(i)}\right)+\sum_{i=1}^{m} \sum_{j=1}^{n} \log p_{j}\left(x_{j}^{(i)} \mid y^{(i)}\right) \\
=& \sum_{y \in \mathcal{Y}} \operatorname{count}(y) \log p(y) \\
&+\sum_{j=1}^{n} \sum_{y \in \mathcal{Y}} \sum_{x \in\{-1,+1\}} \text { count }_{j}(x \mid y) \log p_{j}(x \mid y)
\end{aligned}
$$
where as before
$$
\begin{gathered}
\operatorname{count}(y)=\sum_{i=1}^{m}1\left(y^{(i)}=y\right) \\
\operatorname{count}_{j}(x \mid y)=\sum_{i=1}^{m}1\left(y^{(i)}=y \wedge x_{j}^{(i)}=x\right)
\end{gathered}
$$
Consider first maximization of this function with respect to the $q(y)$ parameters. It is easy to see that the term
$$
\sum_{j=1}^{d} \sum_{y \in \mathcal{Y}} \sum_{x \in\{-1,+1\}} \operatorname{count}_{j}(x \mid y) \log p_{j}(x \mid y)
$$
does not depend on the $p(y)$ parameters at all. Hence to pick the optimal $p(y)$ parameters, we need to simply maximize
$$
\sum_{y \in \mathcal{Y}} \operatorname{count}(y) \log p(y)
$$
Subject to the constraints $p(y) \geq 0$ and $\sum_{y=1}^{k} p(y)=1$, by the consequence of **( i )** , the values for $q(y)$ which maximize this expression under these constraints is simply
$$
p(y)=\frac{\operatorname{count}(y)}{\sum_{y=1}^{k} \operatorname{count}(y)}=\frac{\operatorname{count}(y)}{n}
$$
By a similar argument, we can maximize each term of the form
$$
\sum_{x \in\{-1,+1\}} \text { count }_{j}(x \mid y) \log p_{j}(x \mid y)
$$
Applying **( i )**, we can get
$$
p_j(x\mid y)=\frac{\text{count}_j(x\mid y)}{\sum_{x\in\{-1,1\}}\text{count}_j(x\mid y)}=\frac{\text{count}_j(x\mid y)}{\text{count}(y)}
$$
