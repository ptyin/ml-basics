# Naive Bayes and EM Algorithm

## Theorem.

> **Bayes' Theorem**
> $$
> P(A\vert B)=\frac{P(B\vert A)P(A)}{P(B)}
> $$

- If $X$ and $Y$are both continuous, then $f_{X\vert Y=y}(x)=\frac{f_{Y\vert X=x}(y)f_X(x)}{f_Y(y)}$

> **Law of total probability**
>
> If ${\displaystyle \left\{{B_{n}:n=1,2,3,\ldots }\right\}}$ is a finite or countably infinite partition of a sample space.
> $$
> P(A)=\sum_nP(A\mid B_n)P(B_n)
> $$

## Warm up

- In linear regression and logistic regression, $x$ and $y$ are linked through (deterministic) hypothesis function
- Given a set of training data $\mathcal{D}=\{x^{(i)}, y^{(i)}\}_{i=1,\cdots,m}$, $P(\mathcal{D})=\prod_{i=1}^mp_{X\vert Y}(x^{(i)}\vert y^{(i)})p_Y(y^{(i)})$

## Gaussian Distribution

- Normal Distribution $p(x ; \mu, \sigma)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)$
- Multivariate normal distribution $p(x ; \mu, \Sigma)=\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)$
- where $\mu\in\mathbb{R}^n$, $\Sigma\in\mathbb{R}^{n\times n}$ is symmetric and postitive semidefinite

## ~~Gaussian Discriminant Analysis (GDA)~~

- $Y\sim Bernoulli(\psi)$
- $p_{X \mid Y}(x \mid 0)=\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}\left(x-\mu_{0}\right)^{T} \Sigma^{-1}\left(x-\mu_{0}\right)\right)$
- $p_{X \mid Y}(x \mid 1)=\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}\left(x-\mu_{1}\right)^{T} \Sigma^{-1}\left(x-\mu_{1}\right)\right)$

$$
\ell\left(\psi, \mu_{0}, \mu_{1}, \Sigma\right)=\sum_{i=1}^{m} \log p_{X \mid Y}\left(x^{(i)} \mid y^{(i)} ; \mu_{0}, \mu_{1}, \Sigma\right)+\sum_{i=1}^{m} \log p_{Y}\left(y^{(i)} ; \psi\right)
$$

- Maximizing $\ell(\psi,\mu_0,\mu_1,\Sigma)$ to get the solutions:

$$
\begin{aligned}
&\psi=\frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\left\{y^{(i)}=1\right\} \\
&\mu_{0}=\sum_{i=1}^{m} \mathbf{1}\left\{y^{(i)}=0\right\} x^{(i)} / \sum_{i=1}^{m} \mathbf{1}\left\{y^{(i)}=0\right\} \\
&\mu_{1}=\sum_{i=1}^{m} \mathbf{1}\left\{y^{(i)}=1\right\} x^{(i)} / \sum_{i=1}^{m} \mathbf{1}\left\{y^{(i)}=1\right\} \\
&\Sigma=\frac{1}{m} \sum_{i=1}^{m}\left(x^{(i)}-\mu_{y^{(i)}}\right)\left(x^{(i)}-\mu_{y^{(i)}}\right)^{T}
\end{aligned}
$$

- Given a test data sample $x$, we can calculate

$$
\begin{aligned}
p_{Y \mid X}(y=1 \mid x) &=\frac{p_{X \mid Y}(x \mid 1) p_{Y}(1)}{p_{X}(x)} \\
&=\frac{p_{X \mid Y}(x \mid 1) p_{Y}(1)}{p_{X \mid Y}(x \mid 1) p_{Y}(1)+p_{X \mid Y}(x \mid 0) p_{Y}(0)} \\
&=\frac{1}{1+\frac{p_{X \mid Y}(x \mid 0) p_{Y}(0)}{p_{X \mid Y}(x \mid 1) p_{Y}(1)}}
\end{aligned}
$$

## Lagrange Multiplier

- Maximize $f(x,y)$ subject to $g(x,y)=k$
- $f (x, y)$ is maximized at point $(x0, y0)$ where they have common tangent line such that the gradient vectors are parallel, i.e., $\nabla f(x_0,y_0)=\lambda\nabla g(x_0,y_0)$.
- 我们只需要下列拉格朗日函数的极值

$$
{\displaystyle {\mathcal {L}}(x,y,\lambda )=f(x,y)-\lambda \cdot g(x,y)}
$$

- Generally,

$$
\begin{array}{ll}
\max & f(x) \\
\text { s.t. } & g_{i}(x)=k_{i}, i=1,2, \cdots, m
\end{array}
$$
where $x \in \mathbb{R}^{n}, f: \mathbb{R}^{n} \rightarrow \mathbb{R}$, and $g_{i}: \mathbb{R}^{n} \rightarrow \mathbb{R}$ for $\forall i=1, \cdots, m$
$$
{\displaystyle {\mathcal {L}}\left(x_{1},\ldots ,x_{n},\lambda _{1},\ldots ,\lambda _{k}\right)=f\left(x_{1},\ldots ,x_{n}\right)-\sum \limits _{i=1}^{k}{\lambda _{i}g_{i}\left(x_{1},\ldots ,x_{n}\right)}}
$$

## Naive Bayes

> Assumption.
>
> For $\forall j\ne j'$ $X_j$ and $X_{j'}$ are conditionally independent given $Y$, i.e., $P\left(Y=y, X_{1}=x_{1}, \cdots, X_{n}=x_{n}\right)=p(y) \prod_{j=1}^{n} p_{j}\left(x_{j} \mid y\right)$

- MLE $\ell(\Omega)=\sum_{i=1}^{m} \log p\left(y^{(i)}\right)+\sum_{i=1}^{m} \sum_{j=1}^{n} \log p_{j}\left(x_{j}^{(i)} \mid y^{(i)}\right)$

> Theorem.
> $$
> p(y)=\frac{count(y)}{m},p_j(x\mid y)=\frac{count_j(x\mid y)}{count(y)} \\
> count(y)=\sum_{i=1}^m1(y^{(i)}=y),\ count_j(x\mid y)=\sum_{i=1}^m1(y^{(i)}=y\and x_j^{(i)}=x), \\
> \forall y=0,1,\forall x=0,1.
> $$

### Classification by NB

$$
\because P(Y=y\mid X_1=\tilde{x_1},\cdots,X_n=\tilde{x_n})=\frac{P(X_1=\tilde{x_1},\cdots,X_n=\tilde{x_n}\mid Y=y)P(Y=y)}{P(X_1=\tilde{x_1},\cdots,X_n=\tilde{x_n})} \\
\therefore \arg \max _{y \in\{0,1\}}\left(p(y) \prod_{j=1}^{n} p_{j}\left(\tilde{x}_{j} \mid y\right)\right)
$$

### Laplace Smoothing

- There may exist some feature, e.g., $X_{j^∗}$ , such that $X_{j^∗}$=1 for some $x^*$ may never happen in the training data. (i.e., $p_{j^{*}}\left(x_{j^{*}}=1 \mid y\right)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y \wedge x_{j^{*}}^{(i)}=1\right)}{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)}=0, \forall y=0,1$)
- $\therefore p(y \mid x)=\frac{p(y) \prod_{j=1}^{n} p_{j}\left(x_{j} \mid y\right)}{\sum_{y} \prod_{j=1}^{n} p_{j}\left(x_{j} \mid y\right) p(y)}=\frac{0}{0}, \forall y=0,1$

> Laplace Smoothing
> $$
> \begin{aligned}
> &p(y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)+1}{m+k} \\
> &p_{j}(x \mid y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y \wedge x_{j}^{(i)}=x\right)+1}{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)+v_{j}}
> \end{aligned}
> $$
> where $k$ is number of the possible values of $y(k=2$ in our case), and $v_{j}$ is the number of the possible values of the $j$-th feature $\left(v_{j}=2\right.$ for $\forall j=1, \cdots, n$ in our case)

### Multinomial Distribution

- 多项分布，二项分布的典型例子是扔硬币，硬币正面朝上概率为p, 重复扔n次硬币，k次为正面的概率即为一个二项分布概率。把二项分布公式推广至多种状态，就得到了多项分布。

> Assumption
>
> Each training sample involves a different number of features
> $$
> x^{(i)}=\left[x_{1}^{(i)}, x_{2}^{(i)}, \cdots, x_{n_{i}}^{(i)}\right]^{\mathrm{T}}
> $$
> The $j$-th feature of $x^{(i)}$ takes a finite set of values, $x_{j}^{(i)} \in\{1,2, \cdots, v\}$

- For example, $x_j^{(i)}$ indicates the $j$-th word in the email.
- Let $p(t\mid y)=P(X_j=t\vert Y=y)$

$$
\begin{aligned}
&P\left(Y=y^{(i)}\right)=p\left(y^{(i)}\right) \\
&P\left(X=x^{(i)} \mid Y=y^{(i)}\right)=\prod_{j=1}^{n_{i}} p\left(x_{j}^{(i)} \mid y^{(i)}\right)
\end{aligned}
$$

> Problem.
> $$
> \begin{array}{ll}
> \max & \ell(\Omega)=\sum_{i=1}^{m} \sum_{j=1}^{n_{i}} \log p\left(x_{j}^{(i)} \mid y^{(i)}\right)+\sum_{i=1}^{m} \log p\left(y^{(i)}\right) \\
> \text { s.t. } & \sum_{y \in\{0,1\}} p(y)=1, \\
> & \sum_{t=1}^{v} p(t \mid y)=1, \forall y=0,1 \\
> & p(y) \geq 0, \forall y=0,1 \\
> & p(t \mid y) \geq 0, \forall t=1, \cdots, v, \forall y=0,1
> \end{array}
> $$
> 

- Solution

$$
\begin{aligned}
&p(t \mid y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right) \text { count }^{(i)}(t)}{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right) \sum_{t=1}^{v} \text { count }^{(i)}(t)} \\
&p(y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)}{m} \\
&\text { where count }^{(i)}(t)=\sum_{j=1}^{n_{i}} \mathbf{1}\left(x_{j}^{(i)}=t\right)
\end{aligned}
$$

- Laplace smoothing

$$
\begin{gathered}
\psi(y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right)+1}{m+k} \\
\psi(t \mid y)=\frac{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right) \operatorname{count}^{(i)}(t)+1}{\sum_{i=1}^{m} \mathbf{1}\left(y^{(i)}=y\right) \sum_{t=1}^{v} \operatorname{count}^{(i)}(t)+v}
\end{gathered}
$$

## Expectation Maximization (EM) Algorithm

> Def.
> $$
> \begin{aligned} 
> \ell(\theta) &= \mathrm{log}\prod_{i=1}^mp(x^{(i)};\theta) \\
> &= \sum_{i=1}^m\mathrm{log}\sum_{z^{(i)}\in\Omega}p(x^{(i)},z^{(i)};\theta)
> \end{aligned}
> $$
> where $z^{(i)}\in\Omega$ is so-called *latent variable* 

- Basic idea of EM algorithm
    - Repeatedly construct a lower-bound on $\ell$ (E-step) 
        - E-Step 主要通过观察数据和现有模型来估计参数，然后用这个估计的参数值来计算似然函数的期望值
    - Then optimize that lower-bound (M-step)
        - M-step 寻找似然函数最大化时对应的参数。由于算法会保证在每次迭代之后似然函数都会增加，所以函数最终会收敛。
- Let $Q_i$ denotes the distribution of $z$ for $i$-th sample. Thus, $\sum_{z\in\Omega}Q_i(z)=1$, $Q_i(z)\ge0$

$$
\begin{aligned}
\ell(\theta) &=\sum_{i=1}^{m} \log \sum_{z^{(i)} \in \Omega} Q_{i}\left(z^{(i)}\right) \frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)} \\
&=\sum_{i=1}^{m} \log E\left[\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{Q_{i}\left(z^{(i)}\right)}\right]
\end{aligned}
$$

- **Since $\mathrm{log}(·)$ is a concave function, according to Jensen’s inequality, we have**

$$
\because \mathrm{log}\left(E\left[\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right]\right)\ge E\left[\mathrm{log}\left(\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}\right)\right] \\
\therefore \ell(\theta)\ge\sum_{i=1}^m\sum_{z^{(i)}\in\Omega}Q_i(z^{(i)})\mathrm{log}\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}
$$

- Tighten the lower bound, the equality holds when $\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}=c$, where $c$ is a constant.
- Therefore, $\sum_{z^{(i)}\in\Omega}p(x^{(i)},z^{(i)};\theta)=c\sum_{z^{(i)}\in\Omega}Q_i(z)=c$

$$
\begin{aligned}
Q_{i}\left(z^{(i)}\right) &=\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{c} \\
&=\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{\sum_{z^{(i)} \in \Omega} p\left(x^{(i)}, z^{(i)} ; \theta\right)} \\
&=\frac{p\left(x^{(i)}, z^{(i)} ; \theta\right)}{p\left(x^{(i)} ; \theta\right)} \\
&=p\left(z^{(i)} \mid x^{(i)} ; \theta\right)
\end{aligned}
$$

> Algorithm.
>
> 1. (E-step) For each $i$, set $Q_i(z^{(i)}):=p(z^{(i)}\mid x^{(i)};\theta)$
> 2. (M-step) set $\theta:=\mathrm{arg}\mathop{\mathrm{max}}\limits_\theta\sum_i\sum_{z^{(i)}\in\Omega}Q_i(z^{(i)})\mathrm{log}\frac{p(x^{(i)},z^{(i)};\theta)}{Q_i(z^{(i)})}$

### EM in NB

#### Naive Bayes with Missing Labels

- When labels are given $\ell(\theta)=\sum_{i=1}^m\mathrm{log}\left[\ p(y^{(i)})\prod_{j=1}^np_j(x_j^{(i)}\mid y^{(i)})\right]$
- When labels are missed $\ell(\theta)=\sum_{i=1}^m\mathrm{log}\sum_{y=1}^k\left[p(y)\prod_{j=1}^np_j(x_j^{(i)}\mid y)\right]$

#### Applying EM to NB

- (E-step) For each $i=1, \cdots, m$ and $y=1, \cdots, k$ set
    $$
    Q_{i}(y)=p\left(y^{(i)}=y \mid x^{(i)}\right)=\frac{p(y) \prod_{j=1}^{n} p_{j}\left(x_{j}^{(i)} \mid y\right)}{\sum_{y^{\prime}=1}^{k} p\left(y^{\prime}\right) \prod_{j=1}^{n} p_{j}\left(x_{j}^{(i)} \mid y^{\prime}\right)}
    $$

- (M-step) Update the parameters (solved by Lagrange multiplier).
    $$
    \begin{aligned}
    &p(y)=\frac{1}{m} \sum_{i=1}^{m} Q_{i}(y), \quad \forall y \\
    &p_{j}(x \mid y)=\frac{\sum_{i: x_{j}^{(i)}=x} Q_{i}(y)}{\sum_{i=1}^{m} Q_{i}(y)}, \quad \forall x, y
    \end{aligned}
    $$