# Regularization

## 1. Overfitting

- Underfitting, or high bias, is when the form of our hypothesis function h maps **poorly** to the trend of the data.
- Overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data.

### Addressing

- Reduce the number of features (manually select, model selection).
- Regularization (keep all the features, but reduce the magnitude of parameters).

## 2. Regularized Linear Regression

$$
\mathop{\mathrm{min}}\limits_\theta\frac1{2m}
\left[
\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^n\theta_j^2
\right],\ \text{where}\ h_\theta(x)=\theta^Tx
$$

- Normal equation

$$
\theta=(X^TX+\lambda\cdot L)^{-1}X^Ty \\
\text{where}\ L=\left[\begin{matrix}
0 &   &   & \\
  & 1 &   & \\
  &   & \ddots & \\
  &   &   & 1
\end{matrix}\right]
$$

> Proof.
> $$
> \left\{
> \begin{array}{lll}
> \frac\partial{\partial\theta_j}J(\theta)=\frac1m\sum_{k=1}^m(\theta^Tx^{(k)}-y^{(k)})x_j^{(k)}  && (j=0)\\
> \frac\partial{\partial\theta_j}J(\theta)=\frac1m\sum_{k=1}^m(\theta^Tx^{(k)}-y^{(k)})x_j^{(k)}+\frac\lambda m\theta_j  && (j\in N^+)
> \end{array}
> \right.
> \\ \Rightarrow
> \nabla_\theta J(\theta)=\frac1 m(X^TX\theta-X^T y)+\frac\lambda mL\theta
> $$

## 3. Regularized Logistic Regression

$$
\mathop{\mathrm{min}}\limits_\theta
\left[
-\frac1m\sum_{i=1}^m
\left(
y^{(i)}\mathrm{log}h_\theta(x^{(i)})+(1-y^{(i)})\mathrm{log}(1-h_\theta(x^{(i)}))
\right) + 
\frac\lambda{2m}\sum_{j=1}^n\theta_j^2
\right]
$$

## 4. MLE & MAP

### Preliminaries

- Assume data are generated via $d\sim p(d;\theta)$
- $D=\{d^{(i)}\}_{i=1,2,\cdots,m}$, where $d^{(i)}$ is i.i.d. (independent of others and same distribution).
- **Goal**: Estimate parameter $\theta$ that best models the data.

### Maximum Likelihood Estimation (MLE)

- Likelihood: $L(\theta)=p(D;\theta)=\prod_{i=1}^mp(d^{(i)};\theta)$
- MLE typically maximizes the **log-likelihood** $l(\theta)$.
- $\theta_{MLE}=\mathop{\mathrm{arg}}\mathop{\mathrm{max}}\limits_\theta\ \sum_{i=1}^m\mathrm{log}p(d^{(i)};\theta)$

### Maximum-a-Posteriori Estimation (MAP)

- Posterior probability of $\theta$ is $p(\theta\vert D)=\frac{p(\theta)p(D\vert\theta)}{p(D)}$
- $p(\theta)$ is prior prbability of $\theta$, where $p(D)$ is probability of the data.
- MAP usually maximizes the **log** of the posteriori probability
- $\theta_{MAP}=\mathop{\mathrm{arg}}\mathop{\mathrm{max}}\limits_\theta\ \left(\mathrm{log}p(\theta)+\sum_{i=1}^m\mathrm{log}p(d^{(i)}\vert\theta)\right)$

### Linear Regression

#### 1. MLE

- Suppose $y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$, where $\epsilon\sim\mathcal{N}(0,\sigma^2)$ 
    - Normal Distribution $p(x ; \mu, \sigma)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)$

- $p(d^{(i)};\theta)={\frac {1}{\sigma {\sqrt {2\pi }}}}\;\exp({-{\frac 1{2\sigma ^{2}}(y^{(i)}-\theta^Tx^{(i)})^2}})\Rightarrow\log p(d^{(i)};\theta)=\log\frac1{\sigma\sqrt{2\pi}}-\frac1{2\sigma^2}(y^{(i)}-\theta x^{(i)})^2$
- $\theta_{MLE}=\mathrm{arg}\mathop{\mathrm{min}}\limits_\theta\sum_{i=1}^m(y^{(i)}-\theta^Tx^{(i)})^2$

#### 2. MAP

- Suppose $\epsilon\sim\mathcal N(0, \sigma^2), \theta\sim\mathcal{N}(0,\lambda^2I)$
    - Multivariate normal distribution $p(x ; \mu, \Sigma)=\frac{1}{(2 \pi)^{n / 2}|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)$
    - where $\mu\in\mathbb{R}^n$, $\Sigma\in\mathbb{R}^{n\times n}$ is symmetric and postitive semidefinite

- $p(\theta)=\frac1{(\sqrt{2\pi}\lambda)^n}\exp(-\frac1{2\lambda^2}\theta^T\theta)\Rightarrow\log\ p(\theta)=n\mathrm{log}\frac1{\sqrt{2\pi}\lambda}-\frac{\theta^T\theta}{2\lambda^2}$
- $\theta_{MAP}=\mathrm{arg}\mathop{\mathrm{min}}\limits_\theta\left\{\sum_{i=1}^m(y^{(i)}-\theta^Tx^{(i)})^2+\frac{\theta^T\theta}{2\lambda^2}\right\}$

#### 3. MLE vs MAP

- MLE (unregularized solution) vs MAP (regularized solution)
- The prior distribution acts as a regularizer in MAP estimation

### Logistic Regression

Similar conclusion as above. 