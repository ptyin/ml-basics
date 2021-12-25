# Logistic Regression

- Classification problem is similar to predict only a small number of discrete values instead of continuous values.

## 1. Logistic Function (Sigmoid Function)

$$
g(z) = \frac1{1+e^{-z}}
$$

### Properties

- Bound: $g(z)\in(0,1)$
- Symmetric: $1-g(z)=g(-z)$
- Gradient: $g'(z)=g(z)(1-g(z))$

## 2. Logistic Regression

$$
h_\theta(x)=g(\theta^Tx)=\frac1{(1+e^{-\theta^Tx})}
$$

- $\theta^Tx$ is called score.
- $Pr(Y=1\vert X=x;\theta)=h(\theta)=\frac1{1+e^{-\theta^Tx}}$
- $Pr(Y=0\vert X=x;\theta)=1-h_\theta(x)=\frac1{1+e^{\theta^Tx}}$

### Decision Boundary

- $\begin{array}{c}Pr(Y=1\vert X=x;\theta)=Pr(Y=0\vert X=x;\theta)&\Rightarrow&\theta^Tx=0\end{array}$
- $\therefore$ The decision boundary is a linear hyperplane.
- **The score $\theta^Tx$ is also a measure of distance $x$ from the hyperplane.**

### Probability Mass Function (概率质量函数)

>  概率质量函数和概率密度函数不同之处在于：概率质量函数是对离散随机变量定义的，本身代表该值的概率；概率密度函数本身不是概率，只有对连续随机变量的概率密度函数在某区间内进行积分后才是概率。

$$
p(y\vert x;\theta)=Pr(Y=y\vert X=x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}, \text{where}\ y\in\{0,1\}
$$

$$
p(y\vert x;\theta)=\frac1{1+exp(-y\theta^Tx)},\text{where}\ y\in\{-1,1\}\ \text{instead of}\ y\in\{0,1\},
$$

- **Maximize** the log likelihood $\begin{aligned}L(\theta)&=\prod_{i=1}^mp(y\vert x;\theta) \\\end{aligned}$
- $l(\theta)=\mathrm{log}L(\theta)=\sum_{i=1}^m(y^{(i)}\mathrm{log}h(x^{(i)})+(1-y^{(i)})\mathrm{log}(1-h(x^{(i)})))$, still assume $y\in\{-1,1\}$

$$
\frac{\partial}{\partial \theta_j}l(\theta)=\sum_{i=1}^m\frac{y^{(i)}-h_\theta(x^{(i)})}{h_\theta(x^{(i)})(1-h_\theta(x^{(i)}))}\cdot\frac{\partial h_\theta(x^{(i)})}{\partial \theta_j}=\sum_{i=1}^m(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}
$$

## 3. Newton's Method

### Properties

- Highly dependent on initial guess 
- Quadratic convergence once it is sufficiently close to $x^*$ 
- If $f' = 0$, only has linear convergence 
- Is not guaranteed to convergence at all, depending on function or initial guess

### Update

$$
x\leftarrow x-\frac{f'(x)}{f''(x)}
$$

- For $l: \mathbb{R}^n\rightarrow \mathbb{R}$,

$$
\theta\leftarrow\theta-H^{-1}\nabla_\theta l(\theta),\ \text{where}\ H_{i,j}=\frac{\partial^2l(\theta)}{\partial\theta_i\partial\theta_j}
$$

### Multiclass Classification

- Transformation to binary 
    - One-vs.-rest (OvR, train a single classifier per class, with the samples of that class as positive samples and all other samples as negative ones)
        - $y^*=\mathop{\mathrm{argmax}}\limits_kf_k(x)$
        - $f_k(x)$ implies hight robability that $x$ is in class $k$.
    - One-vs.-one (OvO, to train $K(K −1)/2$ binary classifiers)
        - $y^*=\mathop{\mathrm{argmax}}\limits_s(\sum_tf_{s,t}(x))$
        - $f_{s,t}(x)$ implies that label $s$ has higher probability than label $t$.
- Extension from binary 
- Hierarchical classification

## 4. Softmax Regression

$$
\begin{aligned}
l(\theta)&=\sum_{i=1}^m\mathrm{log}p(y^{(i)}\vert x^{(i)};\theta) \\
&=\sum_{i=1}^m\mathrm{log}\prod_{k=1}^K\left(\frac{exp(\theta^{(k)^T}x^{(i)})}{\sum_{k'=1}^Kexp(\theta^{(k')^T}x^{(i)})}\right)^{\mathbb{I}(y^{(i)}=k)}
\end{aligned}
$$

- where $\mathbb{I}:\{True, False\}\rightarrow\{0,1\}$ is an indicator function.