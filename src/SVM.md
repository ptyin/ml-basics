# Support Vector Machine

## Primal Form

- A hyperplane that separates a n-dimensional space into two half-spaces.
- Prediction rule: $y=sign(\omega^Tx+b)$
- Margin
  - Geometric margin ($\ge0$): $\gamma^{(i)}=y^{(i)}((\frac{\omega}{\vert\vert\omega\vert\vert})^Tx^{(i)}+\frac{b}{\vert\vert\omega\vert\vert})$
  - Whole training set, the margin is $\gamma=\mathop{\mathrm{min}}\limits_i\gamma^{(i)}$
- Goal: Learn $\omega$ and $b$ that achieves the maximum margin $\mathop{\mathrm{max}}\limits_{\omega,b}\ \mathop{\mathrm{min}}\limits_i\gamma^{(i)}$

$$
\begin{array}{ll}
\mathop{\mathrm{max}}\limits_{\gamma,\omega,b}\ \gamma \\
s.t.\ y^{(i)}(\omega^Tx^{(i)}+b)\ge\gamma\vert\vert\omega\vert\vert, & \forall i
\end{array}
$$

- Scaling $(\omega,b)$ such that $\gamma\vert\vert\omega\vert\vert=1$, i.e. $\omega'=\frac\omega{\gamma||\omega||}$ and $b'=\frac b{\gamma||\omega||}$.
- $\therefore||\omega'||=\frac{||\omega||}{\gamma||\omega||}=\frac1\gamma,\ y^{(i)}(\omega'^Tx^{(i)}+b')\ge1$
- the problem becomes

$$
\begin{array}{ll}
\mathop{\mathrm{max}}\limits_{\omega,b}\ \frac1{\vert\vert\omega\vert\vert} \Leftrightarrow \mathop{\mathrm{min}}\limits_{\omega,b}\ \omega^T\omega \Leftrightarrow \min\limits_{\omega,b}\frac12\vert\vert\omega\vert\vert^2 \\
s.t.\ y^{(i)}(\omega^Tx^{(i)}+b)\ge1, & \forall i
\end{array}
$$

- $\mathop{\mathrm{max}}\limits_{\omega,b}\ \frac1{\vert\vert\omega\vert\vert}$ is equivalent to $\mathop{\mathrm{min}}\limits_{\omega,b}\ \omega^T\omega$ 

> Def. The primal problem
> $$
> \begin{array}{ll}
> \min\limits_{\omega,b}\frac12\vert\vert\omega\vert\vert^2 \\
> s.t.\ y^{(i)}(\omega^Tx^{(i)}+b)\ge1, & \forall i
> \end{array}
> $$

## Duality of  SVM

>  Preliminaries should be mastered in chapter Optimization of appendix.

-  The Lagrangian problem for SVM 

$$
\min_{\omega,b}\max_{\alpha}\mathcal L(\omega, b, \alpha)=\frac12||\omega||^2+\sum_{i=1}^m\alpha_i(1-y^{(i)}(\omega^Tx^{(i)}+b))
$$

- The Lagrangian dual problem for SVM is $\max_\alpha\mathcal{G}(\alpha)=\max_\alpha\min_{\omega,b}\mathcal{L}(\omega,b,\alpha)$

$$
\begin{array}{ll}
\max\limits_{\alpha} & \mathcal{G}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left(x^{(i)}\right)^{T} x^{(j)} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 \\
& \alpha_{i} \geq 0 \quad \forall i
\end{array}
$$

- Proof.
  - $\frac\partial{\partial\omega}\mathcal{L}(\omega,b,\alpha)=\omega-\sum_{i=1}^m\alpha_iy^{(i)}x^{(i)}=0$ and $\frac\partial{\partial b}\mathcal{L}(\omega,b,\alpha)=\sum_{i=1}^m\alpha_iy^{(i)}=0$
  - $\mathcal{L}$ is a convex function.
- It suffices **Slarter's Condition**. Thus, the problem can be solved by QP solver (MATLAB, ...)
- Since we have the solution $\alpha^*$ for the dual problem, we can calculate the solution for the primal problem.

$$
\begin{array}{ll}
\omega^*=\sum_{i=1}^m\alpha^*y^{(i)}x^{(i)} & \\
b^*=y^{(i)}-{\omega^*}^Tx^{(i)} & \text{if}\ \alpha_i^*>0
\end{array}
$$

- For robustness, the optimal value for $b$ is calculated by taking the averages across all $b^*$

$$
b^{*}=\frac{\sum_{i: \alpha_{i}^{*}>0}\left(y^{(i)}-\omega^{* T} x^{(i)}\right)}{\sum_{i=1}^{m} \mathbf{1}\left(\alpha_{i}^{*}>0\right)}
$$

- However, according to **Complementary Slackness**, $\alpha_{i}^{*}\left[1-y^{(i)}\left(\omega^{* T} x^{(i)}+b^{*}\right)\right]=0$.
- $\alpha_i^*$ is non-zero only if $x^{(i)}$ lies on the margin, i.e., $y^{(i)}\left(\omega^{* T} x^{(i)}+b^{*}\right)=1$. (**Support Vector**, $\mathcal{S}$).

$$
\therefore \omega=\sum_{s\in \mathcal{S}}\alpha_sy^{(s)}x^{(s)}
$$

## Kernel

- Basic idea: mapping data to higher dimensions where it exhibits linear patterns.
- Each kernel $K$ has an associated feature mapping $\phi: \mathcal{X}\rightarrow\mathcal{F}$ from input to feature space.
    - e.g., quadratic mapping $\phi: x\rightarrow\{x_1^2,x_2^2,\cdots,x_1x_2,\cdots,x_1x_n,\cdots,x_{n-1}x_n\}$
- Kernel $K(x,z)=\phi(x)^T\phi(z),\ K:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$ **takes two inputs and gives their similarity** in $\mathcal{F}$.

> Thereom. **Mercer's Condition**.
>
> For $K$ to be a kernel function if $K$ is a positive definite function.
> $$
> \int\int f(x)K(x,z)f(z)dxdz>0 \\
> \forall f,\ s.t.\ \int_{-\infty}^{\infty}f^2(x)dx<\infty
> $$

- Composing rules
    - Direct sum $K(x,z)=K_1(x,z)+K_2(x,z)$
    - Scalar product $K(x,z)=\alpha K_1(x,z)$
    - Direct product $K(x,z)=K_1(x,z)K_2(x,z)$

> Def. Kernel Matrix.
> $$
> K_{i,j}=K(x^{(i)},x^{(j)})=\phi(x^{(i)})^T\phi(x^{(j)})
> $$

### Example Kernel

- Linear (trivial) Kernal $K(x, z)=x^{T} z$
- Quadratic Kernel $K(x, z)=\left(x^{T} z\right)^{2} \text { or }\left(1+x^{T} z\right)^{2}$
- Polynomial Kernel (of degree $d$ ) $K(x, z)=\left(x^{T} z\right)^{d} \text { or }\left(1+x^{T} z\right)^{d}$
- Gaussian Kernel $K(x, z)=\exp \left(-\frac{\|x-z\|^{2}}{2 \sigma^{2}}\right)$
- Sigmoid Kernel $K(x, z)=\tanh \left(\alpha x^{T}+c\right)$

### Applicable Algorithm

- SVM, linear regression, etc.
- K-means, PCA, etc.

### Kernelized SVM

- Optimization problem

$$
\begin{array}{c}
\begin{array}{ll}
\max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left(x^{(i)}\right)^{T} x^{(j)} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 \\
& \alpha_{i} \geq 0 \quad \forall i
\end{array} \\
\\ \Downarrow \\ \\
\begin{array}{ll}
\max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}K_{i,j} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 \\
& \alpha_{i} \geq 0 \quad \forall i
\end{array}
\end{array}
$$

- Solution

$$
\begin{aligned}
\omega^{*} &=\sum_{i: \alpha_{i}^{*}>0} \alpha_{i}^{*} y^{(i)} \phi\left(x^{(i)}\right) \\
b^{*} &=y^{(i)}-\omega^{* T} \phi\left(x^{(i)}\right) \\
&=y^{(i)}-\sum_{j: \alpha_{j}^{*}>0} \alpha_{j}^{*} y^{(j)} \phi^{T}\left(x^{(j)}\right) \phi\left(x^{(i)}\right) \\
&=y^{(i)}-\sum_{j: \alpha_{j}^{*}>0} \alpha_{j}^{*} y^{(j)} K_{i j}
\end{aligned}
$$

- Prediction

$$
\begin{aligned}
y &=\operatorname{sign}\left(\sum_{i: \alpha_{i}^{*}>0} \alpha_{i}^{*} y^{(i)} \phi\left(x^{(i)}\right)^{T} \phi(x)+b^{*}\right) \\
&=\operatorname{sign}\left(\sum_{i: \alpha_{i}^{*}>0} \alpha_{i}^{*} y^{(i)} K\left(x^{(i)}, x\right)+b^{*}\right)
\end{aligned}
$$

- Kenerlized SVM needs to compute kernel when testing, whereas computed $\omega^*$ and $b^* $ are enough in the unkenerlized version.

## Soft Margin

- Relax the constraints from $y^{(i)}(\omega^Tx^{(i)}+b)\ge1$ to $y^{(i)}(\omega^Tx^{(i)}+b)\ge1-\xi_i$
- $\xi_i\ge0$ is called slack variable

> Def. **Soft Margin SVM**
> $$
> \begin{array}{lll}
> \min_{\omega,b,\xi}&\frac12||\omega||^2+C\sum_{i=1}^m\xi_i& \\
> s.t. & y^{(i)}(\omega^Tx^{(i)}+b)\ge1-\xi_i, &\forall i=1,\cdots,m \\
> & \xi_i\ge0, &\forall i=1,\cdots,m
> \end{array}
> $$

- $C$ is a hyper-parameter that controls the relative weighting between $\frac12||\omega||^2$ for **larger margins** and $\sum_{i=1}^m\xi_i$ for **fewer misclassified examples**.
- Lagrangian function

$$
\mathcal{L}(\omega, b, \xi, \alpha, r)=\frac{1}{2} \omega^{T} \omega+C \sum_{i=1}^{m} \xi_{i}-\sum_{i=1}^{m} \alpha_{i}\left[y^{(i)}\left(\omega^{T} x^{(i)}+b\right)-1+\xi_{i}\right]-\sum_{i=1}^{m} r_{i} \xi_{i}
$$

- KKT conditions (the optimal values of $\omega, b, \xi, \alpha$, and $r$ should satisfy the following conditions)

    - $\nabla_{\omega} \mathcal{L}(\omega, b, \xi, \alpha, r)=0 \Rightarrow \omega^{*}=\sum_{i=1}^{m} \alpha_{i}^{*} y^{(i)} x^{(i)}$

    - $\nabla_{b} \mathcal{L}(\omega, b, \xi, \alpha, r)=0 \Rightarrow \sum_{i=1}^{m} \alpha_{i}^{*} y^{(i)}=0$

    - $\nabla_{\xi_{i}} \mathcal{L}(\omega, b, \xi, \alpha, r)=0 \Rightarrow \alpha_{i}^{*}+r_{i}^{*}=C$, for $\forall i$

    - $\alpha_{i}^{*}, r_{i}^{*}, \xi_{i}^{*} \geq 0$, for $\forall i$

    - $y^{(i)}\left(\omega^{* T} x^{(i)}+b^{*}\right)+\xi_{i}^{*}-1 \geq 0$, for $\forall i$

    - $\alpha_{i}^{*}\left(y^{(i)}\left(\omega^{*} x^{(i)}+b^{*}\right)+\xi_{i}^{*}-1\right)=0$, for $\forall i$

    - $r_{i}^{*} \xi_{i}^{*}=0$, for $\forall i$

- Dual problem

$$
\begin{array}{ll}
\max _{\alpha} & \mathcal{J}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}<x^{(i)}, x^{(j)}> \\
\text { s.t. } \quad & 0 \leq \alpha_{i} \leq C, \quad \forall i=1, \cdots, m \\
& \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0
\end{array}
$$

- Solution
    - $\omega^*=\sum_{i=1}^m\alpha_i^*y^{(i)}x^{(i)}$
    - $b^*=\frac{\sum_{i:0<{\alpha_i^*}<C}(y^{(i)}-{\omega^*}^Tx^{(i)})}{\sum_{i=1}^m1(0<\alpha_i^*<C)}$
- Proof. 

$$
\begin{array}{l}\because r_i^*\xi_i^*=0\Leftrightarrow (C-\alpha_i^*)\xi_i^*=0 \\ \therefore \forall i, \alpha_i^*\ne C\Rightarrow\xi_i=0\Rightarrow \alpha_i(y^{(i)}({\omega^*}^Tx^{(i)}+b^*)-1)=0 \\ \therefore \forall i, \alpha_i^*\in(0,C)\Rightarrow y^{(i)}({\omega^*}^Tx^{(i)}+b^*)=1 \Rightarrow {\omega^*}^Tx^{(i)}+b^*=y^{(i)}\end{array}
$$

- Corollaries of KKT conditions for soft-margin SVM
    - When $\alpha_{i}^{*}=0, y^{(i)}\left(\omega^{* T} x^{(i)}+b^{*}\right) \geq 1$, correctly classified.
    - When $\alpha_{i}^{*}=C, y^{(i)}\left(\omega^{* T} x^{(i)}+b^{*}\right) \leq 1$, misclassified. 
    - When $0<\alpha_{i}^{*}<C, y^{(i)}\left(\omega^{* T} x^{(i)}+b^{*}\right)=1$, support vector.

