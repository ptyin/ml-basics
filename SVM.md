# Support Vector Machine

## Primal Form

- A hyperplane that separates a n-dimensional space into two half-spaces.
- Prediction rule: $y=sign(\omega^Tx+b)$
- Margin
  - Geometric margin ($\ge0$): $\gamma^{(i)}=y^{(i)}((\frac{\omega}{\vert\vert\omega\vert\vert})^Tx^{(i)}+\frac{b}{\vert\vert\omega\vert\vert})$
  - Whole training set, the margin is $\gamma=\mathop{\mathrm{min}}\limits_i\gamma^{(i)}$
- Goal: Learn $\omega$ and $b$ that achieves the maximum margin $\mathop{\mathrm{max}}\limits_{\omega,b}\ \mathop{\mathrm{min}}\limits_i\gamma^{(i)}$

$$
\begin{array}{l}
\mathop{\mathrm{max}}\limits_{\gamma,\omega,b}\ \gamma \\
s.t.\ y^{(i)}(\omega^Tx^{(i)}+b)\ge\gamma\vert\vert\omega\vert\vert, & \forall i
\end{array}
$$

- Scaling $(\omega,b)$ such that $\gamma\vert\vert\omega\vert\vert=1$, the problem becomes

$$
\begin{array}{l}
\mathop{\mathrm{max}}\limits_{\omega,b}\ \frac1{\vert\vert\omega\vert\vert} \Leftrightarrow \mathop{\mathrm{min}}\limits_{\omega,b}\ \omega^T\omega \Leftrightarrow \min\limits_{\omega,b}\frac12\vert\vert\omega\vert\vert^2 \\
s.t.\ y^{(i)}(\omega^Tx^{(i)}+b)\ge1, & \forall i
\end{array}
$$

- $\mathop{\mathrm{max}}\limits_{\omega,b}\ \frac1{\vert\vert\omega\vert\vert}$ is equivalent to $\mathop{\mathrm{min}}\limits_{\omega,b}\ \omega^T\omega$ 

> Def. The primal problem
> $$
> \begin{array}{l}
> \min\limits_{\omega,b}\frac12\vert\vert\omega\vert\vert^2 \\
> s.t.\ y^{(i)}(\omega^Tx^{(i)}+b)\ge1, & \forall i
> \end{array}
> $$

## Duality of  SVM

-  The Lagrangian problem for SVM 

$$
\min_{\omega,b,\alpha}\mathcal L(\omega, b, \alpha)=\frac12||\omega||^2+\sum_{i=1}^m\alpha_i(y^{(i)}(\omega^Tx^{(i)}+b)-1)
$$

- The Lagrangian dual problem for SVM is $\max_\alpha\mathcal{G}(\alpha)=\inf_{\omega,b}\mathcal{L}(\omega,b,\alpha)$

$$
\begin{array}{ll}
\min\limits_{\alpha} & \mathcal{G}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left(x^{(i)}\right)^{T} x^{(j)} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 \\
& \alpha_{i} \geq 0 \quad \forall i
\end{array}
$$

- Proof.
  - $\frac\part{\part\omega}\mathcal{L}(\omega,b,\alpha)=\omega-\sum_{i=1}^m\alpha_iy^{(i)}x^{(i)}=0$ and $\frac\part{\part b}\mathcal{L}(\omega,b,\alpha)=\sum_{i=1}^m\alpha_iy^{(i)}=0$
  - $\mathcal{L}$ is a convex function.
- It suffices **Slarter's Condition** (why?). Thus, the problem can be solved by QP solver (MATLAB, ...)
- Since we have the solution $\alpha^*$ for the dual problem, we can calculate the solution for the primal problem.

$$
\omega^*=\sum_{i=1}^m\alpha^*y^{(i)}x^{(i)}\\
b^*=y^{(i)}-{\omega^*}^Tx^{(i)},\ \text{if}\ \alpha^*>0
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
- Each kernel $K$ has an associated feature mapping from input to feature space $\phi: \mathcal{X}\rightarrow\mathcal{F} $.
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
- For SVM

$$
\begin{array}{ll}
\max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left(x^{(i)}\right)^{T} x^{(j)} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 \\
& \alpha_{i} \geq 0 \quad \forall i
\end{array}
\Rightarrow 
\begin{array}{ll}
\max _{\alpha} & \sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}K_{i,j} \\
\text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 \\
& \alpha_{i} \geq 0 \quad \forall i
\end{array}
$$

