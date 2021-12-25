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

## Convex Optimization

$$
\begin{array}{ll}
\min_\omega & f(\omega) \\
s.t. & g_i(\omega)\le0 & i=1,\cdots,k \\
& h_j(\omega)=0 & j=1,\cdots,l
\end{array}
$$

- *Lagrangian* (Lagrange function) of the above optimization.

$$
\mathcal{L}(\omega,\alpha,\beta)=f(\omega)+\sum_{i=1}^k\alpha_ig_i(\omega)+\sum_{j=1}^l\beta_jh_j(\omega)
$$

- $\alpha_i,\beta_j$ are Lagrange multiplier, $\alpha_i\ge0$.
- Lagrange dual function $\mathcal{G}:\mathbb{R}^k\times\mathbb{R}^l\rightarrow\mathbb{R}$ as an infimum (下确界) of $\mathcal{L}$ with respect to $\omega$.

$$
\begin{aligned}
\mathcal{G}(\alpha, \beta) &=\inf _{\omega \in \mathcal{D}} \mathcal{L}(\omega, \alpha, \beta) \\
&=\inf _{\omega \in \mathcal{D}}\left(f(\omega)+\sum_{i=1}^{k} \alpha_{i} g_{i}(\omega)+\sum_{j=1}^{l} \beta_{j} h_{j}(\omega)\right)
\end{aligned}
$$

> Theorem. **Lower Bounds Property**
>
> If $\alpha\ge0$, then $\mathcal{G}(\alpha,\beta)\le p^*$ where $p^*$ is the optimal value of the (original) primal problem.

- The Lagrange dual function provides a non-trivial lower bound to the primal optimization problem.
- The *Lagrange dual problem* with respect to the primal problem. The optimal value is $d^*$, and $d^*\le p^*$.

$$
\begin{array}{ll}
\max_{\alpha,\beta} & \mathcal{G}(\alpha,\beta) \\
s.t. & \alpha\ge0 & \forall i=1,\cdots,k
\end{array}
$$

### Karush-Kuhn-Tucker (KKT) Conditions

- Let $\omega^*$ be a primal optimal point and $(\alpha^*, \beta^*)$ be a dual optimal solution.

> Theorem. **Complementary Slackness** (互补松弛)
>
> If strong duality holds, then $\alpha_i^*g_i(\omega^*)=0,\ \forall i=1,2,\cdots,k.$

- Since $\omega^*$ is the minimizer of $\mathcal{L}(\omega,\alpha^*,\beta^*)$ over $\omega$.

> Def. **Stationarity Condition**
> $$
> \nabla f\left(\omega^{*}\right)+\sum_{i=1}^{k} \alpha_{i}^{*} \nabla g_{i}\left(\omega^{*}\right)+\sum_{j=1}^{l} \beta_{j}^{*} \nabla h_{j}\left(\omega^{*}\right)=0
> $$

- The primal feasibility conditions and the dual feasibility condition holds:

$$
\begin{aligned}
&g_{i}\left(\omega^{*}\right) \leq 0, \forall i=1, \cdots, k \\
&h_{j}\left(\omega^{*}\right)=0, \forall j=1, \cdots, l \\
&\alpha_{i}^{*} \geq 0, \forall i=1, \cdots, k
\end{aligned}
$$

- Altogether, these conditions formulate the KKT conditions

$$
\begin{aligned}
& \alpha_i^*g_i(\omega^*)=0,\ \forall i=1,2,\cdots,k. \\
& \nabla f\left(\omega^{*}\right)+\sum_{i=1}^{k} \alpha_{i}^{*} \nabla g_{i}\left(\omega^{*}\right)+\sum_{j=1}^{l} \beta_{j}^{*} \nabla h_{j}\left(\omega^{*}\right)=0 \\
&g_{i}\left(\omega^{*}\right) \leq 0, \forall i=1, \cdots, k \\
&h_{j}\left(\omega^{*}\right)=0, \forall j=1, \cdots, l \\
&\alpha_{i}^{*} \geq 0, \forall i=1, \cdots, k
\end{aligned}
$$

### Convex Optimization

- If objective function $f(\omega)$ and inequality constraints $g_i(\omega)$ are convex, and the equality constraints $h_j(\omega)$ are affine functions. A convex optimization problem can be represented by

$$
\begin{array}{ll}
\min _{\omega} & f(w) \\
\text { s.t. } & g_{i}(w) \leq 0, i=1, \cdots, k \\
& A w-b=0
\end{array}
$$

- where, $A\in\mathbb{R}^{l\times n}$ and $b\in\mathbb{R}^{l}$.

> Theorem. **Slarter's Condition** (one of so-called *constraint qualification*, a sufficient condition)
>
>  Strong duality holds for a convex problem if it is strictly feasible, i.e., 
> $$
> \exists \omega \in \operatorname{relint} \mathcal{D}: g_{i}(\omega)<0, i=1, \cdots, m, A w=b
> $$
> relint (relative interior, 相对内部) 是指拓扑线性空间中的集合在相对意义下的内部

## Duality of  SVM

> Theorem. Dual optimization problem of SVM
> $$
> \begin{array}{ll}
> \max _{\alpha} & \mathcal{G}(\alpha)=\sum_{i=1}^{m} \alpha_{i}-\frac{1}{2} \sum_{i, j=1}^{m} y^{(i)} y^{(j)} \alpha_{i} \alpha_{j}\left(x^{(i)}\right)^{T} x^{(j)} \\
> \text { s.t. } & \sum_{i=1}^{m} \alpha_{i} y^{(i)}=0 \\
> & \alpha_{i} \geq 0 \quad \forall i
> \end{array}
> $$

- Proof.
  - $\mathcal{L}(\omega,b,\alpha)=\frac12\vert\vert\omega\vert\vert^2-\sum_{i=1}^m\alpha_i(y^{(i)}(\omega^Tx^{(i)}+b)-1)$
  - $\frac\part{\part\omega}\mathcal{L}(\omega,b,\alpha)=\omega-\sum_{i=1}^m\alpha_iy^{(i)}x^{(i)}=0$ and $\frac\part{\part b}\mathcal{L}(\omega,b,\alpha)=\sum_{i=1}^m\alpha_iy^{(i)}=0$
  - Reduction ... (omitted).
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

