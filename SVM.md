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
\mathop{\mathrm{max}}\limits_{\omega,b}\ \frac1{\vert\vert\omega\vert\vert} \Leftrightarrow \mathop{\mathrm{min}}\limits_{\omega,b}\ \omega^T\omega \\
s.t.\ y^{(i)}(\omega^Tx^{(i)}+b)\ge1, & \forall i
\end{array}
$$

- $\mathop{\mathrm{max}}\limits_{\omega,b}\ \frac1{\vert\vert\omega\vert\vert}$ is equivalent to $\mathop{\mathrm{min}}\limits_{\omega,b}\ \omega^T\omega$ 

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



#### Complementary Slackness

