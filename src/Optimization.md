# Optimization

## Lagrange Multiplier

>  Def. **Lagrange Multiplier**.
>
> Use to convert a optimization problem with constraints to one without constraints.
> $$
> \begin{array}{l}
> \begin{array}{lll}
> \min_\omega & f(\omega) \\
> s.t. & g_i(\omega)\le0 & i=1,\cdots,k \\
> & h_j(\omega)=0 & j=1,\cdots,l
> \end{array}
> \Rightarrow \\
> \min_\limits{\omega,\alpha,\beta}\mathcal{L}(\omega,\alpha,\beta)=f(\omega)+\sum_{i=1}^k\alpha_ig_i(\omega)+\sum_{j=1}^l\beta_jh_j(\omega)
> \end{array}
> $$
> $\alpha_i,\beta_j$ are so-called Lagrange multiplier, $\alpha_i\ge0$.

## Lagrange Duality

- The solution to the dual problem provides a lower bound to the solution of the primal problem.

> Def. Lagrange Dual Function
> $$
> \begin{aligned}
> \mathcal{G}(\alpha, \beta) &=\inf _{\omega \in \mathcal{D}} \mathcal{L}(\omega, \alpha, \beta) \\
> &=\inf _{\omega \in \mathcal{D}}\left(f(\omega)+\sum_{i=1}^{k} \alpha_{i} g_{i}(\omega)+\sum_{j=1}^{l} \beta_{j} h_{j}(\omega)\right)
> \end{aligned}
> $$

- The *Lagrange dual problem* with respect to the primal problem. The optimal value is $d^*$, and $d^*\le p^*$.

$$
\begin{array}{ll}
\max_{\alpha,\beta} & \mathcal{G}(\alpha,\beta) \\
s.t. & \alpha\ge0 & \forall i=1,\cdots,k
\end{array}
$$

## Karush-Kuhn-Tucker (KKT) Conditions

- Let $\omega^*$ be a primal optimal point and $(\alpha^*, \beta^*)$ be a dual optimal solution.

> Def. **KKT Conditions**
>
> - Stationarity: $\nabla f\left(\omega^{*}\right)+\sum_{i=1}^{k} \alpha_{i}^{*} \nabla g_{i}\left(\omega^{*}\right)+\sum_{j=1}^{l} \beta_{j}^{*} \nabla h_{j}\left(\omega^{*}\right)=0$
> - Primal feasibility:  $g_{i}\left(\omega^{*}\right) \leq 0, \forall i=1, \cdots, k \\ h_{j}\left(\omega^{*}\right)=0, \forall j=1, \cdots, l$
> - Dual feasibility: $\alpha_{i}^{*} \geq 0, \forall i=1, \cdots, k$
> - Complementary slackness: $\alpha_i^*g_i(\omega^*)=0,\ \forall i=1,2,\cdots,k$



- Proof. **Stationarity Condition**
    - $\omega^*$ is the minimizer of $\mathcal{L}(\omega,\alpha^*,\beta^*)$ over $\omega$. Thus, $\nabla\mathcal L=0$

- The primal feasibility conditions holds natrually.
- Proof. **Dual Feasibility**
    - If $\alpha\ge0$ and $\tilde\omega$ is feasible, then
    - $f(\tilde{\omega}) \geq \mathcal{L}(\tilde{\omega}, \alpha, \beta) \geq \mathcal{G}(\alpha, \beta)=\inf _{\omega \in \mathcal{D}} \mathcal{L}(\omega, \alpha, \beta)$
- Proof. **Complementary Slackness** (互补松弛)
    - If strong duality holds, then
    - $\begin{aligned}f(\omega^*)&=\mathcal{G}(\alpha^*,\beta^*)\\ &\le f(\omega^*)+\sum_{i=1}^k\alpha^*g_i(\omega^*)+\sum_{j=1}^l\beta_j^*h_J(\omega^*) \\&\le f(\omega^*)\end{aligned}$
    - $\therefore \sum_{i=1}^k\alpha_i^*g_i(\omega^*)=0$
    - Since each term is nonpositive, $\alpha_i^*g_i(\omega)=0$.

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
> Strong duality holds for a convex problem if it is strictly feasible, i.e., 
> $$
> \exists \omega \in \operatorname{relint} \mathcal{D}: g_{i}(\omega)<0, i=1, \cdots, m, A w=b
> $$
> relint (relative interior, 相对内部) 是指拓扑线性空间中的集合在相对意义下的内部