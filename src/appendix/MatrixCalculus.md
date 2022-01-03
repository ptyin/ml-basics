# Matrix Calculus

## Gradient

> Def. **Directional Derivative**
>
> The directional derivative of function $f:\mathbb{R}^n\rightarrow \mathbb{R}$ in the direction $u$ is
> $$
> \nabla_uf(x)=\mathop{\mathrm{lim}}\limits_{h\rightarrow 0}\frac{f(x+hu)-f(x)}{h}
> $$

- When $u$ is the $i$-th standard unit vector $e_i$, then $\nabla_uf(x)=f_i'(x)=\frac{\partial f(x)}{\partial x_i}$.
- For any $n$-dimensional vector $u$, the directional derivative of $f$ in the direction of $u$ can be represented as $\nabla_uf(x)=\sum_{i=1}^nf_i'(x)\cdot u_i$.
    - Proof.$\Rightarrow \begin{array}{l} \text{ let } g(h)=f(x+hu) \\ \nabla_uf(x)=g'(0)=\mathop{\mathrm{lim}}\limits_{h\rightarrow 0}\frac{f(x+hu)-g(0)}{h}  \\ \because g'(h)=\sum_{i=1}^n f_i'(x)\frac{d}{dh}(x_i+hu_i)=\sum_{i=1}^nf_i'(x)u_i  \\ \text{let }h=0\ \therefore \nabla_uf(x)=\sum_{i=1}^nf_i'(x)u_i \end{array}$

>  Def. **Gradient**
>
>  The gradient of $f$ is a vector function $\nabla f:\mathbb{R}^n\rightarrow \mathbb{R}^n$ defined by
>  $$
>  \begin{aligned}
>  & \nabla f(x)=\sum_{i=1}^n\frac{\partial f}{\partial x_i}e_i \\
>  \Rightarrow & \nabla f(x)=\left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \cdots, \frac{\partial f}{\partial x_n}\right]
>  \end{aligned}
>  $$

- $\nabla_uf(x)=\nabla f(x)\cdot u=\vert\vert\nabla f(x)\vert\vert \mathrm{cos}\ a$ Where $u$ is a unit vector.
- When $u=\nabla f(x)$ such that $a=0$, we have the maximum directional derivative of $f$.

## Matrix Derivatives

- The derivative of $f: \mathbb{R}^{m\times n}\rightarrow\mathbb{R}$ with respect to $A$ is defined as:

$$
\nabla f(A)=
\left[\begin{array}{ccc}
\frac{\partial f}{\partial A_{11}}& \cdots & \frac{\partial f}{\partial A_{n}}\\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial A_{m1}}& \cdots & \frac{\partial f}{\partial A_{mn}}\\
\end{array}\right]
$$

### trace

> Def. $trA=\sum_{i=1}^nA_ii$

- $trABCD=trDABC=trCDAB=trBCDA$
- $trA=trA^T,tr(A+B)=trA+trB,tr(aA)=a\cdot trA$
- $\nabla_AtrAB=B^T,\nabla_{A^T}f(A)=(\nabla_Af(A))^T$
- $\nabla_AtrABA^TC=CAB+C^TAB^T,\nabla_A\vert A\vert=\vert A\vert(A^{-1})^T$
- Funky trace derivative $\nabla_{A^T}trABA^TC=B^TA^TC^T+BA^TC$

### Jacobian Matrix

$$
{\displaystyle \mathbf {J} ={\begin{bmatrix}{\dfrac {\partial \mathbf {f} }{\partial x_{1}}}&\cdots &{\dfrac {\partial \mathbf {f} }{\partial x_{n}}}\end{bmatrix}}={\begin{bmatrix}{\dfrac {\partial f_{1}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{1}}{\partial x_{n}}}\\\vdots &\ddots &\vdots \\{\dfrac {\partial f_{m}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{m}}{\partial x_{n}}}\end{bmatrix}}}
$$



### Hesse Matrix

$$
{\displaystyle G(x_{0})={\begin{bmatrix}{\frac {\partial ^{2}f}{\partial x_{1}^{2}}}&{\frac {\partial ^{2}f}{\partial x_{1}\,\partial x_{2}}}&\cdots &{\frac {\partial ^{2}f}{\partial x_{1}\,\partial x_{n}}}\\\\{\frac {\partial ^{2}f}{\partial x_{2}\,\partial x_{1}}}&{\frac {\partial ^{2}f}{\partial x_{2}^{2}}}&\cdots &{\frac {\partial ^{2}f}{\partial x_{2}\,\partial x_{n}}}\\\\\vdots &\vdots &\ddots &\vdots \\\\{\frac {\partial ^{2}f}{\partial x_{n}\,\partial x_{1}}}&{\frac {\partial ^{2}f}{\partial x_{n}\,\partial x_{2}}}&\cdots &{\frac {\partial ^{2}f}{\partial x_{n}^{2}}}\end{bmatrix}}_{x_{0}}\,}
$$

- $H(f)=J(\nabla f)$