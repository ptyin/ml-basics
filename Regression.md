## Basics

- Linear hypothesis: $h(x)=\theta_1x+\theta_0,\ \theta_i(i=1,2\text{ for 2D cases})$.
- cost function: 

$$
\begin{array}{cc}
J(\theta)=\frac12\sum_{i=1}^m{(h_\theta(x^{(i)})-y^{(i)})^2}, & 
h_\theta(x)=\sum_{i=0}^n\theta_i x_i=\theta^Tx
\end{array}
$$

- best choice for $\theta=\mathop{\mathrm{argmin}}\limits_\theta\ J(\theta)$

## Gradient

> Def. **Directional Derivative**
>
> The directional derivative of function $f:\mathbb{R}^n\rightarrow \mathbb{R}$ in the direction $u$ is
> $$
> \nabla_uf(x)=\mathop{\mathrm{lim}}\limits_{h\rightarrow 0}\frac{f(x+hu)-f(x)}{h}
> $$

- When $u$ is the $i$-th standard unit vector $e_i$, then $\nabla_uf(x)=f_i'(x)=\frac{\part f(x)}{\part x_i}$.

- For any $n$-dimensional vector $u$, the directional derivative of $f$ in the direction of $u$ can be represented as $\nabla_uf(x)=\sum_{i=1}^nf_i'(x)\cdot u_i$.

    - > Prof.
        > $$
        > \begin{array}{l}
        > \text{let } g(h)=f(x+hu) \\
        > 
        > \nabla_uf(x)=g'(0)=\mathop{\mathrm{lim}}\limits_{h\rightarrow 0}\frac{f(x+hu)-g(0)}{h}  \\
        > 
        > \because g'(h)=\sum_{i=1}^n f_i'(x)\frac{d}{dh}(x_i+hu_i)=\sum_{i=1}^nf_i'(x)u_i  \\
        > 
        > \text{let }h=0\ \therefore \nabla_uf(x)=\sum_{i=1}^nf_i'(x)u_i
        > 
        > \end{array}
        > $$

>  Def. **Gradient**
>
> The gradient of $f$ is a vector function $\nabla f:\mathbb{R}^n\rightarrow \mathbb{R}^n$ defined by
> $$
> \begin{align}
> & \nabla f(x)=\sum_{i=1}^n\frac{\part f}{\part x_i}e_i \\
> \Rightarrow & \nabla f(x)=\left[\frac{\part f}{\part x_1}, \frac{\part f}{\part x_2}, \cdots, \frac{\part f}{\part x_n}\right]
> \end{align}
> $$

- $\nabla_uf(x)=\nabla f(x)\cdot u=\vert\vert\nabla f(x)\vert\vert \mathrm{cos}\ a$ Where $u$ is a unit vector.
- When $u=\nabla f(x)$ such that $a=0$, we have the maximum directional derivative of $f$.

## Gradient Descent (GD) Algorithm

> Algorithm.
>
> ```pseudocode
> Given a starting point \theta in dom J
> while converence criterion is satisfied
> 	Calculate gradient \nabla J(\theta)
> 	Update \theta \leftarrow \theta - \alpha\nabla J(\theta)
> ```
>
> $\theta$ Is usually initialized randomly, and $\alpha$ is so-called learning rate.

- For linear regression,

$$
\begin{array}{l}
\theta_j\leftarrow\theta_j-\alpha\frac{\part J(\theta)}{\part\theta_j},\ \forall j=0,1,\cdots,n,\ x_0^{(i)}=1 \\
\begin{align}
\frac{\part J(\theta)}{\part \theta_j} &= \frac{\part}{\part \theta_j}\frac12\sum_{i=1}^m(\theta^Tx^{(i)}-y^{(i)})^2 \\
& =\frac{\part}{\part \theta_j}\frac12\sum_{i=1}^m(\sum_{j=0}^n\theta_j x_j^{(i)}-y^{(i)})^2 \\
& =\sum_{i=1}^m{(\theta^Tx^{(i)}-y^{(i)})x_j^{(i)}}
\end{align}
\end{array}
$$

- Another commonly used form $J(\theta)=\frac1{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$.
- $m$ is introduced to scale the objective function to deal with differently sized training set.

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

### Jacobian Matrix (雅可比矩阵)

$$
{\displaystyle \mathbf {J} ={\begin{bmatrix}{\dfrac {\partial \mathbf {f} }{\partial x_{1}}}&\cdots &{\dfrac {\partial \mathbf {f} }{\partial x_{n}}}\end{bmatrix}}={\begin{bmatrix}{\dfrac {\partial f_{1}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{1}}{\partial x_{n}}}\\\vdots &\ddots &\vdots \\{\dfrac {\partial f_{m}}{\partial x_{1}}}&\cdots &{\dfrac {\partial f_{m}}{\partial x_{n}}}\end{bmatrix}}}
$$



### Hesse Matrix (黑塞矩阵)

$$
{\displaystyle G(x_{0})={\begin{bmatrix}{\frac {\partial ^{2}f}{\partial x_{1}^{2}}}&{\frac {\partial ^{2}f}{\partial x_{1}\,\partial x_{2}}}&\cdots &{\frac {\partial ^{2}f}{\partial x_{1}\,\partial x_{n}}}\\\\{\frac {\partial ^{2}f}{\partial x_{2}\,\partial x_{1}}}&{\frac {\partial ^{2}f}{\partial x_{2}^{2}}}&\cdots &{\frac {\partial ^{2}f}{\partial x_{2}\,\partial x_{n}}}\\\\\vdots &\vdots &\ddots &\vdots \\\\{\frac {\partial ^{2}f}{\partial x_{n}\,\partial x_{1}}}&{\frac {\partial ^{2}f}{\partial x_{n}\,\partial x_{2}}}&\cdots &{\frac {\partial ^{2}f}{\partial x_{n}^{2}}}\end{bmatrix}}_{x_{0}}\,}
$$

- $H(f)=J(\nabla f)$

## Revisiting Least Square with Matrix Form

$$
X=\left[\begin{array}{c}
(x^{(1)})^T \\
\vdots \\
(x^{(m)})^T
\end{array}\right], Y=\left[\begin{array}{c}
y^{(1)} \\
\vdots \\
y^{(m)}
\end{array}\right] \\
J(\theta)=\frac12\sum_{i=1}^m{(\theta^Tx^{(i)}-y^{(i)})^2}=\frac12(X\theta-Y)^T(X\theta-Y)
$$

- Minimize $J(\theta)=\frac12(Y-X\theta)^T(Y-X\theta)$

$$
\begin{aligned}
\nabla_\theta J(\theta) &=\nabla_\theta \frac12(Y-X\theta)^T(Y-X\theta) \\
&= \frac12\nabla_\theta tr(Y^TY-Y^TX\theta-\theta^TX^TY+\theta^TX^TX\theta)\\
&= \frac12 \nabla_\theta tr(\theta^TX^TX\theta)-X^TY \\
&= \frac12 (X^TX\theta+X^TX\theta)-X^TY \\
&=X^TX\theta-X^TY
\end{aligned}
$$

- > <u>***Theorem***</u>.
    >
    > The matrix $A^TA$ is invertible if and only if the columns of $A$ are linearly independent. In this case, there exists only one least-squares solution.
    > $$
    > \theta=(X^TX)^{-1}X^TY
    > $$

- 以上为least-squares解析解 (Normal Equation)

