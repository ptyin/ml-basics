# K Means

## Clustering

- Given: $N$ unlabeled examples $\{x_1,\cdots, x_N\}$ and number of desired partitions $K$.
- Goal: : Group the examples into $K$ “homogeneous” partitions.

> Def.
>
> Given a set of observations $X = \{x_1, x_2, · · · , x_N\} (x_i\in\mathbb{R}^D)$, partition the $N$ observations into $K$ sets ($K ≤ N$) $\{\mathcal{C}_k\}_{k=1,··· ,K}$ such that the sets minimize the within-cluster sum of squares:
> $$
> \{\mathcal{C}_k\}=\arg\min_{\{\mathcal{C}_k\}}\sum_{i=1}^K\sum_{x\in\mathcal{C}_i}\vert\vert x-\mu_i\vert\vert^2
> $$
> where $\mu$ is the mean of points in set $\mathcal{C_i}$.

## K Means Algorithm

> Algorithm
>
> - (Re)-Assign each example $x_{i}$ to its closest cluster center (based on the smallest Euclidean distance)
>
> $$
> \mathcal{C}_{k}=\left\{x_{i} \mid\left\|x_{i}-\mu_{k}\right\|^{2} \leq\left\|x_{i}-\mu_{k^{\prime}}\right\|^{2}, \text { for } \forall k^{\prime} \neq k\right\}
> $$
> - ( $\mathcal{C}_{k}$ is the set of examples assigned to cluster $k$ with center $\mu_{k}$ )
> - Update the cluster means
>
> $$
> \mu_{k}=\operatorname{mean}\left(\mathcal{C}_{k}\right)=\frac{1}{\left|\mathcal{C}_{k}\right|} \sum_{x \in \mathcal{C}_{k}} x
> $$

- Let $z_{i,k}$ be an indicator

$$
z_{i,k}=\left\{\begin{array}{ll}
1, &x_i\in\mathcal{C}_k\\
0, &otherwise
\end{array}\right.
$$

- and $z_i=[z_{i,1},\cdots,z_{i,k}]^T$ represents the one-hot encoding of $x_i$.
- The loss is $L(\mu, \mathbf{X}, \mathbf{Z})=\sum_{i=1}^{N} \sum_{k=1}^{K} z_{i, k}\left\|x_{i}-\mu_{k}\right\|^{2}=\|\mathbf{X}-\mathbf{Z} \mu\|^{2}$
- where $\mathbf{X}\in\mathbb{R}^{N\times D}$,  $\mathbf{Z}\in\mathbb{R}^{N\times K}$ , $\mu\in\mathbb R^{K\times D}$

## Limitations

- Makes **hard** assignments of points to clusters
- Works well only is the clusters are roughly of equal sizes
- K-means also works well only when the clusters are round-shaped and does badly if the clusters have non-convex shapes

## Kernel K Means

- Basic idea: Replace the Euclidean distance/similarity computations in K-means by the kernelized versions

$$
\begin{aligned}
d\left(x_{i}, \mu_{k}\right) &=\left\|\phi\left(x_{i}\right)-\phi\left(\mu_{k}\right)\right\| \\
\left\|\phi\left(x_{i}\right)-\phi\left(\mu_{k}\right)\right\|^{2} &=\left\|\phi\left(x_{i}\right)\right\|^{2}+\left\|\phi\left(\mu_{k}\right)\right\|^{2}-2 \phi\left(x_{i}\right)^{T} \phi\left(\mu_{k}\right) \\
&=k\left(x_{i}, x_{i}\right)+k\left(\mu_{k}, \mu_{k}\right)-2 k\left(x_{i}, \mu_{k}\right)
\end{aligned}
$$

