# information_geometry
Python package that numerically calculates the structure of probability distributions on statistical manifolds.

See `statistical_manifolds.ipynb` for examples and `information_geometry.py` for the primary code.

![image](https://github.com/Noeloikeau/information_geometry/blob/main/example.png)

For external references from different backgrounds, see [statistical manifold learning](https://proceedings.mlr.press/v32/suna14.pdf) (for machine learning & neuroscience), [curved exponential families](https://bsi-ni.brain.riken.jp/database/file/85/076.pdf) (for pure math), [emergent general relativity](https://arxiv.org/pdf/1310.1831.pdf) (for theoretical physics), [sensor network localization](https://www.mdpi.com/1099-4300/19/7/308/htm) (for applied engineering). 

Here we study *information geometry*, a framework for quantifying the geometry of probability distributions. The objects of interest are *statistical manifolds*, which are spaces of probability distributions smoothly connected by changes in parameters. The distance between nearby probability distributions on the manifold is interpreted as their difference in information content with respect to a varied parameter. As a result, the shape ensuing from all possible parameter variations about a point exactly characterizes the "latent" information content of the distribution. This is analogous to paths taken by light being used to characterize a gravitational field -- except that the curvature of the space arises from a distribution of probability, not mass, and the paths are in parameter space, not spacetime. 

We consider normalized probability density functions $f(x,a):R^D\times R^M\rightarrow R$ such that $\int_{X} f(x,a) dx = 1$, where

> $x$ : D-dimensional random variable;

> $a$ : M-dimensional distribution parameter;

> $X=R^{D \times N}$ : sample domain of observations of $x$, taken as $N\rightarrow \infty$. 

The metric $g_{ij}(a)$ in this space is the Fisher information matrix. Its components are products of the partial derivatives of the probability distribution with respect to the parameter axes, evaluated at a particular point in parameter space. It is computed from the logarithm of the reciprocal probability (i.e. the information $I$): 

> $I(x,a) = - \log[f(x,a)]$

The Fisher information quantifies the difference between infinitesimally distinct probability distributions. Each component gives the distance (in bits base $e$) between nearby points on a statistical manifold. It can be interpreted as the change in KL-divergence between two probability distributions as a result of infinitesimally varying parameters $a_{i}$ and $a_{j}$ about point $a$ and "integrating out" the sample space $X$:

> $g_{ij}(a) =  \int_{X} [\partial_{a_{i}}I(x,a)][\partial_{a_{j}}I(x,a)] f(x,a) dx $
>            := $\langle I_{i}I_{j} \rangle$

where $\partial_{i}I:=I_{i}$ is an element of the Jacobian, and $\langle (.) \rangle = \int_{X}(.)p(x,a)dx$.

It can be showed that $\langle I_{i}I_{j} \rangle = \langle I_{ij} \rangle$, i.e. that the expectation values of the Hessian $\partial_{i}\partial_{j}I:=I_{ij}$ and products of the Jacobian are equal under this metric. Therefore $g_{ij}(a)$ can be computed from either. Additionally $\langle I_{i} \rangle = 0$, which can be obtained by differentiating the normalization condition (see https://arxiv.org/pdf/1310.1831.pdf). These constraints are useful for evaluating the accuracy of numerical approaches to computing the metric, as is done here.

The partial derivatives of the metric give the geodesic equations and define the curvature of the space. This curvature can be interpreted as the "latent" information stored in the derivatives of the distribution with respect to its parameter vectors. Geodesics minimize the distance between distributions and can be used to smoothly interpolate between points on a manifold. In other words, successive changes in a probability distribution along a geodesic are minimized.

Analytically it can be shown that for the Fisher information metric the metric derivatives are:
> $\partial_{k}g_{ij} = \langle -I_{i}I_{j}I_{k}+I_{i}I_{kj}+I_{j}I_{ki}\rangle$. 

The connection coefficients relate the metric derivatives to the geodesic acceleration by means of a linear combination of functions. These Christoffel symbols are defined by:

>  $\Gamma^{l}_{ij} = \frac{1}{2} g^{lk} \langle \partial_{i}g_{kj} + \partial_{j}g_{ki}-\partial_{k}g_{ij} \rangle = g^{lk} \langle I_{ij}I_{k}-\frac{1}{2}I_{i}I_{j}I_{k}\rangle $

where the second equality is the analytic solution obtained for the Fisher information metric specifically, and $g^{lk}$ is the inverse matrix of $g_{lk}$. Note the implicit summation over repeated indices.

The geodesic equations are finally:
>   $\ddot{a}^{l} = -\Gamma^{l}_{ij}\dot{a}^{i}\dot{a}^{j}$

where $\dot{a}^{l}=\frac{da^{l}}{dt}$ is the derivative of the $l$'th component of $a$ with respect to the affine parameter of the geodesic path $t$. These equations are an initial value problem specified by the initial point in parameter space $a_{0}$, the initial tangent vector $\dot{a}_{0}$, and the number of steps along the affine path of the geodesic $T$. Note the initial tangents are contravariant vectors that must be normalized by the metric at that point:

>   $|\dot{a}_{0}|^{2}=g_{ii}(a_{0})\dot{a}_{0}^{i}\dot{a}_{0}^{i}$. 

The solution is obtained numerically using a 1st-order finite-difference method (Euler). The geodesic path is parametrized in terms of an affine, unit interval of $T$ points $t\in[0,1]$ such that $dt=\frac{1}{T}$. The resulting algorithm is:

>   $a^{l}(t+1) = a^{l}(t) + dt \cdot \dot{a}^{l}(t)$

>   $\dot{a}^{l}(t+1) = \dot{a}^{l}(t) + dt \cdot F^{l}(a(t),\dot{a}(t))$

>   $F^{l}(a(t),\dot{a}(t))=  -\sum_{i,j}\Gamma^{l}_{ij}(a(t)) \cdot \dot{a}^{i}(t)\dot{a}^{j}(t)$.

All other derivatives are computed numerically using 2nd-order finite-differences (perturbations above and below the given vector). 
