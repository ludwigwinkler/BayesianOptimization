<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  CommonHTML: {
    scale: 130
  }
});
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
<script type="text/javascript" async
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


## Introduction

Many problems in science and engineering can be formulated as a mathematical optimization problem in which an optimal solution is sought, either locally or globally.
The field of global optimization is the application of applied mathematics and numerical analysis towards finding the overall optimal solution in a set of candidate solutions.
Local optimization is considered an easier problem, in which it suffices to find an optimum which is optimal with respect to its immediate vicinity.
Such a local optimum is obviously a suboptimal solution and, while harder to find, global optima are more preferred.

Generally, optimization problems are formulated as finding the optimal solution which minimizes, respectively maximizes, a criterion, which is commonly referred to as the objective function.
Further constraints on the the set of solutions can be formulated, such that only a subset of solutions are permissible as candidates for the optimum.

Optimization is commonly done in an iterative manner where the objective function is evaluated for multiple candidate solutions.
Due to the iterative nature, it becomes desirable to evaluate this function as few times as possible over the course of the entire optimization, which becomes even more crucial when the evaluation of the objective function itself is costly.
Therefore, it would be advantageous to infer information about the objective function beyond the evaluations themselves, which only provide punctual information.

Bayesian inference models provide such advantages since they compute predictive distributions instead of punctual evaluations.
One class of Bayesian inference models are Gaussian processes (GP), which can be applied to model previous evaluations of the objective function as a multi-variate Gaussian distribution.
Given such a Gaussian distribution over the previous evaluations, information can be inferred over all candidate solutions in the feasible set at once.

## Gaussian Processes

In most situations where observations have many small independent components, their distribution tends towards the Gaussian distribution.
Compared to other probability distributions, the Gaussian distribution is tractable and it's parameters have intuitive meaning.
The theory of the central limit theorem (CLT) makes the Gaussian distribution a versatile distribution which is used in numerous situations in science and engineering.

A convenient property of the Gaussian distribution for a random variable $X$ is its complete characterization by its mean $$\mu$$ and variance $\Sigma$:

$$
\begin{align}
     \mu &= \mathbb{E}[X] \\
     \Sigma &= \mathbb{E}[(X-\mu)^T(X-\mu)]
\end{align}
$$

Mathematically, a multivariate Gaussian for a vector $x \in \mathbb{R}^d$ is defined by its mean $\mu \in \mathbb{R}^d$ and covariance function $\Sigma \in \mathbb{R}^{d \times d}$:

$$
\begin{align}
          \mathcal{N}(x \ | \ \mu, \Sigma}) &=
               \frac{1}{\sqrt{(2 \pi)^d |\Sigma}|^2}}
               \exp \left[
               -\frac{1}{2}
               (\x-\mu)^T \Sigma^{-1}(\x-\mu)
               \right] \\
               &\propto
               \exp \left[
               -\frac{1}{2}
               (\x-\mu)^T \Sigma^{-1}(\x-\mu)
               \right]
\end{align}
$$

A useful property of the Gaussian distribution is that its shape is determined by its mean and covariance in the exponential term.
This allows us to omitt the normalization constant and determine the relevant mean and covariance terms from the exponential term, as seen in \eqref{eq:gausspropto}.


### Markdown5

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

$$
a^2 + b^2 = c^2
$$

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3



- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ludwigwinkler/BayesianOptimization/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
