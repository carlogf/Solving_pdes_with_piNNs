# Solving_pdes_with_pinns
 
We use Physics Informed Neural Networks (PINNs) to solve some basic example PDEs such as the logistic equation and the heat equation. We use [sciann](https://www.sciann.com/) wichs is based on tensorflow and keras to build and train our models.

In myFirstEquation we solve the logistic equation for a fixed initial value
$$
\left\lbrace
 \begin{array} 
\frac{\mathrm{d}}{\mathrm{d}x} u(x) = u(x)(1-u(x)) \\
u(0) = 2
\end{array} \right. 
$$

