# Bayesian Box–Cox Regression with Infer.NET

## Overview

The **Box–Cox transform** lifts positive responses into (approximately) Gaussian space by learning a power parameter $\lambda$:

$$
z_i(\lambda)=
\begin{cases}
\dfrac{y_i^\lambda - 1}{\lambda}, & \lambda \neq 0,\\
\log y_i, & \lambda = 0.
\end{cases}
$$

`Regression.cs` shows how to use this transform inside a linear regression model, inferring $\lambda$ jointly with the regression coefficients and residual precision.  
It is implemented with **Infer.NET** and expectation propagation (EP).

## Current Bayesian Model

Each observed response $y_i > 0$ is scaled by the geometric mean to remove sensitivity to units:

$$
u_i = \frac{y_i}{\dot{y}}, \qquad
\dot{y} = \left(\prod_{i=1}^{n} y_i\right)^{1/n}.
$$

Let $\mathbf{x}_i \in \mathbb{R}^p$ denote the predictor vector (including an intercept) and $\boldsymbol{\beta}$ the regression coefficients.  
The model fitted in `Regression.cs` is:

$$
\begin{aligned}
u_i &= \frac{y_i}{\dot{y}},\\
z_i(\lambda) &= \text{Box–Cox}(u_i,\lambda),\\
z_i \mid \boldsymbol{\beta},\tau &\sim \mathcal{N}(\mathbf{x}_i^\top \boldsymbol{\beta},\,\tau^{-1}),\\
\boldsymbol{\beta} &\sim \mathcal{N}_p\!\left(\mathbf{0},\, (2.5^2)\,\mathbf{I}\right),\\
\tau &\sim \mathrm{Gamma}(2,1),\\
\lambda &\sim \mathcal{N}(0,\,1.5^2).
\end{aligned}
$$

### Key features

- **Scale invariance:** the geometric-mean normalisation ensures $\lambda$ depends only on ratios among observations.
- **Joint inference:** $\lambda$, $\boldsymbol{\beta}$ and $\tau$ are inferred together; uncertainty propagates automatically.
- **Weakly informative priors:** broad Gaussian prior on coefficients, sensible Gamma prior on $\sigma$, and a centred prior on $\lambda$.

## Files of interest

- `Regression.cs` – executable that builds the Bayesian regression model and prints posterior summaries.
- `BoxCoxTransformOp.cs` – custom Infer.NET operator supplying messages for the Box–Cox factor.
- `BoxCoxModel.cs` – shared helpers (transform implementation, $\log$-sum utility).
- `BoxCoxRegression.csproj` – project file used to build/run the regression example.

## How to Run

```
dotnet build
dotnet run --project BoxCox/BoxCoxRegression.csproj
```

Typical output includes the posterior over $\lambda$, regression coefficients (mean ± 95 % interval), residual precision, and transformed data evaluated at $\mathbb{E}[\lambda]$.

## Using Your Own Data

1. Ensure responses are strictly positive.
2. Update the data section near the top of `Regression.cs`:
   ```csharp
   double[] yData = { /* positive responses */ };
   double[][] xData = {
       new double[]{1.0, x1_0, x2_0, ...}, // intercept + predictors
       ...
   };
   ```
3. Optionally customise priors if you have domain knowledge (e.g. shrinkage on coefficients, tighter prior on $\lambda$).
4. Rebuild and run from the repository root:
   ```
   dotnet run --project BoxCox/BoxCoxRegression.csproj
   ```

## Interpreting Results

- **$\lambda$ posterior:** reveals which power transform best Gaussianises the residuals.
  - $\lambda \approx 0$ corresponds to a log transform.
  - $\lambda \approx 1$ implies no transform.
- **$\boldsymbol{\beta}$ posterior:** inspect means and intervals to understand predictor effects in transformed space.
- **Precision posterior ($\tau$):** describes residual variability after transformation and regression.

## Benefits of the Bayesian Approach

- Full uncertainty quantification for both the transformation parameter and regression parameters.
- Robust to small samples compared with classical profile-likelihood approaches.
- Naturally extendable to hierarchical or multilevel models.

## Limitations

- EP provides an approximation; convergence can be sensitive to extreme $\lambda$ values.
- The model assumes the transformed responses are adequately modelled by a Gaussian linear regression.
- Ensure predictors and responses are prepared so the geometric mean is well defined (no zeros or negatives).

