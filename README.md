# Box–Cox Transform with Bayesian Inference

## Overview

The **Box–Cox transform** is a family of power transforms used to stabilise variance and improve approximate normality in strictly positive data. For an observation $y_i > 0$ and transformation parameter $\lambda$:

$$
z_i(\lambda)=
\begin{cases}
\dfrac{y_i^\lambda - 1}{\lambda}, & \lambda \neq 0,\\[6pt]
\log y_i, & \lambda = 0.
\end{cases}
$$

This repository implements a **fully Bayesian** Box–Cox model using **Infer.NET**, inferring posterior distributions over:
- the transformation parameter $\lambda$,
- the mean $\mu$,
- the precision $\tau = 1/\sigma^2$.

The classical Box–Cox procedure finds a point estimate for $\lambda$; here we infer a full posterior distribution.

## Current Bayesian Model

To ensure the model is **scale-invariant**, the observed data are first standardised by their geometric mean:

$$
u_i = \frac{y_i}{\dot{y}}, \qquad
\dot{y} = \left(\prod_{i=1}^{n} y_i\right)^{1/n}.
$$

Inference is then performed on the standardised data $(u_i)$, and no Jacobian terms or correction factors are needed.

The Bayesian model implemented in `Program.cs` is:

$$
\begin{aligned}
u_i &= \frac{y_i}{\dot{y}},\\
z_i(\lambda) &= \mathrm{Box\text{-}Cox}(u_i,\lambda),\\
z_i \mid \mu,\tau &\sim \mathcal{N}(\mu,\tau^{-1}),\\
\mu &\sim \mathcal{N}(0,\,2.5^2),\\
\tau &\sim \mathrm{Gamma}(2,1),\\
\lambda &\sim \mathcal{N}(0,\,1.5^2).
\end{aligned}
$$

### Key features of this formulation
- **Scale invariance:** Standardising by the geometric mean ensures that inference on $\lambda$ is unaffected by rescaling of the data (e.g., metres vs. centimetres).
- **Weakly informative priors:**  
  - $\mu$ centred at 0 matches standardised-data behaviour.  
  - $\mathrm{Gamma}(2,1)$ implies a sensible distribution over $\sigma$ for standardised data.  
  - $\lambda$ prior centred at 0 (log-transform) but allows wide variability.
- **No custom Jacobian operators needed** (unlike the earlier version of the repo).

## Folder Structure

- `Program.cs` – main entry point; performs standardisation, constructs the Bayesian model, runs inference, and prints posterior summaries.
- `BoxCoxTransformOp.cs` – custom Infer.NET operator needed for the Box–Cox factor.
- `README.md` – explanation of the model (this file).

## Running the Example

```
dotnet build
dotnet run --project BoxCox
```

Sample output:

```
mu posterior       : Gaussian(Mean=..., Variance=...)
precision posterior: Gamma(Shape=..., Rate=...)
lambda posterior   : Gaussian(Mean=..., Variance=...)

Transformed data z using E[lambda]=...:
  z[0] = ...
  z[1] = ...
```

The transformed values are computed using the posterior mean of $\lambda$, though full Bayesian workflows should sample $\lambda$ from its posterior.

## Using Your Own Data

1. Ensure all data are strictly positive.
2. Edit the data section in `Program.cs`:
   ```csharp
   double[] yData = { /* your data */ };
   ```
3. Optionally adjust priors if you have specific prior knowledge:
   ```csharp
   mu ~ Normal(μ0, σμ²)
   tau ~ Gamma(shape, scale)
   lambda ~ Normal(λ0, σλ²)
   ```
4. Run the program as usual.

## Interpreting Results

- **λ posterior:**  
  Describes plausible power transformations.  
  - λ ≈ 0 → log transform  
  - λ ≈ 1 → no transform  
  - λ < 0 or > 1 → nonlinear stretch/compression  

- **μ, τ posteriors:**  
  Describe the Gaussian fit in the transformed space.

- **Transformed data:**  
  Useful for diagnosing normality or feeding into downstream models.

## Benefits of the Bayesian Approach

- Provides full uncertainty quantification for the transformation parameter.
- Avoids instability issues of classical MLE, especially with small samples.
- Easy to embed in hierarchical or regression models.
- Priors provide regularization and increased robustness.

## Limitations

- Variational/EP inference is approximate.
- Extreme values of λ may require careful numerical settings.
- The model assumes data become approximately Gaussian after transformation.
