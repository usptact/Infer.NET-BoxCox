# Box–Cox Transform with Bayesian Inference

## Overview

The **Box–Cox transform** is a family of power transforms that stabilise variance and improve approximate normality of positive data.  For an observation $y_i > 0$ and transformation parameter $\lambda$, the Box–Cox transform is

$$
z_i(\lambda) =
\begin{cases}
\dfrac{y_i^\lambda - 1}{\lambda}, & \lambda \neq 0,\\
\log y_i, & \lambda = 0.
\end{cases}
$$

Classically, $\lambda$ is chosen by maximising the (profile) log-likelihood of a Gaussian model applied to the transformed data.  Once $\lambda$ is fixed, the data are transformed once and ordinary Gaussian modelling continues.

This repository implements the same model **probabilistically** in Infer.NET.  Instead of a point estimate, we treat $\lambda$ (as well as the mean $\mu$ and precision $\tau$) as random variables and infer their posterior distributions.  The Jacobian term

$$
w(\lambda) = \exp\Bigl((\lambda - 1)\sum_{i=1}^N \log y_i\Bigr)
$$

is represented explicitly to ensure a correct change-of-variables density.

## Bayesian Model Used

Given strictly positive data $\mathbf{y} = (y_1,\dots,y_N)$, we assume

$$
\begin{aligned}
\lambda &\sim \mathcal{N}(0, 2^2),\\
\mu &\sim \mathcal{N}(0, 100),\\
\tau &\sim \mathrm{Gamma}(2, 1),\\
z_i \mid y_i,\lambda &= z_i(\lambda) \quad\text{(Box–Cox transform)},\\
z_i \mid \mu,\tau &\sim \mathcal{N}(\mu, \tau^{-1}) \quad\text{independently},\\
\log w &= (\lambda - 1)\sum_{i=1}^N \log y_i \quad\text{(Jacobian)}.
\end{aligned}
$$

Infer.NET uses expectation propagation (EP) with custom message operators (`BoxCoxTransformOp`, `BoxCoxJacobianOp`) to approximate the joint posterior
$p(\lambda, \mu, \tau \mid \mathbf{y})$.  Posterior summaries (means, variances) are readily available, and transformed data can be computed post hoc via $z_i(\hat{\lambda})$ where $\hat{\lambda} = \mathbb{E}[\lambda \mid \mathbf{y}]$.

### Why Bayesian?

Compared with the classical MLE pipeline, this approach offers:

- **Uncertainty quantification**: a posterior over $\lambda$ instead of a single estimate.
- **Regularisation**: priors discourage extreme values when data are scarce.
- **Extensibility**: the model can be embedded in richer hierarchical settings without re-deriving optimisation routines.

## Folder Structure

- `Program.cs` – main entry point; runs Bayesian Box–Cox inference and prints posterior summaries.
- `BoxCoxTransformOp.cs`, `BoxCoxJacobianOp.cs` – custom Infer.NET operator implementations for EP message passing.
- `AssemblyAttributes.cs` – registers message operators with Infer.NET.

## Running the Example

```bash
dotnet build
dotnet run --project BoxCox
```

Sample output:

```
mu posterior      : Gaussian(Mean=2.53, Variance=0.48)
precision posterior: Gamma(Shape=4.12, Rate=2.03)
lambda posterior   : Gaussian(Mean=0.18, Variance=0.05)
Transformed data z using E[lambda]=0.18:
  z[0] = ...
```

## Using Your Own Data

1. **Ensure positivity**  
   Box–Cox requires strictly positive observations.  Apply a constant shift if your data contain zeros or negatives.

2. **Replace `yData` in `Program.cs`**  
   ```csharp
   double[] yData = { /* your positive observations */ };
   ```

3. **Optional: adjust priors**  
   The priors on $\mu$, $\tau$, and $\lambda$ are intentionally weak.  Modify
   ```csharp
   Variable<double> mu =
       Variable.GaussianFromMeanAndVariance(mu0, muVariance).Named("mu");
   Variable<double> precision =
       Variable.GammaFromShapeAndScale(shape, scale).Named("precision");
   Variable<double> lambda =
       Variable.GaussianFromMeanAndVariance(lambda0, lambdaVariance)
       .Named("lambda");
   ```
   to encode domain knowledge or stronger regularisation if desired.

4. **Build and run**  
```bash
dotnet run --project BoxCox
```
   The program prints posterior means for $\mu$, $\tau$, and $\lambda$, followed by transformed data computed at $\mathbb{E}[\lambda \mid \mathbf{y}]$.

### Interpreting Results

- **$\lambda$ posterior** – mean and variance indicate the most plausible transform and its uncertainty.
- **$\mu,\tau$ posteriors** – describe the Gaussian fit in transformed space.
- **Transformed data** – useful for downstream analysis (e.g., regression, diagnostics).  For full Bayesian predictions, consider sampling $\lambda$ from its posterior and re-transforming rather than using the single plug-in mean.

## Concepts Referenced

- **Expectation propagation (EP)**: an approximate inference algorithm used by Infer.NET to propagate beliefs through the factor graph.
- **Gamma distribution**: here parameterised by shape $\alpha$ and scale $\beta$; mean $=\alpha \beta$.
- **Change of variables / Jacobian**: ensures densities transform correctly when applying a deterministic transform such as Box–Cox.

## Limitations

- The current operators assume `y` is observed (no missing-data messages).
- Accuracy depends on the numerical quadrature within the custom operators.
- Posterior transformed data currently uses a plug-in $\hat{\lambda}$; draw samples if you need full uncertainty propagation.

Despite these caveats, the Bayesian approach provides a principled and extensible alternative to classical MLE-based Box–Cox fitting.

