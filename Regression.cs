using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Math;

namespace BoxCox;

using Range = Microsoft.ML.Probabilistic.Models.Range;

public static class Regression
{
    public static void Main()
    {
        //
        // Example data for regression:
        //
        // y_i is positive response
        // x_i is vector: intercept + predictor
        //

        double[] yData = { 1.2, 3.5, 2.7, 4.1, 2.0 };       // response
        double[][] xData = {                               // predictors
            new double[]{1.0, 0.0}, // intercept + x
            new double[]{1.0, 1.0},
            new double[]{1.0, 2.0},
            new double[]{1.0, 3.0},
            new double[]{1.0, 4.0},
        };

        int nObs = yData.Length;
        int nFeatures = xData[0].Length;

        //
        // Geometric-mean standardization for scale invariance
        //

        double sumLog = yData.Sum(v => Math.Log(v));
        double geoMean = Math.Exp(sumLog / nObs);
        double[] uData = yData.Select(v => v / geoMean).ToArray();

        //
        // Build model
        //

        var engine = new InferenceEngine();

        // Priors
        Variable<double> lambda =
            Variable.GaussianFromMeanAndVariance(0.0, 1.5 * 1.5).Named("lambda");

        Range i = new Range(nObs).Named("i");

        // Predictor matrix observed
        Vector[] xVectors = xData
            .Select(row => Vector.FromArray(row))
            .ToArray();
        VariableArray<Vector> X =
            Variable.Observed(xVectors, i);

        // Response (standardized)
        VariableArray<double> y =
            Variable.Observed(uData, i).Named("uData");

        // Regression coefficients
        double betaVariance = 2.5 * 2.5;
        double betaPrecision = 1.0 / betaVariance;
        var betaPrior = PositiveDefiniteMatrix.IdentityScaledBy(nFeatures, betaPrecision);
        Variable<Vector> beta =
            Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(nFeatures), betaPrior)
                .Named("beta");

        // Residual precision
        Variable<double> precision =
            Variable.GammaFromShapeAndScale(2.0, 1.0).Named("precision");

        // Latent z-values
        VariableArray<double> z =
            Variable.Array<double>(i).Named("z");

        using (Variable.ForEach(i))
        {
            Variable<double> mean = Variable.InnerProduct(beta, X[i]);
            z[i] = Variable.GaussianFromMeanAndPrecision(mean, precision);

            // Box–Cox transform: z[i] == BoxCox(u_i, lambda)
            Variable<double> transformed =
                Variable<double>.Factor(BoxCoxModel.BoxCoxTransform, y[i], lambda);

            Variable.ConstrainEqual(z[i], transformed);
        }


        //
        // Posterior inference
        //
        var betaPost = engine.Infer<VectorGaussian>(beta);
        var lambdaPost = engine.Infer<Gaussian>(lambda);
        var precisionPost = engine.Infer<Gamma>(precision);

        Console.WriteLine("Posterior over lambda:");
        Console.WriteLine(lambdaPost);
        Console.WriteLine();

        Console.WriteLine("Posterior over beta coefficients:");
        Vector betaMean = betaPost.GetMean();
        PositiveDefiniteMatrix betaCov = betaPost.GetVariance();
        for (int k = 0; k < nFeatures; k++)
        {
            double meanCoeff = betaMean[k];
            double sdCoeff = Math.Sqrt(betaCov[k, k]);
            Console.WriteLine($"  beta[{k}] ≈ {meanCoeff:F4} ± {1.96 * sdCoeff:F4}");
        }
        Console.WriteLine();

        Console.WriteLine("Posterior over precision:");
        Console.WriteLine(precisionPost);

        //
        // Example: print transformed data using posterior mean of lambda
        //
        double lamMean = lambdaPost.GetMean();
        Console.WriteLine("\nTransformed data using E[lambda]:");
        for (int k = 0; k < nObs; k++)
        {
            double t = BoxCoxModel.BoxCoxTransform(uData[k], lamMean);
            Console.WriteLine($"  z[{k}] = {t:F4}");
        }
    }
}
