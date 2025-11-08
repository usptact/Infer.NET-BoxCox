using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Utilities;

namespace BoxCox;

[FactorMethod(typeof(BoxCoxModel), nameof(BoxCoxModel.BoxCoxJacobianFactor))]
[Quality(QualityBand.Preview)]
public static class BoxCoxJacobianOp
{
    private const double MinVariance = 1e-12;

    public static Gaussian lambdaAverageConditional(Gaussian boxCoxJacobianFactor, double sumLogY, Gaussian lambda)
    {
        // Factor contributes exp((lambda - 1) * sumLogY), which shifts the natural parameter
        // MeanTimesPrecision by sumLogY while leaving precision unchanged.
        return Gaussian.FromNatural(sumLogY, 0.0);
    }

    public static Gaussian lambdaAverageConditional(Gaussian boxCoxJacobianFactor, Gaussian sumLogY, Gaussian lambda)
    {
        double summedLog = ExtractScalar(sumLogY);
        return lambdaAverageConditional(boxCoxJacobianFactor, summedLog, lambda);
    }

    public static Gaussian boxCoxJacobianFactorAverageConditional(Gaussian lambda, double sumLogY)
    {
        if (lambda.IsUniform())
        {
            return Gaussian.Uniform();
        }

        if (lambda.IsPointMass)
        {
            double value = BoxCoxModel.BoxCoxJacobianFactor(lambda.Point, sumLogY);
            return Gaussian.PointMass(value);
        }

        double mean = lambda.GetMean();
        double variance = lambda.GetVariance();

        double logMean = (mean - 1.0) * sumLogY + 0.5 * variance * sumLogY * sumLogY;
        double meanWeight = Math.Exp(logMean);

        double logVariance = variance * sumLogY * sumLogY;
        double varianceWeight = meanWeight * meanWeight * (Math.Exp(logVariance) - 1.0);
        varianceWeight = Math.Max(varianceWeight, MinVariance);

        return Gaussian.FromMeanAndVariance(meanWeight, varianceWeight);
    }

    public static Gaussian boxCoxJacobianFactorAverageConditional(Gaussian lambda, Gaussian sumLogY)
    {
        double summedLog = ExtractScalar(sumLogY);
        return boxCoxJacobianFactorAverageConditional(lambda, summedLog);
    }

    public static double LogAverageFactor(Gaussian boxCoxJacobianFactor, double sumLogY, Gaussian lambda)
    {
        if (lambda.IsUniform())
        {
            return 0.0;
        }

        double mean = lambda.IsPointMass ? lambda.Point : lambda.GetMean();
        double variance = lambda.IsPointMass ? 0.0 : lambda.GetVariance();

        return (mean - 1.0) * sumLogY + 0.5 * variance * sumLogY * sumLogY;
    }

    public static double LogAverageFactor(Gaussian boxCoxJacobianFactor, Gaussian sumLogY, Gaussian lambda)
    {
        double summedLog = ExtractScalar(sumLogY);
        return LogAverageFactor(boxCoxJacobianFactor, summedLog, lambda);
    }

    public static double LogEvidenceRatio(Gaussian boxCoxJacobianFactor, double sumLogY, Gaussian lambda, Gaussian toLambda)
    {
        return LogAverageFactor(boxCoxJacobianFactor, sumLogY, lambda);
    }

    public static double LogEvidenceRatio(Gaussian boxCoxJacobianFactor, Gaussian sumLogY, Gaussian lambda, Gaussian toLambda)
    {
        double summedLog = ExtractScalar(sumLogY);
        return LogAverageFactor(boxCoxJacobianFactor, summedLog, lambda);
    }

    private static double ExtractScalar(Gaussian dist)
    {
        if (dist.IsPointMass)
        {
            return dist.Point;
        }

        double mean;
        double variance;
        dist.GetMeanAndVariance(out mean, out variance);
        return mean;
    }
}

