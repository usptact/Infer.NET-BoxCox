using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors.Attributes;
using Microsoft.ML.Probabilistic.Utilities;

namespace BoxCox;

[FactorMethod(typeof(BoxCoxModel), nameof(BoxCoxModel.BoxCoxTransform))]
[Quality(QualityBand.Preview)]
public static class BoxCoxTransformOp
{
    private const double TruncationStdDevs = 6.0;
    private const int IntegrationSteps = 240; // must be even for Simpson integration
    private const double MinVariance = 1e-8;
    private const double PointMassVariance = 1e-6;

    public static Gaussian boxCoxTransformAverageConditional(
        Gaussian boxCoxTransform,
        Gaussian to_boxCoxTransform,
        double y,
        Gaussian lambda,
        Gaussian to_lambda,
        Gaussian result)
    {
        Gaussian message = ComputeResultMessage(y, lambda);
        result.SetTo(message);
        return result;
    }

    public static Gaussian boxCoxTransformAverageConditional(
        Gaussian boxCoxTransform,
        Gaussian to_boxCoxTransform,
        Gaussian y,
        Gaussian lambda,
        Gaussian to_lambda,
        Gaussian result)
    {
        double yValue = ExtractObservedValue(y);
        return boxCoxTransformAverageConditional(boxCoxTransform, to_boxCoxTransform, yValue, lambda, to_lambda, result);
    }

    public static Gaussian lambdaAverageConditional(
        Gaussian boxCoxTransform,
        Gaussian to_boxCoxTransform,
        double y,
        Gaussian lambda,
        Gaussian to_lambda,
        Gaussian result)
    {
        Gaussian message = ComputeLambdaMessage(boxCoxTransform, y, lambda);
        result.SetTo(message);
        return result;
    }

    public static Gaussian lambdaAverageConditional(
        Gaussian boxCoxTransform,
        Gaussian to_boxCoxTransform,
        Gaussian y,
        Gaussian lambda,
        Gaussian to_lambda,
        Gaussian result)
    {
        double yValue = ExtractObservedValue(y);
        return lambdaAverageConditional(boxCoxTransform, to_boxCoxTransform, yValue, lambda, to_lambda, result);
    }

    public static double LogAverageFactor(Gaussian boxCoxTransform, double y, Gaussian lambda)
    {
        return ComputeLogAverageFactor(boxCoxTransform, y, lambda);
    }

    public static double LogAverageFactor(Gaussian boxCoxTransform, Gaussian y, Gaussian lambda)
    {
        double yValue = ExtractObservedValue(y);
        return ComputeLogAverageFactor(boxCoxTransform, yValue, lambda);
    }

    public static double LogEvidenceRatio(Gaussian boxCoxTransform, double y, Gaussian lambda, Gaussian to_lambda)
    {
        return ComputeLogAverageFactor(boxCoxTransform, y, lambda);
    }

    public static double LogEvidenceRatio(Gaussian boxCoxTransform, Gaussian y, Gaussian lambda, Gaussian to_lambda)
    {
        double yValue = ExtractObservedValue(y);
        return ComputeLogAverageFactor(boxCoxTransform, yValue, lambda);
    }

    private static Gaussian ComputeResultMessage(double y, Gaussian lambda)
    {
        if (lambda.IsPointMass)
        {
            double transformed = BoxCoxModel.BoxCoxTransform(y, lambda.Point);
            return Gaussian.PointMass(transformed);
        }

        if (lambda.IsUniform())
        {
            return Gaussian.Uniform();
        }

        var stats = ComputeIntegralStats(lambda, Gaussian.Uniform(), y);
        double mean = stats.ZMean;
        double variance = Math.Max(stats.ZSecondMoment - mean * mean, MinVariance);
        return Gaussian.FromMeanAndVariance(mean, variance);
    }

    private static Gaussian ComputeLambdaMessage(Gaussian boxCoxTransform, double y, Gaussian lambda)
    {
        if (lambda.IsUniform() || boxCoxTransform.IsUniform())
        {
            return Gaussian.Uniform();
        }

        if (lambda.IsPointMass)
        {
            return Gaussian.PointMass(lambda.Point);
        }

        var stats = ComputeIntegralStats(lambda, boxCoxTransform, y);
        double mean = stats.LambdaMean;
        double variance = Math.Max(stats.LambdaSecondMoment - mean * mean, MinVariance);

        var posterior = Gaussian.FromMeanAndVariance(mean, variance);
        var message = new Gaussian();
        message.SetToRatio(posterior, lambda, forceProper: true);
        return message;
    }

    private static double ComputeLogAverageFactor(Gaussian boxCoxTransform, double y, Gaussian lambda)
    {
        if (lambda.IsUniform())
        {
            return 0.0;
        }

        if (lambda.IsPointMass)
        {
            double transformed = BoxCoxModel.BoxCoxTransform(y, lambda.Point);
            double logLike = EvaluateLikelihoodLog(boxCoxTransform, transformed);
            return logLike;
        }

        var stats = ComputeIntegralStats(lambda, boxCoxTransform, y);
        return stats.LogNormalizer;
    }

    private static IntegralStats ComputeIntegralStats(Gaussian lambda, Gaussian z, double y)
    {
        if (lambda.IsPointMass)
        {
            double lambdaVal = lambda.Point;
            double zVal = BoxCoxModel.BoxCoxTransform(y, lambdaVal);
            double logLike = EvaluateLikelihoodLog(z, zVal);
            double logNormalizerPoint = logLike;
            return new IntegralStats(
                norm: 1.0,
                lambdaMean: lambdaVal,
                lambdaSecondMoment: lambdaVal * lambdaVal,
                zMean: zVal,
                zSecondMoment: zVal * zVal,
                logNormalizer: logNormalizerPoint);
        }

        double meanLambda;
        double varianceLambda;
        try
        {
            meanLambda = lambda.GetMean();
            varianceLambda = lambda.GetVariance();
        }
        catch
        {
            meanLambda = 0.0;
            varianceLambda = 1e2;
        }

        varianceLambda = Math.Max(varianceLambda, MinVariance);
        double sigmaLambda = Math.Sqrt(varianceLambda);

        double lower = meanLambda - TruncationStdDevs * sigmaLambda;
        double upper = meanLambda + TruncationStdDevs * sigmaLambda;

        int steps = IntegrationSteps;
        if (steps % 2 == 1)
        {
            steps += 1;
        }

        double h = (upper - lower) / steps;
        double[] lambdaVals = new double[steps + 1];
        double[] transformVals = new double[steps + 1];
        double[] logWeights = new double[steps + 1];

        double maxLogWeight = double.NegativeInfinity;

        for (int i = 0; i <= steps; i++)
        {
            double lambdaVal = lower + i * h;
            lambdaVals[i] = lambdaVal;
            double transformVal = BoxCoxModel.BoxCoxTransform(y, lambdaVal);
            transformVals[i] = transformVal;

            double logWeight = lambda.GetLogProb(lambdaVal) + EvaluateLikelihoodLog(z, transformVal);
            logWeights[i] = logWeight;
            if (logWeight > maxLogWeight)
            {
                maxLogWeight = logWeight;
            }
        }

        double sumW = 0.0;
        double sumLambda = 0.0;
        double sumLambda2 = 0.0;
        double sumZ = 0.0;
        double sumZ2 = 0.0;

        for (int i = 0; i <= steps; i++)
        {
            double coeff = (i == 0 || i == steps) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
            double weight = Math.Exp(logWeights[i] - maxLogWeight);
            double scaledWeight = coeff * weight;

            sumW += scaledWeight;
            double lambdaVal = lambdaVals[i];
            double zVal = transformVals[i];

            sumLambda += scaledWeight * lambdaVal;
            sumLambda2 += scaledWeight * lambdaVal * lambdaVal;
            sumZ += scaledWeight * zVal;
            sumZ2 += scaledWeight * zVal * zVal;
        }

        double integral = sumW * h / 3.0;
        if (integral <= 0.0 || double.IsNaN(integral) || double.IsInfinity(integral))
        {
            double fallbackZ = BoxCoxModel.BoxCoxTransform(y, meanLambda);
            return new IntegralStats(
                norm: double.Epsilon,
                lambdaMean: meanLambda,
                lambdaSecondMoment: meanLambda * meanLambda + varianceLambda,
                zMean: fallbackZ,
                zSecondMoment: fallbackZ * fallbackZ + MinVariance,
                logNormalizer: maxLogWeight + Math.Log(Math.Max(double.Epsilon, integral)));
        }

        double lambdaMean = (sumLambda * h / 3.0) / integral;
        double lambdaSecondMoment = (sumLambda2 * h / 3.0) / integral;
        double zMean = (sumZ * h / 3.0) / integral;
        double zSecondMoment = (sumZ2 * h / 3.0) / integral;

        double logNormalizer = maxLogWeight + Math.Log(integral);

        return new IntegralStats(
            norm: integral,
            lambdaMean: lambdaMean,
            lambdaSecondMoment: lambdaSecondMoment,
            zMean: zMean,
            zSecondMoment: zSecondMoment,
            logNormalizer: logNormalizer);
    }

    private static double ExtractObservedValue(Gaussian dist)
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

    private static double EvaluateLikelihoodLog(Gaussian z, double transformed)
    {
        if (z.IsUniform())
        {
            return 0.0;
        }

        if (z.IsPointMass)
        {
            double diff = transformed - z.Point;
            double variance = PointMassVariance;
            return -0.5 * diff * diff / variance - 0.5 * Math.Log(2.0 * Math.PI * variance);
        }

        return z.GetLogProb(transformed);
    }

    public static double BoxCoxDerivative(double y, double lambda)
    {
        double logY = Math.Log(y);
        if (Math.Abs(lambda) < 1e-6)
        {
            double logY2 = logY * logY;
            return 0.5 * logY2;
        }

        double yPow = Math.Exp(lambda * logY);
        double numerator = lambda * yPow * logY - (yPow - 1.0);
        double denominator = lambda * lambda;
        return numerator / denominator;
    }

    public static double InvertBoxCox(double y, double target, double initialGuess)
    {
        double lambda = initialGuess;
        for (int iter = 0; iter < 50; iter++)
        {
            double value = BoxCoxModel.BoxCoxTransform(y, lambda) - target;
            if (Math.Abs(value) < 1e-8)
            {
                break;
            }

            double derivative = BoxCoxDerivative(y, lambda);
            if (Math.Abs(derivative) < 1e-10)
            {
                break;
            }

            lambda -= value / derivative;
            lambda = Math.Clamp(lambda, -20.0, 20.0);
        }

        return lambda;
    }

    private readonly struct IntegralStats
    {
        public IntegralStats(double norm, double lambdaMean, double lambdaSecondMoment, double zMean, double zSecondMoment, double logNormalizer)
        {
            Norm = norm;
            LambdaMean = lambdaMean;
            LambdaSecondMoment = lambdaSecondMoment;
            ZMean = zMean;
            ZSecondMoment = zSecondMoment;
            LogNormalizer = logNormalizer;
        }

        public double Norm { get; }
        public double LambdaMean { get; }
        public double LambdaSecondMoment { get; }
        public double ZMean { get; }
        public double ZSecondMoment { get; }
        public double LogNormalizer { get; }
    }
}

