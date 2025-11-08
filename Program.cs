using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace BoxCox;

using Range = Microsoft.ML.Probabilistic.Models.Range;

public static class Program
{
    public static void Main()
    {
        double[] yData = { 1.2, 3.5, 2.7, 4.1, 2.0 };
        int length = yData.Length;

        var engine = new InferenceEngine();

        Variable<double> mu = Variable.GaussianFromMeanAndVariance(0, 100).Named("mu");
        Variable<double> precision = Variable.GammaFromShapeAndScale(2, 1).Named("precision");
        Variable<double> lambda = Variable.GaussianFromMeanAndVariance(0, 4).Named("lambda");

        Range n = new Range(length).Named("n");
        VariableArray<double> y = Variable.Observed(yData, n).Named("y");
        double sumLogY = BoxCoxModel.SumLog(yData);

        using (Variable.ForEach(n))
        {
            Variable<double> transformed = Variable<double>.Factor(BoxCoxModel.BoxCoxTransform, y[n], lambda).Named("z_point");
            Variable<double> zPrior = Variable.GaussianFromMeanAndPrecision(mu, precision).Named("z_prior");
            Variable.ConstrainEqual(transformed, zPrior);
        }

        Variable<double> jacobianWeight = Variable<double>.Factor(BoxCoxModel.BoxCoxJacobianFactor, lambda, sumLogY).Named("jacobian_weight");
        Variable.ConstrainEqual(jacobianWeight, 1.0);

        Gaussian muPosterior = engine.Infer<Gaussian>(mu);
        Gamma precisionPosterior = engine.Infer<Gamma>(precision);
        Gaussian lambdaPosterior = engine.Infer<Gaussian>(lambda);

        Console.WriteLine($"mu posterior      : {muPosterior}");
        Console.WriteLine($"precision posterior: {precisionPosterior}");
        Console.WriteLine($"lambda posterior   : {lambdaPosterior}");

        double lambdaMean = lambdaPosterior.GetMean();
        double[] transformedData = yData.Select(value => BoxCoxModel.BoxCoxTransform(value, lambdaMean)).ToArray();

        Console.WriteLine($"Transformed data z using E[lambda]={lambdaMean:F4}:");
        for (int i = 0; i < transformedData.Length; i++)
        {
            Console.WriteLine($"  z[{i}] = {transformedData[i]:F4}");
        }
    }
}

public static class BoxCoxModel
{
    public static double BoxCoxTransform(double y, double lambda)
    {
        if (Math.Abs(lambda) < 1e-8)
            return Math.Log(y);
        return (Math.Pow(y, lambda) - 1.0) / lambda;
    }

    public static double BoxCoxJacobianFactor(double lambda, double sumLogY)
    {
        double logW = (lambda - 1.0) * sumLogY;
        return Math.Exp(logW);
    }

    public static double SumLog(double[] y)
    {
        double s = 0.0;
        foreach (double value in y)
        {
            s += Math.Log(value);
        }
        return s;
    }
}
