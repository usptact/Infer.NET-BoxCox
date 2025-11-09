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
        double sumLogY = BoxCoxModel.SumLog(yData);
        double geoMean = Math.Exp(sumLogY / length);
        double[] uData = yData.Select(v => v / geoMean).ToArray();  // geometric mean normalized data

        var engine = new InferenceEngine();

        // std deviation 1–2.5 centers the location plausibly around 0 without being wildly vague
        Variable<double> mu = Variable.GaussianFromMeanAndVariance(0, 2.5*2.5).Named("mu");
        
        /* gives an implied σ distribution with median ~0.77 and 95% upper tail ~2.0,
           which is reasonable for standardized data; also avoids extremely heavy tails that Gamma(1,1) yields. */
        Variable<double> precision = Variable.GammaFromShapeAndScale(2, 1).Named("precision");

        // std deviation 1–1.5 keeps the log-transform (0) as the prior center while allowing plausible λ in roughly ±2 range.
        Variable<double> lambda = Variable.GaussianFromMeanAndVariance(0, 1.5*1.5).Named("lambda");

        Range n = new Range(length).Named("n");
        VariableArray<double> y = Variable.Observed(uData, n).Named("y");

        var z = Variable.Array<double>(n).Named("z");

        using (Variable.ForEach(n))
        {
            z[n] = Variable.GaussianFromMeanAndPrecision(mu, precision).Named("z_prior");
            Variable<double> transformed = Variable<double>.Factor(BoxCoxModel.BoxCoxTransform, y[n], lambda).Named("z_point");
            Variable.ConstrainEqual(z[n], transformed);
        }

        Gaussian muPosterior = engine.Infer<Gaussian>(mu);
        Gamma precisionPosterior = engine.Infer<Gamma>(precision);
        Gaussian lambdaPosterior = engine.Infer<Gaussian>(lambda);

        Console.WriteLine($"mu posterior      : {muPosterior}");
        Console.WriteLine($"precision posterior: {precisionPosterior}");
        Console.WriteLine($"lambda posterior   : {lambdaPosterior}");

        double lambdaMean = lambdaPosterior.GetMean();
        double[] transformedData = uData.Select(value => BoxCoxModel.BoxCoxTransform(value, lambdaMean)).ToArray();

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
