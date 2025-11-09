using System;

namespace BoxCox;

public static class BoxCoxModel
{
    public static double BoxCoxTransform(double y, double lambda)
    {
        if (Math.Abs(lambda) < 1e-8)
            return Math.Log(y);

        return (Math.Pow(y, lambda) - 1.0) / lambda;
    }

    public static double SumLog(double[] values)
    {
        double sum = 0.0;
        foreach (double value in values)
        {
            sum += Math.Log(value);
        }

        return sum;
    }
}


