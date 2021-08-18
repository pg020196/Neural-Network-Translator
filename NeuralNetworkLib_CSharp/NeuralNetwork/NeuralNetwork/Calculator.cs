/*
 * A Workaround for missing numeric type constraints in C#
 * Compare https://stackoverflow.com/a/34186 
 * Code based on https://gist.github.com/klmr/314d05b66c72d62bd8a184514568e22f
 */

using System;
using System.Collections.Generic;

namespace Tensor
{
    public interface ICalculator { }

    public interface ICalculator<T> : ICalculator
    {
        T getZero();
        T getOne();
        T getValue(double value);
        T getPosInfinity();
        T getNegInfinity();
        T getRandUniform(double min, double max);
        T getRandNormal(double mean, double std);
        T add(T a, T b);
        T divide(T a, T b);
        T divide(T a, int b);
        T multiply(T a, T b);
        T subtract(T a, T b);
        T exp(T a);
        T negate(T a);
        T invert(T a);
        T abs(T a);
        bool isGreater(T a, T b);
        bool isLess(T a, T b);
        T floor(T a);
        int floorInt(T a);
        T round(T a);
        int roundInt(T a);
        T ceil(T a);
        int ceilInt(T a);
        T modulo(T a, T b);
    }

    static class Calculators
    {
        public static readonly Dictionary<Type, ICalculator> calculators = new Dictionary<Type, ICalculator>() {
            { typeof(float), new FloatCalculator() },
            { typeof(double), new DoubleCalculator() }
        };

        public static ICalculator<T> GetInstance<T>()
        {
            return (ICalculator<T>)calculators[typeof(T)];
        }
    }

    class FloatCalculator : ICalculator<float>
    {
        Random rand = new Random();
        public float getZero() { return 0f; }
        public float getOne() { return 1f; }
        public float getValue(double value) { return (float)value; } 
        public float getPosInfinity() { return float.PositiveInfinity; }
        public float getNegInfinity() { return float.NegativeInfinity; }
        public float getRandUniform(double min, double max) { return (float) (min + rand.NextDouble() * (max - min)); }
        public float getRandNormal(double mean, double std) 
        {
            // use Box Muller Transform to convert uniform to normal distribution
            // code based on https://stackoverflow.com/a/218600

            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random floats
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + std * randStdNormal; //random normal(mean,std)
            return (float)randNormal;
        }
        public float add(float a, float b) { return a + b; }
        public float divide(float a, float b) { return a / b; }
        public float divide(float a, int b) { return a / ((float)b); }
        public float multiply(float a, float b) { return a * b; }
        public float subtract(float a, float b) { return a - b; }
        public float exp(float a) { return (float)Math.Exp(a); }
        public float negate(float a) { return -a; }
        public float invert(float a) { return (float)(1d / a); }
        public float abs(float a) { return Math.Abs(a); }
        public bool isGreater(float a, float b) { return a > b; }
        public bool isLess(float a, float b) { return a < b; }
        public float floor(float a) { return (float)Math.Floor(a); }
        public int floorInt(float a) { return (int)Math.Floor(a); }
        public float round(float a) { return (float)Math.Round(a); }
        public int roundInt(float a) { return (int)Math.Round(a); }
        public float ceil(float a) { return (float)Math.Ceiling(a); }
        public int ceilInt(float a) { return (int)Math.Ceiling(a); }
        public float modulo(float a, float b) { return a % b; }
    }

    class DoubleCalculator : ICalculator<double>
    {
        Random rand = new Random();
        public double getZero() { return 0d; }
        public double getOne() { return 1d; }
        public double getValue(double value) { return value; }
        public double getPosInfinity() { return double.PositiveInfinity; }
        public double getNegInfinity() { return double.NegativeInfinity; }
        public double getRandUniform(double min, double max) { return min + rand.NextDouble() * (max - min); }
        public double getRandNormal(double mean, double std) 
        {
            // use Box Muller Transform to convert uniform to normal distribution
            // code based on https://stackoverflow.com/a/218600

            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + std * randStdNormal; //random normal(mean,std)
            return randNormal;
        }
        public double add(double a, double b) { return a + b; }
        public double divide(double a, double b) { return a / b; }
        public double divide(double a, int b) { return a / ((double)b); }
        public double multiply(double a, double b) { return a * b; }
        public double subtract(double a, double b) { return a - b; }
        public double exp(double a) { return Math.Exp(a); }
        public double negate(double a) { return -a; }
        public double invert(double a) { return 1d / a; }
        public double abs(double a) { return Math.Abs(a); }
        public bool isGreater(double a, double b) { return a > b; }
        public bool isLess(double a, double b) { return a < b; }
        public double floor(double a) { return Math.Floor(a); }
        public int floorInt(double a) { return (int)Math.Floor(a); }
        public double round(double a) { return Math.Round(a); }
        public int roundInt(double a) { return (int)Math.Round(a); }
        public double ceil(double a) { return Math.Ceiling(a); }
        public int ceilInt(double a) { return (int)Math.Ceiling(a); }
        public double modulo(double a, double b) { return a % b; }
    }
}