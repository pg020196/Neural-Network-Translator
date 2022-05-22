using Tensor;
using NUnit.Framework;

namespace NeuralNetworkTests
{
    [TestFixture]
    public class CalculatorTests
    {
        [Test]
        public void getZero_Float_ReturnsFloatZero()
        {
            var floatCalculator = new FloatCalculator();

            var zero = floatCalculator.getZero();

            Assert.That(zero, Is.InstanceOf<float>());
            Assert.That(zero, Is.EqualTo(0));
        }
    }
}
