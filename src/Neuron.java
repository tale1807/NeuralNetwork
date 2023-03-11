import java.util.function.DoubleUnaryOperator;

public class Neuron {

    public double[] weights;
    public final double learningRate;
    public double outputCache;
    public double delta;
    public final DoubleUnaryOperator activationFunction;
    public final DoubleUnaryOperator derivativeActivationFunction;

    public Neuron(double[] weights, double learningRate, DoubleUnaryOperator activationFunction,
            DoubleUnaryOperator derivativeActivationFunction) {
        this.weights = weights;
        this.learningRate = learningRate;
        outputCache = 0.0;
        delta = 0.0;
        this.activationFunction = activationFunction;
        this.derivativeActivationFunction = derivativeActivationFunction;

    }

    public double output(double[] inputs) {
        outputCache = Tools.dotProduct(inputs, weights);
        return activationFunction.applyAsDouble(outputCache);
    }
}
