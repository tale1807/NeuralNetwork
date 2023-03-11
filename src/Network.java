import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

public class Network<T> {

    private List<Layer> layers = new ArrayList<>();

    public Network(int[] layerStructure, double learningRate, DoubleUnaryOperator activationFunction,
            DoubleUnaryOperator derivativeActivationFunction) {
        if (layerStructure.length < 3) {
            throw new IllegalArgumentException("Should be at least 3 layers(1 input 1 hidden 1 output)");
        }

        Layer inputLayer = new Layer(Optional.empty(), layerStructure[0], learningRate, activationFunction,
                derivativeActivationFunction);
        layers.add(inputLayer);

        for (int i = 1; i < layerStructure.length; i++) {
            Layer nextLayer = new Layer(Optional.of(layers.get(i - 1)), layerStructure[i], learningRate,
                    activationFunction, derivativeActivationFunction);
            layers.add(nextLayer);
        }
    }

    private double[] outputs(double[] input) {
        double[] result = input;
        for (Layer layer : layers) {
            result = layer.outputs(result);
        }
        return result;
    }

    private void backpropagate(double[] expected) {
        int lastLayer = layers.size() - 1;
        layers.get(lastLayer).calculateDeltasForOutputLayer(expected);
        for (int i = lastLayer - 1; i >= 0; i--) {
            layers.get(i).calculateDeltasForHiddenLayer(layers.get(i + 1));
        }
    }

    private void updateWeights() {
        for (Layer layer : layers.subList(1, layers.size())) {
            for (Neuron neuron : layer.neurons) {
                for (int w = 0; w < neuron.weights.length; w++) {
                    neuron.weights[w] = neuron.weights[w]
                            + (neuron.learningRate * layer.previousLayer.get().outputCache[w] * neuron.delta);
                }
            }
        }
    }

    public void train(List<double[]> inputs, List<double[]> expecteds) {
        for (int i = 0; i < inputs.size(); i++) {
            double[] xs = inputs.get(i);
            double[] ys = expecteds.get(i);
            outputs(xs);
            backpropagate(ys);
            updateWeights();
        }
    }

    public class Results {

        public final int correct;
        public final int trials;
        public final double percentage;

        public Results(int correct, int trials, double percentage) {
            this.correct = correct;
            this.trials = trials;
            this.percentage = percentage;
        }
    }

    public Results validate(List<double[]> inputs, List<T> expecteds, Function<double[], T> interpret) {
        int correct = 0;
        for (int i = 0; i < inputs.size(); i++) {
            double[] input = inputs.get(i);
            T expected = expecteds.get(i);
            T result = interpret.apply(outputs(input));
            if (result.equals(expected)) {
                correct++;
            }
        }
        double percentage = (double) correct / (double) inputs.size();
        return new Results(correct, inputs.size(), percentage);
    }
}
