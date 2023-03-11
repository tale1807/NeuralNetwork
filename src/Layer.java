import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

public class Layer {

    public Optional<Layer> previousLayer;
    public List<Neuron> neurons = new ArrayList<>();
    public double[] outputCache;

    public Layer(Optional<Layer> previousLayer, int numNeurons, double learningRate,
            DoubleUnaryOperator activationFunction, DoubleUnaryOperator derivativeActivationFunction) {
        this.previousLayer = previousLayer;
        Random random = new Random();
        for (int i = 0; i < numNeurons; i++) {
            double[] randomWeights = null;
            if (previousLayer.isPresent()) {
                randomWeights = random.doubles(previousLayer.get().neurons.size()).toArray();
            }

            Neuron neuron = new Neuron(randomWeights, learningRate, activationFunction, derivativeActivationFunction);
            neurons.add(neuron);
        }
        outputCache = new double[numNeurons];
    }

    public double[] outputs(double[] inputs) {
        if (previousLayer.isPresent()) {
            outputCache = neurons.stream().mapToDouble(n -> n.output(inputs)).toArray();
        } else {
            outputCache = inputs;
        }

        return outputCache;
    }

    public void calculateDeltasForOutputLayer(double[] expected) {
        for (int n = 0; n < neurons.size(); n++) {
            neurons.get(n).delta = neurons.get(n).derivativeActivationFunction.applyAsDouble(neurons.get(n).outputCache)
                    * (expected[n] - outputCache[n]);
        }

    }

    public void calculateDeltasForHiddenLayer(Layer nextLayer) {
        for (int i = 0; i < neurons.size(); i++) {
            int index = i;
            double[] nextWeights = nextLayer.neurons.stream().mapToDouble(n -> n.weights[index]).toArray();
            double[] nextDeltas = nextLayer.neurons.stream().mapToDouble(n -> n.delta).toArray();
            double sumWeightsAndDeltas = Tools.dotProduct(nextWeights, nextDeltas);
            neurons.get(i).delta = neurons.get(i).derivativeActivationFunction.applyAsDouble(neurons.get(i).outputCache)
                    * sumWeightsAndDeltas;
        }
    }

}
