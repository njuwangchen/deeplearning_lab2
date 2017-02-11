import java.util.*;

/**
 * Created by chenwang on 2/10/17.
 */
public class NeuralNet {

    InputLayer inputLayer;
    List<HiddenLayer> hiddenLayers;
    OutputLayer outputLayer;

    List<Instance> trainingSet;
    List<Instance> tuningSet;
    List<Instance> testingSet;

    public NeuralNet(int inputLayerSize, int outputLayerSize, int hiddenLayersNum, int[] hiddenLayersSizes,
                     int ACT_HIDDEN, int ACT_OUTPUT, double learning_rate,
                     List<Instance> trainingSet, List<Instance> tuningSet, List<Instance> testingSet) {
        if (hiddenLayersSizes.length != hiddenLayersNum) {
            System.err.println("Hidden Layers Num doesn't match with the number of sizes");
            System.exit(1);
        }

        this.inputLayer = new InputLayer(inputLayerSize);
        Layer prevLayer = this.inputLayer;

        this.hiddenLayers = new ArrayList<HiddenLayer>();
        for (int i=0; i<hiddenLayersNum; ++i) {
            HiddenLayer hiddenLayer = new HiddenLayer(prevLayer, hiddenLayersSizes[i], ACT_HIDDEN, learning_rate);
            prevLayer = hiddenLayer;
            this.hiddenLayers.add(hiddenLayer);
        }

        this.outputLayer = new OutputLayer(prevLayer, outputLayerSize, ACT_OUTPUT, learning_rate);

        this.trainingSet = trainingSet;
        this.tuningSet = tuningSet;
        this.testingSet = testingSet;
    }

    public static void main(String[] args) {
        // write test code here
        List<Double> features = new ArrayList<Double>();
        features.add(1.0);
        features.add(2.0);
        features.add(3.0);
        String label = "a";
        Map<String, Integer> labelMap = new HashMap<String, Integer>();
        labelMap.put("a", 0);
        labelMap.put("b", 1);

        Instance instance = new Instance(features, label, labelMap);
        List<Instance> trainingList = new ArrayList<Instance>();
        trainingList.add(instance);

        NeuralNet nn = new NeuralNet(3, 2, 1, new int[]{3}, Layer.ACT_RELU, Layer.ACT_SIGMOID,
                0.1, trainingList, null, null);

        for (int i=0; i<50; ++i)
            nn.trainOneInstance(instance);
    }

    public double trainOneEpoch() {
        int count = 0;
        for (Instance instance: this.trainingSet) {
            if (trainOneInstance(instance)) ++count;
        }
        return (double)count/this.trainingSet.size();
    }

    public double tuneAccuracy() {
        int count = 0;
        for (Instance instance: this.tuningSet) {
            if (testOneInstance(instance)) ++count;
        }
        return (double)count/this.tuningSet.size();
    }

    public double testAccuracy() {
        int count = 0;
        for (Instance instance: this.testingSet) {
            if (testOneInstance(instance)) ++count;
        }
        return (double)count/this.testingSet.size();
    }

    public boolean trainOneInstance(Instance instance) {
        forwardOneInstance(instance);
        boolean ret = this.outputLayer.isCorrect();
        backOneInstance(instance);
        return ret;
    }

    public boolean testOneInstance(Instance instance) {
        forwardOneInstance(instance);
        boolean ret = this.outputLayer.isCorrect();
        return ret;
    }

    private void forwardOneInstance(Instance instance) {

        // forward propagation
        this.inputLayer.feedInput(instance.feature);
        this.inputLayer.forward();
        this.inputLayer.activate();

        Iterator<HiddenLayer> iterator = this.hiddenLayers.iterator();
        while (iterator.hasNext()) {
            HiddenLayer hiddenLayer = iterator.next();
            hiddenLayer.forward();
            hiddenLayer.activate();
        }

        this.outputLayer.feedLabel(instance.label);
        this.outputLayer.forward();
        this.outputLayer.activate();
        this.outputLayer.calOutput();
    }

    private void backOneInstance(Instance instance) {

        //back propagation
        this.outputLayer.feedLabel(instance.label);
        this.outputLayer.back();

        for (int i=hiddenLayers.size()-1; i>=0; --i) {
            HiddenLayer hiddenLayer = hiddenLayers.get(i);
            hiddenLayer.back();
        }

        //adjust weights
        this.outputLayer.adjustWeights();
        for (int i=hiddenLayers.size()-1; i>=0; --i) {
            HiddenLayer hiddenLayer = hiddenLayers.get(i);
            hiddenLayer.adjustWeights();
        }
    }
}
