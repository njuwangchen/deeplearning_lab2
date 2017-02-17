import java.util.*;

/**
 * Created by chenwang on 2/10/17.
 */
public class NeuralNet {

    private final static boolean debug = false;
    private final static boolean output_label = false;

    InputLayer inputLayer;
    List<HiddenLayer> hiddenLayers;
    OutputLayer outputLayer;

    List<Instance> trainingSet;
    List<Instance> tuningSet;
    List<Instance> testingSet;

    Map<Integer, String> reverseLabelMap;

    public NeuralNet(int inputLayerSize, int outputLayerSize, int hiddenLayersNum, int[] hiddenLayersSizes,
                     int ACT_HIDDEN, int ACT_OUTPUT, double learning_rate, double momentum, double weight_decay,
                     List<Instance> trainingSet, List<Instance> tuningSet, List<Instance> testingSet,
                     Map<Integer, String> reverseLabelMap) {
        if (hiddenLayersSizes.length != hiddenLayersNum) {
            System.err.println("Hidden Layers Num doesn't match with the number of sizes");
            System.exit(1);
        }

        this.inputLayer = new InputLayer(inputLayerSize);
        Layer prevLayer = this.inputLayer;

        this.hiddenLayers = new ArrayList<HiddenLayer>();
        for (int i=0; i<hiddenLayersNum; ++i) {
            HiddenLayer hiddenLayer = new HiddenLayer(prevLayer, hiddenLayersSizes[i],
                    ACT_HIDDEN, learning_rate, momentum, weight_decay);
            prevLayer = hiddenLayer;
            this.hiddenLayers.add(hiddenLayer);
        }

        this.outputLayer = new OutputLayer(prevLayer, outputLayerSize, ACT_OUTPUT,
                learning_rate, momentum, weight_decay);

        this.trainingSet = trainingSet;
        this.tuningSet = tuningSet;
        this.testingSet = testingSet;

        this.reverseLabelMap = reverseLabelMap;
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

        Map<Integer, String> reverse = new HashMap<Integer, String>();
        reverse.put(0, "a");
        reverse.put(1, "b");

        Instance instance = new Instance(features, 3, label, labelMap);
        List<Instance> trainingList = new ArrayList<Instance>();
        trainingList.add(instance);

        NeuralNet nn = new NeuralNet(3, 2, 1, new int[]{3},
                Layer.ACT_RELU, Layer.ACT_SIGMOID,
                0.01, 0.9, 0.0, trainingList, null, null, reverse);

        for (int i=0; i<50; ++i)
            nn.trainOneInstance(instance);
    }

    public double trainOneEpoch() {
        int count = 0;
        Collections.shuffle(this.trainingSet);
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

            if (output_label) {
                int outInt = 0;
                for (int i = 0; i < outputLayer.finalOutput.dimension; ++i) {
                    if (outputLayer.finalOutput.getElementAt(i) == 1) {
                        outInt = i;
                        break;
                    }
                }
                String outLabel = reverseLabelMap.get(outInt);
                System.out.println(outLabel);
            }

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

    public void startTrainingWithEarlyStopping(int stepLimit) {
        double highestTuningAccuracy = 0.0;
        double testingAccuracy = 0.0;
        // List<Matrix> bestHiddenWeights = new ArrayList<Matrix>();
        // Matrix bestOutputWeights = null;

        int countOfStep = 0;

        while (true) {
            if (countOfStep > stepLimit) {
                break;
            }
            System.out.println("Training accuracy is " + this.trainOneEpoch());

            double thisTimeAccuracy = this.tuneAccuracy();
            System.out.println("Tuning accuracy is " + thisTimeAccuracy);
            double thisTimeTestAccuracy = this.testAccuracy();
            System.out.println("Testing accuracy is " + thisTimeTestAccuracy);
            System.out.println();

            if (thisTimeAccuracy > highestTuningAccuracy) {
//                for (HiddenLayer hiddenLayer: this.hiddenLayers) {
//                    Matrix weights = new Matrix(hiddenLayer.weightMat);
//                    bestHiddenWeights.add(weights);
//                }

//                bestOutputWeights = new Matrix(this.outputLayer.weightMat);
                highestTuningAccuracy = thisTimeAccuracy;
                testingAccuracy = thisTimeTestAccuracy;

                countOfStep = 0;
            } else {
                ++countOfStep;
            }
        }

        System.out.println("Highest tuning accuracy is " + highestTuningAccuracy);
        System.out.println("Final Testing accuracy is " + testingAccuracy);
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
