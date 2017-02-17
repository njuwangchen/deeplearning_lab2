import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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

    int act_hidden;
    int act_output;

    double learning_rate;
    double momentum;
    double weight_decay;

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

        this.act_hidden = ACT_HIDDEN;
        this.act_output = ACT_OUTPUT;

        this.learning_rate = learning_rate;
        this.momentum = momentum;
        this.weight_decay = weight_decay;
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

        String filename = String.format("python/data/"+
                this.hiddenLayers.get(0).output_size + "_" + "%.2f" + "_"+
                "%.2f"+"_"+"%.2f"+"_"+
                (this.act_hidden == Layer.ACT_SIGMOID ? "Sigmoid" : "RELU"),
                this.learning_rate, this.momentum, this.weight_decay);

        File output = new File(filename);

        FileWriter fileWriter = null;

        try {

            fileWriter = new FileWriter(output);

            double highestTuningAccuracy = 0.0;
            double testingAccuracy = 0.0;
            // List<Matrix> bestHiddenWeights = new ArrayList<Matrix>();
            // Matrix bestOutputWeights = null;

            int countOfStep = 0;

            while (true) {
                if (countOfStep > stepLimit) {
                    break;
                }
                String train_str = "Training accuracy is " + this.trainOneEpoch() + "\n";
                System.out.print(train_str);
                fileWriter.write(train_str);

                double thisTimeAccuracy = this.tuneAccuracy();
                String tune_str = "Tuning accuracy is " + thisTimeAccuracy + "\n";
                System.out.print(tune_str);
                fileWriter.write(tune_str);

                double thisTimeTestAccuracy = this.testAccuracy();
                String test_str = "Testing accuracy is " + thisTimeTestAccuracy + "\n";
                System.out.print(test_str);
                fileWriter.write(test_str);

                System.out.println();
                fileWriter.write("\n");

                fileWriter.flush();

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

            String high_str = "Highest tuning accuracy is " + highestTuningAccuracy + "\n";
            System.out.print(high_str);
            fileWriter.write(high_str);

            String final_str = "Final Testing accuracy is " + testingAccuracy + "\n";
            System.out.print(final_str);
            fileWriter.write(final_str);

            fileWriter.flush();
            fileWriter.close();

        }catch (IOException ex) {
            ex.printStackTrace();
        }
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
