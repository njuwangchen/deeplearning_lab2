package submit;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by chenwang on 2/17/17.
 */
public class Lab2 {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Usage: Lab2 Filename");
            System.exit(1);
        }

        String fileName = args[0];
        int hu = 10;
        double learning_rate = 0.01;
        double momentum = 0.9;
        double weight_decay = 0.0;

        String hidden_act_str = "Sigmoid";
        int hidden_act = -1;
        if (hidden_act_str.equals("Sigmoid")) {
            hidden_act = Layer.ACT_SIGMOID;
        } else if (hidden_act_str.equals("RELU")) {
            hidden_act = Layer.ACT_RELU;
        }

        DataParser dp = new DataParser();
        dp.parseFile(fileName);

        NeuralNet neuralNet = new NeuralNet(dp.data_feature_size * dp.feature_value_num,
                dp.data_label_size, 1, new int[]{hu},
                hidden_act, Layer.ACT_SIGMOID,
                learning_rate, momentum, weight_decay,
                dp.trainingSet, dp.tuningSet, dp.testingSet, dp.reverseLabelMap);

        neuralNet.startTrainingWithEarlyStopping(10);
    }
}

class DataParser {

    List<Instance> instanceList;

    List<Instance> trainingSet;
    List<Instance> tuningSet;
    List<Instance> testingSet;

    Map<Integer, String> reverseLabelMap;

    int data_feature_size;
    int data_label_size;
    int feature_value_num;

    public DataParser() {
        this.instanceList = new ArrayList<Instance>();

        this.trainingSet = new ArrayList<Instance>();
        this.tuningSet = new ArrayList<Instance>();
        this.testingSet = new ArrayList<Instance>();

        this.reverseLabelMap = new HashMap<Integer, String>();

        this.data_feature_size = 17;
        this.data_label_size = 3;
        this.feature_value_num = 21;
    }

    public static void main(String[] args) {
        DataParser dp = new DataParser();
        dp.parseFile("data/protein-secondary-structure.txt");
    }

    public void parseFile(String path) {
        if (path == null || path.length() == 0) {
            System.err.println("File name not valid");
            System.exit(1);
        }

        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(path));
        } catch (FileNotFoundException ffe) {
            System.err.println("Cannot locate file "+ path);
            System.exit(1);
        }

        // We will set up a map to store the corresponding relationship between a amino acid and a numerical value
        Map<String, Double> aminoAcidMap = new HashMap<String, Double>();
        Map<String, Integer> labelMap = new HashMap<String, Integer>();
        Queue<Double> aminoSequenceInProtein = new LinkedList<Double>();
        Queue<Double> slidingWindow = new LinkedList<Double>();
        Queue<String> labelQueue = new LinkedList<String>();
        aminoAcidMap.put("#", 0.0);

        int countOfProtein = 0;

        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // skip empty lines and comments, also end
            if (line.length() == 0 || line.startsWith("#") || line.startsWith("end") || line.startsWith("<end>")) {
                continue;
            }

            if (line.startsWith("<>")) {
                // the start of one protein

                // first, we need to proceed the previous protein information
                makeInstanceFromPreviousSequence(aminoAcidMap, labelMap,
                        aminoSequenceInProtein, slidingWindow, labelQueue, countOfProtein);

                ++countOfProtein;

                // then we prepare for the new protein
                slidingWindow.clear();
                // pad 8 # amino at the beginning
                for (int i=0; i < 8; ++i) {
                    aminoSequenceInProtein.offer(aminoAcidMap.get("#"));
                }
            } else {
                String[] tokens = line.split(" ");
                String amino = tokens[0];
                String label = tokens[1];
                if (!aminoAcidMap.containsKey(amino)) {
                    aminoAcidMap.put(amino, new Double(aminoAcidMap.size()));
                }
                aminoSequenceInProtein.offer(aminoAcidMap.get(amino));
                if (!labelMap.containsKey(label)) {
                    labelMap.put(label, labelMap.size());
                    this.reverseLabelMap.put(reverseLabelMap.size(), label);
                }
                labelQueue.offer(label);
            }
        }

        makeInstanceFromPreviousSequence(aminoAcidMap, labelMap,
                aminoSequenceInProtein, slidingWindow, labelQueue, countOfProtein);
    }

    private void makeInstanceFromPreviousSequence(Map<String, Double> aminoAcidMap, Map<String, Integer> labelMap,
                                                  Queue<Double> aminoSequenceInProtein, Queue<Double> slidingWindow,
                                                  Queue<String> labelQueue, int countOfProtein) {
        if (!aminoSequenceInProtein.isEmpty()) {
            // pad 8 # amino at the end
            for (int i = 0; i < 8; ++i) {
                aminoSequenceInProtein.offer(aminoAcidMap.get("#"));
            }
            while (!aminoSequenceInProtein.isEmpty()) {
                slidingWindow.offer(aminoSequenceInProtein.poll());
                if (slidingWindow.size() == this.data_feature_size) {
                    // We will creat an instance from sliding window and then pop one amino from it
                    String label = labelQueue.poll();

                    List<Double> segment = new ArrayList<Double>(slidingWindow);

                    slidingWindow.poll();

                    Instance instance = new Instance(segment, 21, label, labelMap);
                    this.instanceList.add(instance);

                    if (countOfProtein % 6 == 5) {
                        this.tuningSet.add(instance);
                    } else if (countOfProtein % 6 == 0) {
                        this.testingSet.add(instance);
                    } else {
                        this.trainingSet.add(instance);
                    }
                }
            }
        }
    }
}

class HiddenLayer extends Layer {

    private final static boolean debug = false;

    public HiddenLayer(Layer prev, int output_size, int ACT_FLAG,
                       double learning_rate, double momentum, double weight_decay) {
        this.prevLayer = prev;
        this.prevLayer.nextLayer = this;
        this.nextLayer = null;

        this.input_size = this.prevLayer.output_size + 1;
        this.output_size = output_size;

        this.weightMat = new Matrix(this.input_size, this.output_size, Matrix.INITIALIZE_RANDOM);
        this.lastDw = new Matrix(this.input_size, this.output_size, Matrix.INITIALIZE_ZERO);
        this.gradMat = null;

        this.weightedSum = null;
        this.activationOutput = null;

        this.ACT_FLAG = ACT_FLAG;
        this.learning_rate = learning_rate;
        this.momentum = momentum;
        this.weight_decay = weight_decay;
    }

    @Override
    public void forward() {
        if (debug) {
            System.out.println("Weight of hidden layer");
            System.out.println(this.weightMat);
        }

        Vector input = prevLayer.activationOutput.addBias();

        this.weightedSum = input.transpose().matMul(this.weightMat).toVector();

        if (debug) {
            System.out.println("Weighted sum of hidden layer");
            System.out.println(this.weightedSum);
        }
    }

    @Override
    public void activate() {
        this.activationOutput = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);
        if (this.ACT_FLAG == Layer.ACT_SIGMOID) {
            for (int i = 0; i<this.activationOutput.dimension; ++i) {
                this.activationOutput.data[i][0] = 1.0/(1.0 + Math.pow(Math.E, -this.weightedSum.data[i][0]));
            }
        } else if (this.ACT_FLAG == Layer.ACT_RELU) {
            for (int i = 0; i<this.activationOutput.dimension; ++i) {
                this.activationOutput.data[i][0] = Math.max(0.0, this.weightedSum.data[i][0]);
            }
        } else if (this.ACT_FLAG == Layer.ACT_LINEAR) {
            for (int i = 0; i<this.activationOutput.dimension; ++i) {
                this.activationOutput.data[i][0] = this.weightedSum.data[i][0];
            }
        }

        if (debug) {
            System.out.println("Activation of hidden layer");
            System.out.println(this.activationOutput);
        }
    }

    @Override
    public void back() {
        Matrix sumMat = this.nextLayer.weightMat
                .matElementWiseMul(this.nextLayer.gradMat);
        // System.out.println(sumMat);
        Vector sumVec = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);
        for (int i=0; i<sumMat.x_dimension-1; ++i) {
            double sum = 0.0;
            for (int j=0; j<sumMat.y_dimension; ++j) {
                sum += sumMat.data[i][j];
            }
            sumVec.data[i][0] = sum;
        }

        if (this.ACT_FLAG == Layer.ACT_RELU) {
            Vector gradVec = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);
            for (int i=0; i<this.output_size; ++i) {
                if (this.activationOutput.data[i][0] > 0) {
                    gradVec.data[i][0] = 1;
                }
            }

            Vector mulVec = gradVec.matElementWiseMul(sumVec).toVector();
            this.gradMat = mulVec.extendHerizontallyToMat(this.input_size);
            // System.out.println(this.gradMat);
            this.gradMat = this.gradMat.transpose();
        } else if (this.ACT_FLAG == Layer.ACT_SIGMOID) {
            // TODO
            Vector gradVec = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);
            for (int i=0; i<this.output_size; ++i) {
                gradVec.data[i][0] = this.activationOutput.data[i][0] * (1 - this.activationOutput.data[i][0]);
            }

            Vector mulVec = gradVec.matElementWiseMul(sumVec).toVector();
            this.gradMat = mulVec.extendHerizontallyToMat(this.input_size);

            this.gradMat = this.gradMat.transpose();

        } else if (this.ACT_FLAG == Layer.ACT_LINEAR) {
            // TODO

        }

        if (debug) {
            System.out.println("grad mat of hidden layer");
            System.out.println(this.gradMat);
        }
    }

    @Override
    public void adjustWeights() {
        Matrix delta = this.dw();
        if (debug) {
            System.out.println("dw of hidden layer");
            System.out.println(delta);
        }
        this.weightMat = this.weightMat.matAdd(delta);

        // momentum
        this.weightMat = this.weightMat.matAdd(this.lastDw.matScalarMul(this.momentum));
        this.lastDw = delta;
    }

    @Override
    protected Matrix dw() {
        Vector input = prevLayer.activationOutput.addBias();
        Matrix result = new Matrix(this.input_size, this.output_size, Matrix.INITIALIZE_ZERO);

        // System.out.println(this.gradMat);

        for (int i=0; i<result.x_dimension; ++i) {
            for (int j=0; j<result.y_dimension; ++j) {
                result.data[i][j] = (input.data[i][0] * this.gradMat.data[i][j]) * learning_rate;
            }
        }

        // weight_decay

        for (int i=0; i<result.x_dimension; ++i) {
            for (int j=0; j<result.y_dimension; ++j) {
                result.data[i][j] = (result.data[i][j] - learning_rate * weight_decay * this.weightMat.data[i][j]);
            }
        }

        return result;
    }
}

interface ILayer {

    // forward propagation to calculate the output, without activation
    public void forward();

    // activate
    public void activate();

    // back propagation
    public void back();

    // apply weight adjustment
    public void adjustWeights();
}

class InputLayer extends Layer {

    Vector inputFeature;

    public InputLayer(int feature_size) {
        this.inputFeature = null;

        this.prevLayer = null;
        this.nextLayer = null;

        this.input_size = feature_size;
        this.output_size = this.input_size;

        this.weightMat = null;
        this.lastDw = null;
        this.gradMat = null;

        this.weightedSum = null;
        this.activationOutput = null;

        this.ACT_FLAG = Layer.ACT_LINEAR;
        this.learning_rate = 0.0;
        this.momentum = 0.0;
        this.weight_decay = 0.0;
    }

    public void feedInput(Vector inputFeature) {
        if (this.input_size != inputFeature.dimension) {
            System.err.println("Feeding input error! Dimensions must be the same");
            System.exit(1);
        }
        this.inputFeature = inputFeature;
    }

    @Override
    public void forward() {
        this.weightedSum = this.inputFeature;
    }

    @Override
    public void activate() {
        this.activationOutput = this.inputFeature;
    }

    @Override
    public void back() {

    }

    @Override
    public void adjustWeights() {

    }

    @Override
    protected Matrix dw() {
        return null;
    }
}

class Instance {
    Vector feature;
    Vector label;

    public Instance(List<Double> featureList, int feature_value_num,
                    String labelString, Map<String, Integer> labelMap) {

        this.feature = new Vector(featureList.size() * feature_value_num, Matrix.INITIALIZE_ZERO);
        for (int i=0; i<featureList.size(); ++i) {
            for (int j=0; j<feature_value_num; ++j) {
                if (featureList.get(i) == j) {
                    this.feature.data[i * feature_value_num + j][0] = 1;
                }
            }
        }

        this.label = new Vector(labelMap.size(), Matrix.INITIALIZE_ZERO);
        int ind = labelMap.get(labelString);
        this.label.data[ind][0] = 1.0;
    }

    public int featureSize() {
        return this.feature.dimension;
    }

    public int labelSize() {
        return this.label.dimension;
    }

    @Override
    public String toString() {
        String ret = "";
        ret += "Feature: ";
        ret += feature.toString();

        ret += "Label: ";
        ret += label.toString();

        return ret;
    }
}

abstract class Layer implements ILayer{
    int input_size;
    int output_size;
    Matrix weightMat;
    Matrix gradMat;
    Matrix lastDw;
    Vector weightedSum;
    Vector activationOutput;
    int ACT_FLAG;
    double learning_rate;
    double momentum;
    double weight_decay;

    Layer prevLayer;
    Layer nextLayer;

    static final int ACT_LINEAR = 0;
    static final int ACT_SIGMOID = 1;
    static final int ACT_RELU = 2;

    // calculate delta w
    protected abstract Matrix dw();
}

class Matrix {

    int x_dimension;
    int y_dimension;
    double[][] data;

    static int INITIALIZE_ZERO = 0;
    static int INITIALIZE_ONE = 1;
    static int INITIALIZE_RANDOM = 2;

    public Matrix(int x_dimension, int y_dimension, int INITIALIZE_FLAG) {
        this.x_dimension = x_dimension;
        this.y_dimension = y_dimension;

        data = new double[x_dimension][y_dimension];

        if (INITIALIZE_FLAG == this.INITIALIZE_ONE) {
            for (int i=0; i<x_dimension; ++i) {
                for (int j=0; j<y_dimension; ++j) {
                    data[i][j] = 1.0;
                }
            }
        }

        if (INITIALIZE_FLAG == this.INITIALIZE_RANDOM) {
            Random r = new Random();
            for (int i=0; i<x_dimension; ++i) {
                for (int j=0; j<y_dimension; ++j) {
                    data[i][j] = -1 + (1-(-1)) * r.nextDouble();
                }
            }
        }
    }

    public Matrix(Matrix m) {
        this.x_dimension = m.x_dimension;
        this.y_dimension = m.y_dimension;

        this.data = new double[this.x_dimension][this.y_dimension];

        for (int i=0; i<this.x_dimension; ++i) {
            for (int j=0; j<this.y_dimension; ++j) {
                this.data[i][j] = m.data[i][j];
            }
        }
    }

    public Matrix matMul(Matrix B) {

        Matrix A = this;
        if (A.y_dimension != B.x_dimension) {
            System.err.println("Mat Mul Dimension Error");
            System.exit(1);
        }

        Matrix result = new Matrix(A.x_dimension, B.y_dimension, 0);

        for (int i=0; i<A.x_dimension; ++i) {
            for (int j=0; j<B.y_dimension; ++j) {
                double tmpSum = 0.0;
                for (int k=0; k<A.y_dimension; ++k) {
                    tmpSum += A.data[i][k] * B.data[k][j];
                }
                result.data[i][j] = tmpSum;
            }
        }

        return result;
    }

    public Matrix matAdd(Matrix B) {

        Matrix A = this;
        if (A.x_dimension != B.x_dimension || A.y_dimension != B.y_dimension) {
            System.err.println("Mat Add Dimension Error");
            System.exit(1);
        }

        Matrix result = new Matrix(A.x_dimension, A.y_dimension, 0);

        for (int i=0; i<A.x_dimension; ++i) {
            for (int j=0; j<A.y_dimension; ++j) {
                result.data[i][j] = A.data[i][j] + B.data[i][j];
            }
        }

        return result;
    }

    public Matrix matSub(Matrix B) {

        Matrix A = this;
        if (A.x_dimension != B.x_dimension || A.y_dimension != B.y_dimension) {
            System.err.println("Mat Sub Dimension Error");
            System.exit(1);
        }

        Matrix result = new Matrix(A.x_dimension, A.y_dimension, 0);

        for (int i=0; i<A.x_dimension; ++i) {
            for (int j=0; j<A.y_dimension; ++j) {
                result.data[i][j] = A.data[i][j] - B.data[i][j];
            }
        }

        return result;
    }

    public Matrix matElementWiseMul(Matrix B) {

        Matrix A = this;

        if (A.x_dimension != B.x_dimension || A.y_dimension != B.y_dimension) {
            System.err.println("Mat Element-wise Mul Dimension Error");
            System.exit(1);
        }

        Matrix result = new Matrix(A.x_dimension, A.y_dimension, 0);

        for (int i=0; i<A.x_dimension; ++i) {
            for (int j=0; j<A.y_dimension; ++j) {
                result.data[i][j] = A.data[i][j] * B.data[i][j];
            }
        }

        return result;
    }

    public Matrix matScalarMul(double scalar) {

        Matrix A = this;

        Matrix result = new Matrix(A.x_dimension, A.y_dimension, 0);

        for (int i=0; i<A.x_dimension; ++i) {
            for (int j=0; j<A.y_dimension; ++j) {
                result.data[i][j] = A.data[i][j] * scalar;
            }
        }

        return result;
    }

    public Matrix transpose() {

        Matrix A = this;

        Matrix result = new Matrix(A.y_dimension, A.x_dimension, 0);

        for (int i=0; i<A.y_dimension; ++i) {
            for (int j=0; j<A.x_dimension; ++j) {
                result.data[i][j] = A.data[j][i];
            }
        }

        return result;
    }

    public double findMax() {
        double max = Double.MIN_VALUE;
        for (int i=0; i<this.x_dimension; ++i) {
            for (int j=0; j<this.y_dimension; ++j) {
                max = Math.max(max, this.data[i][j]);
            }
        }
        return max;
    }

    public Vector toVector() {
        if (this.x_dimension != 1 && this.y_dimension != 1) {
            System.err.println("You cannot cast this matrix to a vector");
            System.exit(1);
        }
        int dimension = this.x_dimension >= this.y_dimension ? this.x_dimension : this.y_dimension;

        Vector result = new Vector(dimension, Matrix.INITIALIZE_ZERO);

        int ind = 0;
        for (int i=0; i<this.x_dimension; ++i) {
            for (int j=0; j<this.y_dimension; ++j) {
                result.data[ind][0] = this.data[i][j];
                ++ind;
            }
        }

        return result;
    }

    @Override
    public String toString() {
        String ret = "";
        ret += ("X Dimension: "+x_dimension+", Y Dimension: "+y_dimension+"\n");
        for (int i=0; i<this.x_dimension; ++i) {
            for (int j=0; j<this.y_dimension; ++j) {
                ret += (this.data[i][j]+" ");
            }
            ret += "\n";
        }
        return ret;
    }

    @Override
    public boolean equals(Object object) {
        if (object instanceof Matrix) {
            Matrix A = this;
            Matrix B = (Matrix)object;

            if (A.x_dimension != B.x_dimension || A.y_dimension != B.y_dimension) {
                return false;
            } else {
                for (int i=0; i<A.x_dimension; ++i) {
                    for (int j=0; j<A.y_dimension; ++j) {
                        if (A.data[i][j] != B.data[i][j]) {
                            return false;
                        }
                    }
                }
                return true;
            }
        } else {
            return false;
        }
    }
}

class NeuralNet {

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

class OutputLayer extends Layer {

    private final static boolean debug = false;

    Vector label;
    Vector finalOutput;

    public OutputLayer(Layer prev, int output_size, int ACT_FLAG,
                       double learning_rate, double momentum, double weight_decay) {
        this.label = null;
        this.finalOutput = null;

        this.prevLayer = prev;
        this.prevLayer.nextLayer = this;
        this.nextLayer = null;

        // because we want to use bias, so we would like to plus 1 here
        this.input_size = this.prevLayer.output_size + 1;
        this.output_size = output_size;

        this.weightMat = new Matrix(this.input_size, this.output_size, Matrix.INITIALIZE_RANDOM);
        this.lastDw = new Matrix(this.input_size, this.output_size, Matrix.INITIALIZE_ZERO);
        this.gradMat = null;

        this.weightedSum = null;
        this.activationOutput = null;

        this.ACT_FLAG = ACT_FLAG;
        this.learning_rate = learning_rate;
        this.momentum = momentum;
        this.weight_decay = weight_decay;
    }

    @Override
    public void forward() {
        if (debug) {
            System.out.println("Weight of output layer");
            System.out.println(this.weightMat);
        }

        Vector input = prevLayer.activationOutput.addBias();

        this.weightedSum = input.transpose().matMul(this.weightMat).toVector();

        if (debug) {
            System.out.println("Weighted sum of output layer");
            System.out.println(this.weightedSum);
        }
    }

    @Override
    public void activate() {
        this.activationOutput = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);
        if (this.ACT_FLAG == Layer.ACT_SIGMOID) {
            for (int i = 0; i<this.activationOutput.dimension; ++i) {
                this.activationOutput.data[i][0] = 1.0/(1.0 + Math.pow(Math.E, -this.weightedSum.data[i][0]));
            }
        } else if (this.ACT_FLAG == Layer.ACT_RELU) {
            for (int i = 0; i<this.activationOutput.dimension; ++i) {
                this.activationOutput.data[i][0] = Math.max(0.0, this.weightedSum.data[i][0]);
            }
        } else if (this.ACT_FLAG == Layer.ACT_LINEAR) {
            for (int i = 0; i<this.activationOutput.dimension; ++i) {
                this.activationOutput.data[i][0] = this.weightedSum.data[i][0];
            }
        }

        if (debug) {
            System.out.println("activation of output layer");
            System.out.println(this.activationOutput);
        }
    }

    public void calOutput() {
        this.finalOutput = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);
        double maxVal = this.activationOutput.findMax();

        for (int i=0; i<this.output_size; ++i) {
            if (maxVal == this.activationOutput.data[i][0]) {
                this.finalOutput.data[i][0] = 1.0;
                break;
            }
        }

        if (debug) {
            System.out.println("final output of output layer");
            System.out.println(this.finalOutput);
        }
    }

    public void feedLabel(Vector label) {
        if (this.output_size != label.dimension) {
            System.err.println("Feeding label error! Dimensions must be the same");
            System.exit(1);
        }
        this.label = label;
    }

    public boolean isCorrect() {
        return this.label.equals(this.finalOutput);
    }

    @Override
    public void back() {
        this.gradMat = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);

        if (this.ACT_FLAG == Layer.ACT_SIGMOID) {
            Vector one_v = new Vector(this.output_size, Matrix.INITIALIZE_ONE);
            this.gradMat = (this.label.matSub(this.activationOutput))
                    .matElementWiseMul(this.activationOutput)
                    .matElementWiseMul(one_v.matSub(this.activationOutput));
            // System.out.println(this.gradMat);
            this.gradMat = this.gradMat.toVector().extendHerizontallyToMat(this.input_size);
            // transpose to an m*n matrix
            this.gradMat = this.gradMat.transpose();
        } else if (this.ACT_FLAG == Layer.ACT_RELU) {
            // TODO

        } else if (this.ACT_FLAG == Layer.ACT_LINEAR) {
            // TODO

        }

        if (debug) {
            System.out.println("grad mat of output layer");
            System.out.println(this.gradMat);
        }
    }

    @Override
    public void adjustWeights() {
        Matrix delta = this.dw();
        if (debug) {
            System.out.println("dw of output layer");
            System.out.println(delta);
        }
        this.weightMat = this.weightMat.matAdd(delta);

        // momentum
        this.weightMat = this.weightMat.matAdd(this.lastDw.matScalarMul(this.momentum));
        this.lastDw = delta;
    }

    @Override
    protected Matrix dw() {
        Vector input = prevLayer.activationOutput.addBias();
        Matrix result = new Matrix(this.input_size, this.output_size, Matrix.INITIALIZE_ZERO);

        for (int i=0; i<result.x_dimension; ++i) {
            for (int j=0; j<result.y_dimension; ++j) {
                result.data[i][j] = (input.data[i][0] * this.gradMat.data[i][j]) * learning_rate;
            }
        }

        // weight_decay

        for (int i=0; i<result.x_dimension; ++i) {
            for (int j=0; j<result.y_dimension; ++j) {
                result.data[i][j] = (result.data[i][j] - learning_rate * weight_decay * this.weightMat.data[i][j]);
            }
        }

        return result;
    }
}

class Vector extends Matrix {
    int dimension;

    public Vector(int dimension, int INITIALIZE_FLAG) {
        super(dimension, 1, INITIALIZE_FLAG);
        this.dimension = dimension;
    }

    public Vector(Vector v) {
        super(v);
        this.dimension = v.dimension;
    }

    public double getElementAt(int index) {
        if (index >= dimension) {
            System.err.println("Index out of range");
            System.exit(1);
        }

        if (this.x_dimension >= y_dimension) {
            return this.data[index][0];
        } else {
            return this.data[0][index];
        }
    }

    // extend the vector bias
    public Vector addBias() {
        Vector result = new Vector(this.dimension+1, Matrix.INITIALIZE_ZERO);
        for (int i=0; i<this.dimension; ++i) {
            result.data[i][0] = this.getElementAt(i);
        }
        result.data[this.dimension][0] = 1;
        return result;
    }

    public Matrix extendHerizontallyToMat(int col_size) {
        Matrix result = new Matrix(this.dimension, col_size, Matrix.INITIALIZE_ZERO);

        for (int i=0; i<result.x_dimension; ++i) {
            for (int j=0; j<result.y_dimension; ++j) {
                result.data[i][j] = this.data[i][0];
            }
        }

        return result;
    }

}