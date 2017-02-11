/**
 * Created by ClarkWong on 8/2/17.
 */
public class Lab2 {

    public static void main(String[] args) {
        DataParser dp = new DataParser();
        dp.parseFile("data/protein-secondary-structure.txt");

        NeuralNet neuralNet = new NeuralNet(dp.data_feature_size, dp.data_label_size, 1, new int[]{20},
                Layer.ACT_RELU, Layer.ACT_SIGMOID, 0.01, dp.trainingSet, dp.tuningSet, dp.testingSet);

        for (int i=0; i<20; ++i) {
            double learningAccuracy = neuralNet.trainOneEpoch();
            System.out.println("Learing accuracy: "+learningAccuracy);
            double tunningAccuracy = neuralNet.tuneAccuracy();
            System.out.println("Tuning accuracy: "+tunningAccuracy);
            double testingAccuracy = neuralNet.testAccuracy();
            System.out.println("Testing accuracy: "+testingAccuracy);
            System.out.println();
        }
    }

}
