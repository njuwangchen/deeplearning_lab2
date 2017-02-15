/**
 * Created by ClarkWong on 8/2/17.
 */
public class Lab2 {

    public static void main(String[] args) {
        DataParser dp = new DataParser();
        dp.parseFile("data/protein-secondary-structure.txt");

        NeuralNet neuralNet = new NeuralNet(dp.data_feature_size * dp.feature_value_num,
                dp.data_label_size, 1, new int[]{8},
                Layer.ACT_SIGMOID, Layer.ACT_SIGMOID,
                0.01, 0.9, dp.trainingSet, dp.tuningSet, dp.testingSet);

//        for (int i=0; i<100; ++i) {
//            double learningAccuracy = neuralNet.trainOneEpoch();
//            System.out.println("Learning accuracy: "+learningAccuracy);
//            double tunningAccuracy = neuralNet.tuneAccuracy();
//            System.out.println("Tuning accuracy: "+tunningAccuracy);
//            double testingAccuracy = neuralNet.testAccuracy();
//            System.out.println("Testing accuracy: "+testingAccuracy);
//            System.out.println();
//        }

        neuralNet.startTrainingWithEarlyStopping(10);
    }

}
