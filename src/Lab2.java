/**
 * Created by ClarkWong on 8/2/17.
 */
public class Lab2 {

    public static void main(String[] args) {
        if (args.length != 6) {
            System.err.println("Usage: Lab2 Filename hu learning_rate momentum weight_decay hidden_act");
            System.exit(1);
        }

        String fileName = args[0];
        int hu = Integer.parseInt(args[1]);
        float learning_rate = Float.parseFloat(args[2]);
        float momentum = Float.parseFloat(args[3]);
        float weight_decay = Float.parseFloat(args[4]);

        String hidden_act_str = args[5];
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
