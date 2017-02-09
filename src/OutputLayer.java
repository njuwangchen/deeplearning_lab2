/**
 * Created by chenwang on 2/9/17.
 */
public class OutputLayer extends Layer {

    Vector label;

    public OutputLayer(Vector label, Layer prev, int output_size, int ACT_FLAG, double learning_rate) {
        this.label = label;

        this.prevLayer = prev;
        this.nextLayer = null;

        // because we want to use bias, so we would like to plus 1 here
        this.input_size = this.prevLayer.output_size + 1;
        this.output_size = output_size;

        this.weightMat = new Matrix(input_size, output_size, Matrix.INITIALIZE_RANDOM);
        this.gradMat = null;

        this.weightedSum = null;
        this.activationOutput = null;

        this.ACT_FLAG = ACT_FLAG;
        this.learning_rate = learning_rate;
    }

    @Override
    public void forward() {

    }

    @Override
    public void activate() {

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
