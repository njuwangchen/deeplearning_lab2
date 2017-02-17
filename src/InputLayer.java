
/**
 * Created by chenwang on 2/10/17.
 */
public class InputLayer extends Layer {

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
