/**
 * Created by chenwang on 2/9/17.
 */
public class OutputLayer extends Layer {

    Vector label;
    Vector finalOutput;

    public OutputLayer(Vector label, Layer prev, int output_size, int ACT_FLAG, double learning_rate) {
        this.label = label;
        this.finalOutput = null;

        this.prevLayer = prev;
        this.nextLayer = null;

        // because we want to use bias, so we would like to plus 1 here
        this.input_size = this.prevLayer.output_size + 1;
        this.output_size = output_size;

        this.weightMat = new Matrix(this.input_size, this.output_size, Matrix.INITIALIZE_RANDOM);
        this.gradMat = null;

        this.weightedSum = null;
        this.activationOutput = null;

        this.ACT_FLAG = ACT_FLAG;
        this.learning_rate = learning_rate;
    }

    @Override
    public void forward() {
        Vector input = prevLayer.activationOutput.addBias();

        this.weightedSum = input.transpose().matMul(this.weightMat).toVector();
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
    }

    @Override
    public void back() {
        this.gradMat = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);

        if (this.ACT_FLAG == Layer.ACT_SIGMOID) {
            Vector one_v = new Vector(this.output_size, Matrix.INITIALIZE_ONE);
            this.gradMat = (this.label.matSub(this.finalOutput))
                    .matElementWiseMul(this.finalOutput)
                    .matElementWiseMul(one_v.matSub(this.finalOutput));
            this.gradMat = this.gradMat.toVector().extendHerizontallyToMat(this.input_size);
            // transpose to an m*n matrix
            this.gradMat = this.gradMat.transpose();
        } else if (this.ACT_FLAG == Layer.ACT_RELU) {
            // TODO

        } else if (this.ACT_FLAG == Layer.ACT_LINEAR) {
            // TODO

        }
    }

    @Override
    public void adjustWeights() {
        this.weightMat = this.weightMat.matAdd(this.dw());
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

        return result;
    }
}
