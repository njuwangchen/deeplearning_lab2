/**
 * Created by chenwang on 2/10/17.
 */
public class HiddenLayer extends Layer {

    public HiddenLayer(Layer prev, int output_size, int ACT_FLAG, double learning_rate) {
        this.prevLayer = prev;
        this.prevLayer.nextLayer = this;
        this.nextLayer = null;

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

    @Override
    public void back() {
        Matrix sumMat = this.nextLayer.weightMat
                .matElementWiseMul(this.nextLayer.gradMat);
        Vector sumVec = new Vector(this.output_size, Matrix.INITIALIZE_ZERO);
        for (int i=0; i<sumMat.x_dimension; ++i) {
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
            this.gradMat = this.gradMat.transpose();
        } else if (this.ACT_FLAG == Layer.ACT_SIGMOID) {
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
