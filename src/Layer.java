/**
 * Created by chenwang on 2/9/17.
 */
public abstract class Layer implements ILayer{
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
