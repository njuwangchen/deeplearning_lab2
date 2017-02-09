/**
 * Created by chenwang on 2/9/17.
 */
public class Vector extends Matrix {
    int dimension;

    public Vector(int dimension, int INITIALIZE_FLAG) {
        super(dimension, 1, INITIALIZE_FLAG);
        this.dimension = dimension;
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

}
