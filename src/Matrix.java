import java.util.Random;

/**
 * Created by chenwang on 2/9/17.
 */
public class Matrix {

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

        if (INITIALIZE_FLAG == 1) {
            for (int i=0; i<x_dimension; ++i) {
                for (int j=0; j<y_dimension; ++j) {
                    data[i][j] = 1.0;
                }
            }
        }

        if (INITIALIZE_FLAG == 2) {
            Random r = new Random();
            for (int i=0; i<x_dimension; ++i) {
                for (int j=0; j<y_dimension; ++j) {
                    data[i][j] = -1 + (1-(-1)) * r.nextDouble();
                }
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

    public Matrix matSub(Matrix B) {

        Matrix A = this;
        if (A.x_dimension != B.x_dimension || A.y_dimension != B.y_dimension) {
            System.err.println("Mat Mul Dimension Error");
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
            System.err.println("Mat Mul Dimension Error");
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


    public static void main(String[] args) {
        Matrix A = new Matrix(9,8,2);
        System.out.println(A);
        Matrix B = new Matrix(8, 2, 1);
        System.out.println(B.transpose());
        System.out.println(A.matMul(B));
        Vector C = new Vector(10,1);
        System.out.println(C);
        System.out.println(C.transpose());
        System.out.println(C.transpose().matMul(C));
    }
}
