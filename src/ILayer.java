/**
 * Created by chenwang on 2/9/17.
 */
public interface ILayer {

    // forward propagation to calculate the output, without activation
    public void forward();

    // activate
    public void activate();

    // back propagation
    public void back();

    // apply weight adjustment
    public void adjustWeights();
}
