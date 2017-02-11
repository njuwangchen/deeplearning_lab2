import java.util.List;
import java.util.Map;

/**
 * Created by chenwang on 2/10/17.
 */
public class Instance {
    Vector feature;
    Vector label;

    public Instance(List<Double> featureList, String labelString, Map<String, Integer> labelMap) {
        this.feature = new Vector(featureList.size(), Matrix.INITIALIZE_ZERO);
        for (int i=0; i<featureList.size(); ++i) {
            this.feature.data[i][0] = featureList.get(i);
        }

        this.label = new Vector(labelMap.size(), Matrix.INITIALIZE_ZERO);
        int ind = labelMap.get(labelString);
        this.label.data[ind][0] = 1.0;
    }

    public int featureSize() {
        return this.feature.dimension;
    }

    public int labelSize() {
        return this.label.dimension;
    }

    @Override
    public String toString() {
        String ret = "";
        ret += "Feature: ";
        ret += feature.toString();

        ret += "Label: ";
        ret += label.toString();

        return ret;
    }
}
