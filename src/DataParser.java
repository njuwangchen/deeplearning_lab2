import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/**
 * Created by chenwang on 2/8/17.
 */
public class DataParser {

    List<Instance> instanceList;

    List<Instance> trainingSet;
    List<Instance> tuningSet;
    List<Instance> testingSet;

    int data_feature_size;
    int data_label_size;
    int feature_value_num;

    public DataParser() {
        this.instanceList = new ArrayList<Instance>();

        this.trainingSet = new ArrayList<Instance>();
        this.tuningSet = new ArrayList<Instance>();
        this.testingSet = new ArrayList<Instance>();

        this.data_feature_size = 17;
        this.data_label_size = 3;
        this.feature_value_num = 21;
    }

    public static void main(String[] args) {
        DataParser dp = new DataParser();
        dp.parseFile("data/protein-secondary-structure.txt");
    }

    public void parseFile(String path) {
        if (path == null || path.length() == 0) {
            System.err.println("File name not valid");
            System.exit(1);
        }

        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(path));
        } catch (FileNotFoundException ffe) {
            System.err.println("Cannot locate file "+ path);
            System.exit(1);
        }

        // We will set up a map to store the corresponding relationship between a amino acid and a numerical value
        Map<String, Double> aminoAcidMap = new HashMap<String, Double>();
        Map<String, Integer> labelMap = new HashMap<String, Integer>();
        Queue<Double> aminoSequenceInProtein = new LinkedList<Double>();
        Queue<Double> slidingWindow = new LinkedList<Double>();
        Queue<String> labelQueue = new LinkedList<String>();
        aminoAcidMap.put("#", 0.0);

        int countOfProtein = 0;

        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // skip empty lines and comments, also end
            if (line.length() == 0 || line.startsWith("#") || line.startsWith("end") || line.startsWith("<end>")) {
                continue;
            }

            if (line.startsWith("<>")) {
                // the start of one protein

                // first, we need to proceed the previous protein information
                makeInstanceFromPreviousSequence(aminoAcidMap, labelMap,
                        aminoSequenceInProtein, slidingWindow, labelQueue, countOfProtein);

                ++countOfProtein;

                // then we prepare for the new protein
                slidingWindow.clear();
                // pad 8 # amino at the beginning
                for (int i=0; i < 8; ++i) {
                    aminoSequenceInProtein.offer(aminoAcidMap.get("#"));
                }
            } else {
                String[] tokens = line.split(" ");
                String amino = tokens[0];
                String label = tokens[1];
                if (!aminoAcidMap.containsKey(amino)) {
                    aminoAcidMap.put(amino, new Double(aminoAcidMap.size()));
                }
                aminoSequenceInProtein.offer(aminoAcidMap.get(amino));
                if (!labelMap.containsKey(label)) {
                    labelMap.put(label, labelMap.size());
                }
                labelQueue.offer(label);
            }
        }

        makeInstanceFromPreviousSequence(aminoAcidMap, labelMap,
                aminoSequenceInProtein, slidingWindow, labelQueue, countOfProtein);

        System.out.println(this.instanceList.size());
        System.out.println(countOfProtein);
        System.out.println(this.trainingSet.size());
        System.out.println(this.tuningSet.size());
        System.out.println(this.testingSet.size());
    }

    private void makeInstanceFromPreviousSequence(Map<String, Double> aminoAcidMap, Map<String, Integer> labelMap,
                                                  Queue<Double> aminoSequenceInProtein, Queue<Double> slidingWindow,
                                                  Queue<String> labelQueue, int countOfProtein) {
        if (!aminoSequenceInProtein.isEmpty()) {
            // pad 8 # amino at the end
            for (int i = 0; i < 8; ++i) {
                aminoSequenceInProtein.offer(aminoAcidMap.get("#"));
            }
            while (!aminoSequenceInProtein.isEmpty()) {
                slidingWindow.offer(aminoSequenceInProtein.poll());
                if (slidingWindow.size() == this.data_feature_size) {
                    // We will creat an instance from sliding window and then pop one amino from it
                    String label = labelQueue.poll();

                    List<Double> segment = new ArrayList<Double>(slidingWindow);

                    slidingWindow.poll();

                    Instance instance = new Instance(segment, 21, label, labelMap);
                    this.instanceList.add(instance);

                    if (countOfProtein % 6 == 5) {
                        this.tuningSet.add(instance);
                    } else if (countOfProtein % 6 == 0) {
                        this.testingSet.add(instance);
                    } else {
                        this.trainingSet.add(instance);
                    }
                }
            }
        }
    }
}
