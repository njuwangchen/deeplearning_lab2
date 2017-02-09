import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/**
 * Created by chenwang on 2/8/17.
 */
public class DataParser {

    public static void main(String[] args) {
        DataParser dp = new DataParser();
        dp.parseFile("data/protein-secondary-structure.train.txt");
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
        Queue<Double> aminoSequenceInProtein = new LinkedList<Double>();
        Queue<Double> slidingWindow = new LinkedList<Double>();
        Queue<String> labelQueue = new LinkedList<String>();
        aminoAcidMap.put("#", 0.0);

        while(fileScanner.hasNext()) {
            String line = fileScanner.nextLine().trim();

            // skip empty lines and comments, also end
            if (line.length() == 0 || line.startsWith("#") || line.startsWith("end") || line.startsWith("<end>")) {
                continue;
            }

            if (line.startsWith("<>")) {
                // the start of one protein

                // first, we need to proceed the previous protein information
                makeInstanceFromPreviousSequence(aminoAcidMap, aminoSequenceInProtein, slidingWindow, labelQueue);

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
                labelQueue.offer(label);
            }
        }

        makeInstanceFromPreviousSequence(aminoAcidMap, aminoSequenceInProtein, slidingWindow, labelQueue);
    }

    private void makeInstanceFromPreviousSequence(Map<String, Double> aminoAcidMap,
                                                  Queue<Double> aminoSequenceInProtein, Queue<Double> slidingWindow,
                                                  Queue<String> labelQueue) {
        if (!aminoSequenceInProtein.isEmpty()) {
            // pad 8 # amino at the end
            for (int i = 0; i < 8; ++i) {
                aminoSequenceInProtein.offer(aminoAcidMap.get("#"));
            }
            while (!aminoSequenceInProtein.isEmpty()) {
                slidingWindow.offer(aminoSequenceInProtein.poll());
                if (slidingWindow.size() == 17) {
                    // We will creat an instance from sliding window and then pop one amino from it
                    String label = labelQueue.poll();
                    List<Double> segment = new ArrayList<Double>(slidingWindow);
                    slidingWindow.poll();

                    for (Double d: segment) {
                        System.out.print(d+", ");
                    }
                    System.out.println(label);
                }
            }
        }
    }
}
