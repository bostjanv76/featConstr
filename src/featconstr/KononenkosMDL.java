package featconstr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import weka.core.ContingencyTables;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Range;
import weka.core.SparseInstance;
import weka.core.SpecialFunctions;
import weka.core.Utils;
import weka.filters.Filter;

/**
 *
 * @author bostjan
 */
@SuppressWarnings("serial")
public class KononenkosMDL extends Filter{
    Instances data;
    public KononenkosMDL(){
    }
  
    public KononenkosMDL(Instances data){
        this.data=data;
        m_DiscretizeCols.setUpper(data.numAttributes() - 1);
    }
    
    //Output binary attributes for discretized attributes.
    protected boolean m_MakeBinary = false;
    //Use Kononenko's MDL criterion instead of Fayyad et al.'s
    protected boolean m_UseKononenko = false;
    //Stores which columns to Discretize
    protected Range m_DiscretizeCols = new Range();
    //Store the current cutpoints
    protected double[][] m_CutPoints = null;
    //Use better encoding of split point for MDL.
    protected boolean m_UseBetterEncoding = false;
/**
    * Set cutpoints for a single attribute using MDL.
    * 
    * @param index the index of the attribute to set cutpoints for
    * @param data the data to work with
*/
    public boolean input(Instance instance){
        if (getInputFormat() == null){
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_NewBatch){
            resetQueue();
            m_NewBatch = false;
        }

        if (m_CutPoints != null){
            convertInstance(instance);
        return true;
        }

        bufferInput(instance);
        
        return false;
    }
   
/**
    * Convert a single instance over. The converted instance is added to the end
    * of the output queue.
    * 
    * @param instance the instance to convert
*/
    protected void convertInstance(Instance instance){
        int index = 0;
        double[] vals = new double[outputFormatPeek().numAttributes()];
        // Copy and convert the values
        for (int i = 0; i < getInputFormat().numAttributes(); i++){
          if (m_DiscretizeCols.isInRange(i) && getInputFormat().attribute(i).isNumeric()){
            int j;
            double currentVal = instance.value(i);
            if (m_CutPoints[i] == null){
              if (instance.isMissing(i)){
                vals[index] = Utils.missingValue();
              } 
              else{
                vals[index] = 0;
              }
              index++;
            } 
            else{
              if (!m_MakeBinary){
                if (instance.isMissing(i)){
                  vals[index] = Utils.missingValue();
                } 
                else{
                  for (j = 0; j < m_CutPoints[i].length; j++){
                    if(currentVal <= m_CutPoints[i][j]){
                        break;
                    }
                  }
                  vals[index] = j;
                }
                index++;
              } else {
                for (j = 0; j < m_CutPoints[i].length; j++) {
                  if (instance.isMissing(i)) {
                    vals[index] = Utils.missingValue();
                  } else if (currentVal <= m_CutPoints[i][j]) {
                    vals[index] = 0;
                  } else {
                    vals[index] = 1;
                  }
                  index++;
                }
              }
            }
          } 
          else{
            vals[index] = instance.value(i);
            index++;
          }
        }

        Instance inst = null;
        if (instance instanceof SparseInstance){
          inst = new SparseInstance(instance.weight(), vals);
        } 
        else {
          inst = new DenseInstance(instance.weight(), vals);
        }

        copyValues(inst, false, instance.dataset(), outputFormatPeek());

        push(inst); //No need to copy instance
    } 
   
    public void printCutPoints(){
        for(int i=0; i<m_CutPoints.length; i++){
            for(int j=0; j<m_CutPoints[i].length;j++){
                System.out.printf("%8.3f",m_CutPoints[i][j]);
              }
              System.out.println();
          }
    }
  
    protected void calculateCutPointsByMDL(int index, Instances data) {
        //Sort instances
        data.sort(data.attribute(index));
        m_CutPoints = new double[data.numAttributes()][];
        //Find first instances that's missing
        int firstMissing = data.numInstances();
        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).isMissing(index)) {
                firstMissing = i;
            break;
            }
        }
        System.out.println("firstMissing: "+firstMissing);
    
        m_CutPoints[index] = cutPointsForSubset(data, index, 0, firstMissing);
        System.out.println("idx: "+index+" value: "+m_CutPoints[index]);
    }
    
    // Generate the cutpoints for each attribute
    protected void calculateCutPoints(){
        Instances copy = null;
        m_CutPoints = new double[data.numAttributes()][];
        for (int i = data.numAttributes() - 1; i >= 0; i--){
            if ((m_DiscretizeCols.isInRange(i)) && (data.attribute(i).isNumeric())){
                if (copy == null) {
                    copy = new Instances(data);
                }
            calculateCutPointsByMDL(i, copy);
            }      
        }
    }
  
/**
    * Gets the cut points for an attribute
    * 
    * @param attributeIndex the index (from 0) of the attribute to get the cut
    *          points of
    * @return an array containing the cutpoints (or null if the attribute
    *         requested isn't being Discretized
*/
    public double[] getCutPoints(int attributeIndex) {
      if (m_CutPoints == null) {
        return null;
      }
      return m_CutPoints[attributeIndex];
    }
    
/**
    * Selects cutpoints for sorted subset.
    * 
    * @param instances
    * @param attIndex
    * @param first
    * @param lastPlusOne
    * @return
*/
    public double[] cutPointsForSubset(Instances instances, int attIndex, int first, int lastPlusOne){
        double[][] counts, bestCounts;
        double[] priorCounts, left, right, cutPoints;
        double currentCutPoint = -Double.MAX_VALUE, bestCutPoint = -1, currentEntropy, bestEntropy, priorEntropy, gain;
        int bestIndex = -1, numCutPoints = 0;
        double numInstances = 0;

        //Compute number of instances in set
        if ((lastPlusOne - first) < 2) {
          return null;
        }

        //Compute class counts.
        counts = new double[2][instances.numClasses()];
        for (int i = first; i < lastPlusOne; i++){
            numInstances += instances.instance(i).weight();   
        //e.g. an instance with a weight of 2 corresponds to two identical instances with a weight of 1. (Instance weights are more flexible though because they don’t need to be integers.
        //Note that any instance without a weight value specified is assumed to have a weight of 1 for backwards compatibility.
            counts[1][(int) instances.instance(i).classValue()] += instances.instance(i).weight();
        }
    
        System.out.println("Method cutPointsForSubset num of instances: "+numInstances);

        //Save prior counts
        priorCounts = new double[instances.numClasses()];
        System.arraycopy(counts[1], 0, priorCounts, 0, instances.numClasses());     //copy elements from counts[l] to priorCounts

        // Entropy of the full set
        priorEntropy = ContingencyTables.entropy(priorCounts);
        bestEntropy = priorEntropy;

        //Find best entropy.
        bestCounts = new double[2][instances.numClasses()];
        for (int i = first; i < (lastPlusOne - 1); i++){
          counts[0][(int) instances.instance(i).classValue()] += instances.instance(i).weight();
          counts[1][(int) instances.instance(i).classValue()] -= instances.instance(i).weight();
          if (instances.instance(i).value(attIndex) < instances.instance(i + 1).value(attIndex)){
            currentCutPoint = (instances.instance(i).value(attIndex) + instances.instance(i + 1).value(attIndex)) / 2.0;
            currentEntropy = ContingencyTables.entropyConditionedOnRows(counts);
            if (currentEntropy < bestEntropy){
              bestCutPoint = currentCutPoint;
              bestEntropy = currentEntropy;
              bestIndex = i;
              System.arraycopy(counts[0], 0, bestCounts[0], 0,instances.numClasses());
              System.arraycopy(counts[1], 0, bestCounts[1], 0,instances.numClasses());
            }
            numCutPoints++;
          }
        }

        //Use worse encoding?
        if (!m_UseBetterEncoding) {
          numCutPoints = (lastPlusOne - first) - 1;
        }

        //Checks if gain is zero
        gain = priorEntropy - bestEntropy;
        if (gain <= 0) {
          return null;
        }

        //Check if split is to be accepted
        if((m_UseKononenko && kononenkosMDL(priorCounts, bestCounts, numInstances,numCutPoints)) || (!m_UseKononenko && fayyadAndIranisMDL(priorCounts, bestCounts, numInstances, numCutPoints))){
          //Select split points for the left and right subsets
          left = cutPointsForSubset(instances, attIndex, first, bestIndex + 1);
          right = cutPointsForSubset(instances, attIndex, bestIndex + 1, lastPlusOne);
          //Merge cutpoints and return them
            if ((left == null) && (right) == null){
                cutPoints = new double[1];
                cutPoints[0] = bestCutPoint;
            } 
            else if (right == null) {
                cutPoints = new double[left.length + 1];
                System.arraycopy(left, 0, cutPoints, 0, left.length);
                cutPoints[left.length] = bestCutPoint;
            } 
            else if (left == null) {
                cutPoints = new double[1 + right.length];
                cutPoints[0] = bestCutPoint;
                System.arraycopy(right, 0, cutPoints, 1, right.length);
            } 
            else {
                cutPoints = new double[left.length + right.length + 1];
                System.arraycopy(left, 0, cutPoints, 0, left.length);
                cutPoints[left.length] = bestCutPoint;
                System.arraycopy(right, 0, cutPoints, left.length + 1, right.length);
            }
          return cutPoints;
        }
        else {
          return null;
        }
    }

/**
   * Test using Kononenko's MDL criterion.
   * 
   * @param priorCounts
   * @param bestCounts
   * @param numInstances
   * @param numCutPoints
   * @return true if the split is acceptable
*/
    private boolean kononenkosMDL(double[] priorCounts, double[][] bestCounts, double numInstances, int numCutPoints){
        double distPrior, instPrior, distAfter = 0, sum, instAfter = 0;
        double before, after;
        int numClassesTotal;

        //Number of classes occuring in the set
        numClassesTotal = 0;
        for (double priorCount : priorCounts) {
          if (priorCount > 0) {
            numClassesTotal++;
          }
        }

        //Encode distribution prior to split
        distPrior = SpecialFunctions.log2Binomial(numInstances + numClassesTotal - 1, numClassesTotal - 1);

        //Encode instances prior to split.
        instPrior = SpecialFunctions.log2Multinomial(numInstances, priorCounts);

        before = instPrior + distPrior;

        //Encode distributions and instances after split.
        for (double[] bestCount : bestCounts) {
          sum = Utils.sum(bestCount);
          distAfter += SpecialFunctions.log2Binomial(sum + numClassesTotal - 1, numClassesTotal - 1);
          instAfter += SpecialFunctions.log2Multinomial(sum, bestCount);
        }

        //Coding cost after split
        after = Utils.log2(numCutPoints) + distAfter + instAfter;

        //Check if split is to be accepted
        return (before > after);
    }
  
    public double kononenkosMDL(Instances instances,int attIndex){
        double distPrior, instPrior, distAfter = 0, sum, instAfter = 0;
        double before, after;
        int numClassesTotal;
        double numInstances = 0;
        //table with frequencies for each class - how many instances occur in a particular class
        double [] priorCounts=Arrays.stream(instances.attributeStats(instances.classIndex()).nominalCounts).asDoubleStream().toArray(); //we convert because we need in log2Multinomial as parameter double array
        //!!!we use opposite indexes (i for attribute values, j for class values) because of easier later summation ...  then we use just reference to the row from 2d array -> for (double[] bestCount : matrixCounts)

        //get nominal labels ... better solution than instances.attributeStats(attIndex).distinctCount because e.g., for num-of-N we can have/generate less values than are in attribute specification (nominla labels)
        String nominalLabels=instances.attribute(attIndex).toString();
        nominalLabels=nominalLabels.substring(nominalLabels.lastIndexOf("{"),(nominalLabels.lastIndexOf("}")+1));
        int numOfLabels=nominalLabels.split(",").length;

        double [][]matrixCounts=new double[numOfLabels][instances.numClasses()];
        //instances

        /*If the attribute is numeric then the value you get from value() is the actual value. If the attribute is nominal or string, then you get the index of the
        actual nominal value returned as a double. The Attribute object of the attribute in question can give you the value (as a String) corresponding to the index. */
        for (int i = 0; i < instances.numInstances(); i++){
            matrixCounts[(int)instances.instance(i).value(attIndex)][(int)instances.instance(i).classValue()]++;
            numInstances += instances.instance(i).weight();          
        }

        // Number of classes occuring in the set
        numClassesTotal = instances.numClasses();

        //Encode distribution prior to split
        distPrior = SpecialFunctions.log2Binomial(numInstances + numClassesTotal - 1, numClassesTotal - 1);

        //Encode instances prior to split.
        instPrior = SpecialFunctions.log2Multinomial(numInstances, priorCounts);

        before = instPrior + distPrior;

        for (double[] bestCount : matrixCounts) {
            sum = Utils.sum(bestCount);     //Utils.sum sum of all numbers in an array
            distAfter += SpecialFunctions.log2Binomial(sum + numClassesTotal - 1, numClassesTotal - 1);
            instAfter += SpecialFunctions.log2Multinomial(sum, bestCount);
        }

        after = distAfter + instAfter;

        return (before-after)/numInstances;
    }
 
/**
   * Test using Fayyad and Irani's MDL criterion.
   * 
   * @param priorCounts
   * @param bestCounts
   * @param numInstances
   * @param numCutPoints
   * @return true if the splits is acceptable
   */
    private boolean fayyadAndIranisMDL(double[] priorCounts,double[][] bestCounts, double numInstances, int numCutPoints) {
        double priorEntropy, entropy, gain;
        double entropyLeft, entropyRight, delta;
        int numClassesTotal, numClassesRight, numClassesLeft;

        // Compute entropy before split.
        priorEntropy = ContingencyTables.entropy(priorCounts);

        // Compute entropy after split.
        entropy = ContingencyTables.entropyConditionedOnRows(bestCounts);

        // Compute information gain.
        gain = priorEntropy - entropy;

        // Number of classes occuring in the set
        numClassesTotal = 0;
        for (double priorCount : priorCounts) {
          if (priorCount > 0) {
            numClassesTotal++;
          }
        }

        // Number of classes occuring in the left subset
        numClassesLeft = 0;
        for (int i = 0; i < bestCounts[0].length; i++) {
          if (bestCounts[0][i] > 0) {
            numClassesLeft++;
          }
        }

        // Number of classes occuring in the right subset
        numClassesRight = 0;
        for (int i = 0; i < bestCounts[1].length; i++) {
          if (bestCounts[1][i] > 0) {
            numClassesRight++;
          }
        }

        // Entropy of the left and the right subsets
        entropyLeft = ContingencyTables.entropy(bestCounts[0]);
        entropyRight = ContingencyTables.entropy(bestCounts[1]);

        // Compute terms for MDL formula
        delta = Utils.log2(Math.pow(3, numClassesTotal) - 2) - ((numClassesTotal * priorEntropy) - (numClassesRight * entropyRight) - (numClassesLeft * entropyLeft));

        // Check if split is to be accepted
        return (gain > (Utils.log2(numCutPoints) + delta) / numInstances);
    }

    public double impuritySplit(Instances data, int idx){
        double bestEstimation;
        ArrayList<SortRec> sortedAttr = new ArrayList<SortRec>();
        int noAttrVal[]=new int[3];
        int noClasses=data.numClasses();
        System.out.println(noClasses);
        int noClassAttrVal[][]=new int[noClasses+1][2+1];
        int trainSize=data.numInstances();
        int numOfMissing=data.attributeStats(idx).missingCount;
        sortedAttr=initArrSortRec(trainSize-numOfMissing);

        int j ;
        int OKvalues = 0 ;
        double attrValue ;
        for (j=0; j<trainSize;j++){   
          attrValue=data.instance(j).value(idx);
            if (Double.isNaN(attrValue)) //controlling missing value
                continue;
            sortedAttr.get(OKvalues).setKey(attrValue); 
            sortedAttr.get(OKvalues).setValue(j);
            noClassAttrVal[(int) data.instance(j).classValue()][2]++;  
          OKvalues++ ;
       }
        
        if (OKvalues <= 1){     //all the cases have missing value of the attribute or only one OK
           bestEstimation = - Double.MAX_VALUE ;
           return - Double.MAX_VALUE;   //smaller than any value, so all examples will go into one branch
        }

         double[][] counts;
         double[] priorCounts;
         double priorEntropy;
         double numInstances = 0;

         //Compute class counts.
         counts = new double[2][data.numClasses()];
         for (int i = 0; i < trainSize; i++){
             numInstances += data.instance(i).weight();   
         //e.g. an instance with a weight of 2 corresponds to two identical instances with a weight of 1. (Instance weights are more flexible though because they don’t need to be integers.
         //Note that any instance without a weight value specified is assumed to have a weight of 1 for backwards compatibility.
             counts[1][(int) data.instance(i).classValue()] += data.instance(i).weight();
         }

         //Save prior counts
         priorCounts = new double[data.numClasses()];
         System.arraycopy(counts[1], 0, priorCounts, 0, data.numClasses());     //copy elements from counts[l] to priorCounts

         //Entropy of the full set
         priorEntropy = ContingencyTables.entropy(priorCounts);
         double priorImpurity=priorEntropy;

        Collections.sort(sortedAttr); 
        bestEstimation = - Double.MAX_VALUE  ;
        double est = 0, splitValue = - Double.MAX_VALUE  ; //smaller than any value, so all examples will go into one branch
        //initially we move some left instance from right to left
        int minNodeWeightEst = 2; //minNodeWeightEst (minimal split to consider in attribute evaluation) should be non-negative ... minimal split to be evaluated
        for (j=0 ; j < minNodeWeightEst ; j++) {
            noClassAttrVal[(int) data.instance(j).classValue()][1]++;    //increase on the left
            noClassAttrVal[(int) data.instance(j).classValue()][2]--;    //decrease on right
        }

        int upperLimit = OKvalues - minNodeWeightEst;
        for ( ; j < upperLimit ; j++){
             //only estimate for unique values 
             if(sortedAttr.get(j).getKey()!=sortedAttr.get(j-1).getKey()){
               //compute heuristic measure
               noAttrVal[1] = j ;
               noAttrVal[2] = OKvalues - j ;

             //Compute class counts.
             counts = new double[2][data.numClasses()];
             for (int i = 0; i < trainSize; i++){
                 numInstances += data.instance(i).weight();   
             //e.g. an instance with a weight of 2 corresponds to two identical instances with a weight of 1. (Instance weights are more flexible though because they don’t need to be integers.
             //Note that any instance without a weight value specified is assumed to have a weight of 1 for backwards compatibility.
                 counts[1][(int) data.instance(i).classValue()] += data.instance(i).weight();
             }

             //Save prior counts
             priorCounts = new double[data.numClasses()];
             System.arraycopy(counts[1], 0, priorCounts, 0, data.numClasses());     //copy elements from counts[l] to priorCounts

             //Entropy of the full set
             priorEntropy = ContingencyTables.entropy(priorCounts);
             priorImpurity=priorEntropy;
             est=priorEntropy;

               if (est > bestEstimation){
                       bestEstimation = est ;
                       splitValue=(sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0;
                       System.out.println("Split value internal "+ splitValue);
               }
            }

                 noClassAttrVal[(int) data.instance(j).classValue()][1]++;   //increase on the left
                 noClassAttrVal[(int) data.instance(j).classValue()][2]--;   //decrease on right
        }
        return splitValue ;
    }
  
    public static ArrayList<SortRec> initArrSortRec(int size){
        ArrayList<SortRec> tmp=new ArrayList<SortRec>(size);
        for(int i=0;i<size;i++){
            tmp.add(i, new SortRec());
        }
        return tmp;
    } 
}
