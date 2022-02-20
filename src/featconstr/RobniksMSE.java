package featconstr;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import weka.core.Instances;

/**
 *
 * @author bostjan
 */
public class RobniksMSE {
    public RobniksMSE(){
    }
    
    ArrayList<Double> NumEstimation = new ArrayList<>();
    ArrayList<Double> DiscEstimation = new ArrayList<>();
    ArrayList<Double> splitPoint = new ArrayList<>();
    ArrayList<Integer> discNoValues = new ArrayList<>();
    List<List<Integer>> DiscValues = new ArrayList<>();
    List<List<Double>> NumValues = new ArrayList<>();
    ArrayList<Double> weight = new ArrayList<>();
    public int noDiscrete;
    public int noNumeric;
  
    double epsilon=1E-7;
    public final void mseDev(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, Instances data){
        int trainSize=data.numInstances();
        weight=calculateWeights(data);
        int i;
        int j;
        ArrayList<Double> valueClass = new ArrayList<>();
        ArrayList<Double> valueWeight = new ArrayList<>();
        ArrayList<Double> squaredValues = new ArrayList<>();
        ArrayList<SortRec> sortedMean = new ArrayList<>();
        int idx;
        int OKvalues;
        double totalWeight;
        double value;
        double bestEstimate;
        double estimate;
        double pLeft;
        double variance;
        double LeftValues;
        double LeftSquares;
        double LeftWeight;
        double RightValues;
        double RightSquares;
        double RightWeight;

        for (i = discAttrFrom ; i < discAttrTo ; i++){
            valueClass.clear();
            valueWeight.clear();
            squaredValues.clear();
            for (j = 0 ; j < trainSize ; j++){
                idx = DiscValues.get(j).get(i);
                value = NumValues.get(j).get(0);
                valueClass.set(idx, valueClass.get(idx)+ weight.get(j) * value);
                valueWeight.set(idx, valueWeight.get(idx) + weight.get(j));
                squaredValues.set(idx, squaredValues.get(idx)+weight.get(j)*value*value);
            }
            
            sortedMean=new ArrayList<>(discNoValues.get(i));
            RightWeight = RightSquares = RightValues = 0.0;
            OKvalues = 0;
          
            for (j = 1 ; j <= discNoValues.get(i); j++){
                if (valueWeight.get(j) > epsilon){
                    sortedMean.get(OKvalues).setKey(valueClass.get(j) / valueWeight.get(j));
                    sortedMean.get(OKvalues).setValue(j);                                
                    OKvalues++;
                    RightWeight +=valueWeight.get(j);
                    RightSquares += squaredValues.get(j);
                    RightValues +=valueClass.get(j);
                }
            }
            totalWeight = RightWeight;

            Collections.sort(sortedMean);
            bestEstimate = Double.MAX_VALUE;
            LeftWeight = LeftSquares = LeftValues = 0.0;
            int upper = OKvalues - 1;
            for (j = 0 ; j < upper ; j++){
                idx = sortedMean.get(j).getValue();
                LeftSquares += squaredValues.get(idx);
                LeftValues +=valueClass.get(idx);
                LeftWeight +=valueWeight.get(idx);
                RightSquares -=squaredValues.get(idx);
                RightValues -=valueClass.get(idx);
                RightWeight -=valueWeight.get(idx);
                pLeft = LeftWeight / totalWeight;
                variance = LeftSquares / LeftWeight - Math.pow((LeftValues / LeftWeight),2);
                if (LeftWeight > epsilon && variance > 0.0)
                    estimate = pLeft * variance;
                else
                    estimate = 0.0;
                  
                variance = RightSquares / RightWeight - Math.pow((RightValues / RightWeight),2);
                if (LeftWeight > epsilon && variance > 0.0)
                     estimate += (1.0 - pLeft) * variance;

                if (estimate < bestEstimate)
                     bestEstimate = estimate;                      
            }
            DiscEstimation.set(i, - bestEstimate);
        }

	//continuous values
        double dVal;
        ArrayList<SortRec> sortedAttr = new ArrayList<>();
        for (i = contAttrFrom ; i < contAttrTo ; i++){
            RightWeight = RightSquares = RightValues = 0.0;
            OKvalues = 0;
            for (j = 0 ; j < trainSize ; j++){
                if (NumValues.get(j).get(i)==null)
                    continue;
                sortedAttr.get(OKvalues).setKey(NumValues.get(j).get(i));
                sortedAttr.get(OKvalues).setValue(j);
                RightWeight += weight.get(j);
                dVal = weight.get(j) * NumValues.get(j).get(0);
                RightValues += dVal;
                dVal *= NumValues.get(j).get(0);
                RightSquares += dVal;
                OKvalues++;
            }
            totalWeight = RightWeight;
            Collections.sort(sortedAttr);
            bestEstimate = Double.MAX_VALUE;
            LeftWeight = LeftSquares = LeftValues = 0.0;
              j = 0;
            while (j < OKvalues){
                 //collect cases with the same value of the attribute - we cannot split between them
                do{
                    idx = sortedAttr.get(j).getValue();
                    dVal = weight.get(idx) * NumValues.get(idx).get(0);
                    LeftValues += dVal;
                    RightValues -= dVal;
                    dVal *= NumValues.get(idx).get(0);
                    LeftSquares += dVal;
                    RightSquares -= dVal;
                    LeftWeight += weight.get(idx);      
                    RightWeight -= weight.get(idx);    
                    j++;
                }while (j < OKvalues && sortedAttr.get(j).getKey() == sortedAttr.get(j-1).getKey());
                if (j == OKvalues)
                    break;
                pLeft = LeftWeight / totalWeight;
                variance = LeftSquares / LeftWeight - Math.pow(LeftValues / LeftWeight,2);
                if (LeftWeight > epsilon && variance > 0.0)
                    estimate = pLeft * variance;                 
                else                 
                    estimate = 0.0;
                 
                variance = RightSquares / RightWeight - Math.pow(RightValues / RightWeight,2);
                if (RightWeight > epsilon && variance > 0.0)
                    estimate += (1.0 - pLeft) * variance;
                 
                if (estimate < bestEstimate){
                    bestEstimate = estimate;
                    splitPoint.set(i, (sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0);
                }
            }
              NumEstimation.set(i, - bestEstimate);
        }
    }
    
    public ArrayList<Double> mseNumericAttr(int contAttrFrom, int contAttrTo, Instances data){
        int i, j, idx, OKvalues;
        double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight;
        int trainSize=data.numInstances();
        
        weight=calculateWeights(data);   
        NumEstimation=new ArrayList<>(Collections.nCopies(contAttrTo-contAttrFrom, 0.0));
        splitPoint=new ArrayList<>(Collections.nCopies(data.numAttributes()-1, 0.0));
	//continuous values
        double dVal;
        ArrayList<SortRec> sortedAttr = new ArrayList<>(trainSize);
        sortedAttr=initArrSortRec(trainSize);

        for (i = contAttrFrom ; i < contAttrTo ; i++){
            RightWeight = RightSquares = RightValues = 0.0;
            OKvalues = 0;
            for (j = 0 ; j < trainSize ; j++){
            if (Double.isNaN(data.instance(j).value(i)))
              continue;
            sortedAttr.get(OKvalues).setKey(data.instance(j).value(i));
            sortedAttr.get(OKvalues).setValue(j);
            RightWeight += data.instance(j).weight();
            dVal = data.instance(j).weight() * data.instance(j).classValue();
            RightValues += dVal;
            dVal *= data.instance(j).classValue();
            RightSquares += dVal;
            OKvalues++;
            }
            totalWeight = RightWeight;
            Collections.sort(sortedAttr);
            bestEstimate = Double.MAX_VALUE;
            LeftWeight = LeftSquares = LeftValues = 0.0;
            j = 0;
            while (j < OKvalues){
                //collect cases with the same value of the attribute - we cannot split between them
                do{
                    idx = sortedAttr.get(j).getValue();
                    dVal = data.instance(idx).weight() * data.instance(idx).classValue();
                    LeftValues += dVal;
                    RightValues -= dVal;
                    dVal *= data.instance(idx).classValue();
                    LeftSquares += dVal;
                    RightSquares -= dVal;
                    LeftWeight += data.instance(idx).weight();
                    RightWeight -= data.instance(idx).weight();
                    j++;
                } while (j < OKvalues && sortedAttr.get(j).getKey() == sortedAttr.get(j-1).getKey());
                if (j == OKvalues)
                    break;
                pLeft = LeftWeight / totalWeight;
                variance = LeftSquares / LeftWeight - Math.pow(LeftValues / LeftWeight,2);
                if (LeftWeight > epsilon && variance > 0.0)
                    estimate = pLeft * variance;                
                else                
                    estimate = 0.0;
                
                variance = RightSquares / RightWeight - Math.pow(RightValues / RightWeight,2);
                if (RightWeight > epsilon && variance > 0.0)
                    estimate += (1.0 - pLeft) * variance;                
                if (estimate < bestEstimate){
                    bestEstimate = estimate;
                    splitPoint.set(i, (sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0);
                }
            }
            NumEstimation.set(i, - bestEstimate);
        }
        return NumEstimation;
    }
    
    public ArrayList<Double> mseDiscreteAttr(int discAttrFrom, int discAttrTo, Instances data){
        int i, j, idx, OKvalues;
	double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight, value;
        int trainSize=data.numInstances();

        weight=calculateWeights(data);
        DiscEstimation=new ArrayList<>(Collections.nCopies(data.numAttributes()-1, 0.0)); 
        
        ArrayList<Double> valueClass = new ArrayList<>();
	ArrayList<Double> valueWeight = new ArrayList<>();
	ArrayList<Double> squaredValues = new ArrayList<>();
	ArrayList<SortRec> sortedMean = new ArrayList<>();

        discNoValues=new ArrayList<>(Collections.nCopies(data.numAttributes()-1, 0));
	//estimationReg of discrete attributtes
        for (i = discAttrFrom ; i < discAttrTo ; i++){
                  discNoValues.set(i, data.numDistinctValues(i));
		  valueClass=new ArrayList<>(Collections.nCopies(data.numAttributes()-1, 0.0));
		  valueWeight=new ArrayList<>(Collections.nCopies(data.numAttributes()-1, 0.0));
                  squaredValues=new ArrayList<>(Collections.nCopies(data.numAttributes()-1, 0.0));
                  
		  for (j = 0 ; j < trainSize ; j++){
			 idx = (int) data.instance(j).value(i); 
			 value = data.instance(j).classValue();
			 valueClass.set(idx, valueClass.get(idx)+ weight.get(j) * value);
			 valueWeight.set(idx, valueWeight.get(idx) + weight.get(j));
			 squaredValues.set(idx, squaredValues.get(idx)+weight.get(j)*value*value);
		  }
		  //define size of sortedMean
                  sortedMean=new ArrayList<>(discNoValues.get(i));
                  sortedMean=initArrSortRec(discNoValues.get(i));
		  RightWeight = RightSquares = RightValues = 0.0;
		  OKvalues = 0;
                  for (j = 0 ; j < discNoValues.get(i); j++){
                        if (valueWeight.get(j) > epsilon){
                            sortedMean.get(OKvalues).setKey(valueClass.get(j) / valueWeight.get(j));
                            sortedMean.get(OKvalues).setValue(j);
                            OKvalues++;
                            RightWeight +=valueWeight.get(j);
                            RightSquares += squaredValues.get(j);
                            RightValues +=valueClass.get(j);
			 }
		  }
		  totalWeight = RightWeight;
		  
                  Collections.sort(sortedMean);
		  bestEstimate = Double.MAX_VALUE;
		  LeftWeight = LeftSquares = LeftValues = 0.0;
		  int upper = OKvalues - 1;
		  for (j = 0 ; j < upper ; j++){
			  idx = sortedMean.get(j).getValue();
			  LeftSquares += squaredValues.get(idx);
			  LeftValues +=valueClass.get(idx);
			  LeftWeight +=valueWeight.get(idx);
			  RightSquares -=squaredValues.get(idx);
			  RightValues -=valueClass.get(idx);
			  RightWeight -=valueWeight.get(idx);
			  pLeft = LeftWeight / totalWeight;
			  variance = LeftSquares / LeftWeight - Math.pow((LeftValues / LeftWeight),2);
			  if (LeftWeight > epsilon && variance > 0.0)
				estimate = pLeft * variance;			  
			  else			  
				estimate = 0.0;			  

			  variance = RightSquares / RightWeight - Math.pow((RightValues / RightWeight),2);
			  if (LeftWeight > epsilon && variance > 0.0)
				 estimate += (1.0 - pLeft) * variance;			  

			  if (estimate < bestEstimate)
				 bestEstimate = estimate;			  
		  }
		  DiscEstimation.set(i, - bestEstimate);
	   }
           return DiscEstimation;
    }
    
    public double mseDiscreteAttr(int attrIdx, Instances data){
        int j, idx, OKvalues;
	double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight, value;
        int trainSize=data.numInstances();
        weight=calculateWeights(data);
        
        ArrayList<Double> valueClass = new ArrayList<>();
	ArrayList<Double> valueWeight = new ArrayList<>();
	ArrayList<Double> squaredValues = new ArrayList<>();
	ArrayList<SortRec> sortedMean = new ArrayList<>();

        discNoValues=new ArrayList<>(Collections.nCopies(data.numAttributes()-1, 0)); //withouth class attribute
	//estimationReg of discrete attributte
        discNoValues.set(attrIdx, data.numDistinctValues(attrIdx));
        valueClass=new ArrayList<>(Collections.nCopies(data.numDistinctValues(attrIdx), 0.0));
        valueWeight=new ArrayList<>(Collections.nCopies(data.numDistinctValues(attrIdx), 0.0));
        squaredValues=new ArrayList<>(Collections.nCopies(data.numDistinctValues(attrIdx), 0.0));
                  
        for (j = 0 ; j < trainSize ; j++){
            idx = (int) data.instance(j).value(attrIdx);
            value = data.instance(j).classValue();
            valueClass.set(idx, valueClass.get(idx)+ weight.get(j) * value);
            valueWeight.set(idx, valueWeight.get(idx) + weight.get(j));
            squaredValues.set(idx, squaredValues.get(idx)+weight.get(j)*value*value);
        }

        sortedMean=new ArrayList<>(discNoValues.get(attrIdx));
        sortedMean=initArrSortRec(discNoValues.get(attrIdx));
        RightWeight = RightSquares = RightValues = 0.0;
        OKvalues = 0;

        for (j = 0 ; j < discNoValues.get(attrIdx); j++){
            if (valueWeight.get(j) > epsilon){
                sortedMean.get(OKvalues).setKey(valueClass.get(j) / valueWeight.get(j));
                sortedMean.get(OKvalues).setValue(j);                                 	                    
                OKvalues++;
                RightWeight +=valueWeight.get(j);
                RightSquares += squaredValues.get(j);
                RightValues +=valueClass.get(j);
             }
        }
        totalWeight = RightWeight;	  
        Collections.sort(sortedMean);
        bestEstimate = Double.MAX_VALUE;
        LeftWeight = LeftSquares = LeftValues = 0.0;
        int upper = OKvalues - 1;
        for (j = 0 ; j < upper ; j++){
            idx = sortedMean.get(j).getValue();
            LeftSquares += squaredValues.get(idx);
            LeftValues +=valueClass.get(idx);
            LeftWeight +=valueWeight.get(idx);
            RightSquares -=squaredValues.get(idx);
            RightValues -=valueClass.get(idx);
            RightWeight -=valueWeight.get(idx);
            pLeft = LeftWeight / totalWeight;
            variance = LeftSquares / LeftWeight - Math.pow((LeftValues / LeftWeight),2);
            if (LeftWeight > epsilon && variance > 0.0)
                estimate = pLeft * variance;			  
            else			  
                estimate = 0.0;

            variance = RightSquares / RightWeight - Math.pow((RightValues / RightWeight),2);
            if (LeftWeight > epsilon && variance > 0.0)
                 estimate += (1.0 - pLeft) * variance;			  

            if (estimate < bestEstimate)
                 bestEstimate = estimate;			  
        }
	   
        return - bestEstimate;
    }
    
    public double mseNumericAttr(int attrIdx, Instances data){
	int j, idx, OKvalues;
	double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight;
        int trainSize=data.numInstances();
        double splitPoint=Double.MAX_VALUE;
	//continuous values
        double dVal;
        int numOfMissing=data.attributeStats(attrIdx).missingCount;
        ArrayList<SortRec> sortedAttr = new ArrayList<SortRec>(trainSize-numOfMissing);
        sortedAttr=initArrSortRec(trainSize-numOfMissing);

        RightWeight = RightSquares = RightValues = 0.0;
        OKvalues = 0;
        for (j = 0 ; j < trainSize ; j++){
            if(Double.isNaN(data.instance(j).value(attrIdx)))
                continue;
            sortedAttr.get(OKvalues).setKey(data.instance(j).value(attrIdx));
            sortedAttr.get(OKvalues).setValue(j);
            RightWeight += data.instance(j).weight();
            dVal = data.instance(j).weight() * data.instance(j).classValue();
            RightValues += dVal;
            dVal *= data.instance(j).classValue();
            RightSquares += dVal;
            OKvalues++;
        }
        totalWeight = RightWeight;
        Collections.sort(sortedAttr);
        bestEstimate = Double.MAX_VALUE;
        LeftWeight = LeftSquares = LeftValues = 0.0;
        j = 0;
        while (j < OKvalues){
             //collect cases with the same value of the attribute - we cannot split between them
            do{
                idx = sortedAttr.get(j).getValue();
                dVal = data.instance(idx).weight() * data.instance(idx).classValue();
                LeftValues += dVal;
                RightValues -= dVal;
                dVal *= data.instance(idx).classValue();
                LeftSquares += dVal;
                RightSquares -= dVal;
                LeftWeight += data.instance(idx).weight();
                RightWeight -= data.instance(idx).weight();
                j++;
            }while (j < OKvalues && sortedAttr.get(j).getKey() == sortedAttr.get(j-1).getKey());
            if (j == OKvalues)
                break;
            pLeft = LeftWeight / totalWeight;
            variance = LeftSquares / LeftWeight - Math.pow(LeftValues / LeftWeight,2);
            if (LeftWeight > epsilon && variance > 0.0)
                estimate = pLeft * variance;             
            else             
                estimate = 0.0;
             
            variance = RightSquares / RightWeight - Math.pow(RightValues / RightWeight,2);
            if (RightWeight > epsilon && variance > 0.0)
                estimate += (1.0 - pLeft) * variance;
             
            if (estimate < bestEstimate){
                bestEstimate = estimate;
                splitPoint=(sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0;
            }
        }
        return - bestEstimate;
    }
    
    public static ArrayList<Double> calculateWeights(Instances data){
        ArrayList<Double> weights = new ArrayList<Double>();
        for(int i=0;i<data.numInstances();i++)
            weights.add(data.instance(i).weight());
        
        return weights;
    }
    
    public static ArrayList<SortRec> initArrSortRec(int size){
        ArrayList<SortRec> tmp=new ArrayList<>(size);
        for(int i=0;i<size;i++)
            tmp.add(i, new SortRec());
        
        return tmp;
    }
}