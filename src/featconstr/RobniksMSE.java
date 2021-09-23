/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
    /*
    mmatrix<int> DiscData, DiscPredictData
        contain values of discrete attributes and class for training and prediction (optional). In classification column 0 always stores class values.
    mmatrix<double> ContData, ContPredictData
        contain values of numeric attributeand prediction values for training and prediction (optional).  In regression column 0 always stores target values.
    
    */
    public RobniksMSE(){

    }
    
    ArrayList<Double> NumEstimation = new ArrayList<Double>();
    ArrayList<Double> DiscEstimation = new ArrayList<Double>();
    ArrayList<Double> splitPoint = new ArrayList<Double>();
    ArrayList<Integer> discNoValues = new ArrayList<Integer>();       //marray<int> discNoValues;
    List<List<Integer>> DiscValues = new ArrayList<List<Integer>>();   //mmatrix<int> DiscValues; C++ ... containing discrete attributes and class values
    List<List<Double>> NumValues = new ArrayList<List<Double>>();    //mmatrix<double> NumValues ; ... containing numeric attribute and prediction values
    ArrayList<Double> weight = new ArrayList<Double>();     //marray<double> weight; C++
    public int noDiscrete;
    public int noNumeric;
  
    double epsilon=1E-7;
    public final void mseDev(int contAttrFrom, int contAttrTo, int discAttrFrom, int discAttrTo, Instances data){

        int trainSize=data.numInstances();
        weight=calculateWeights(data);
        
        /*dynamic2D.add(new ArrayList<Integer>());
        dynamic2D.add(new ArrayList<Integer>());
        dynamic2D.add(new ArrayList<Integer>());

        dynamic2D.get(0).add(5);
        dynamic2D.get(0).add(6);
        dynamic2D.get(0).add(7);

        System.out.println(dynamic2D.get(0).get(0)); // 5
        System.out.println(dynamic2D.get(0).get(1)); // 6
        System.out.println(dynamic2D.get(0).get(2)); // 7*/
        
        
        
	   // initialization of estimationRegs
	   //NumEstimation.init(contAttrFrom,contAttrTo, 0.0);
	   //DiscEstimation.init(discAttrFrom,discAttrTo, 0.0);
	   //splitPoint.init(contAttrFrom,contAttrTo, Double.MAX_VALUE);

	   int i;
	   int j;
           ArrayList<Double> valueClass = new ArrayList<Double>();      //marray<Double> valueClass = new marray<Double>(); C++
	   ArrayList<Double> valueWeight = new ArrayList<Double>();     //marray<Double> valueWeight = new marray<Double>();
	   ArrayList<Double> squaredValues = new ArrayList<Double>();   //marray<Double> squaredValues = new marray<Double>();
	   ArrayList<SortRec> sortedMean = new ArrayList<SortRec>();          //marray<sortRec> sortedMean = new marray<sortRec>();
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
	   // estimationReg of discrete attributtes

	   for (i = discAttrFrom ; i < discAttrTo ; i++){
		  valueClass.clear();           //valueClass.create(discNoValues[i] + 1, 0.0); C++
		  valueWeight.clear();          //valueWeight.create(discNoValues[i] + 1, 0.0); C++
                  squaredValues.clear();        //squaredValues.create(discNoValues[i] + 1, 0.0); C++
		  for (j = 0 ; j < trainSize ; j++){
			 idx = DiscValues.get(j).get(i);    //DiscValues(j,i);
			 value = NumValues.get(j).get(0);   //NumValues(j,0);
			 valueClass.set(idx, valueClass.get(idx)+ weight.get(j) * value);   //valueClass[idx] += weight[j] * value;
			 valueWeight.set(idx, valueWeight.get(idx) + weight.get(j));        //valueWeight[idx] += weight[j];
			 squaredValues.set(idx, squaredValues.get(idx)+weight.get(j)*value*value);      //squaredValues[idx] += weight[j] * sqr(value);
		  }
		  //sortedMean.create(discNoValues[i]);   //define size of sortedMean
                  sortedMean=new ArrayList<SortRec>(discNoValues.get(i));
		  RightWeight = RightSquares = RightValues = 0.0;
		  OKvalues = 0;
		  //for (j = 1 ; j <= discNoValues[i]; j++){
                  for (j = 1 ; j <= discNoValues.get(i); j++){
                        if (valueWeight.get(j) > epsilon){
                            sortedMean.get(OKvalues).setKey(valueClass.get(j) / valueWeight.get(j));        //sortedMean[OKvalues].key = valueClass[j] / valueWeight[j];
                            sortedMean.get(OKvalues).setValue(j);                                           //sortedMean[OKvalues].value = j;
                /*???*/     //sortedMean.set(OKvalues, new SortRec(valueClass.get(j) / valueWeight.get(j)));  
                /*???*/     //sortedMean.set(OKvalues, new SortRec(j));                                       
	                    
                            OKvalues++;

                            RightWeight +=valueWeight.get(j);       //RightWeight += valueWeight[j];
                            RightSquares += squaredValues.get(j);   //RightSquares += squaredValues[j];
                            RightValues +=valueClass.get(j);        //RightValues += valueClass[j];
			 }
		  }
		  totalWeight = RightWeight;
	/*???*/	  //sortedMean.setFilled(OKvalues);       // sets the point to which array is filled
		  
                  Collections.sort(sortedMean); //sortedMean.qsortAsc();
		  bestEstimate = Double.MAX_VALUE;
		  LeftWeight = LeftSquares = LeftValues = 0.0;
		  int upper = OKvalues - 1;
		  for (j = 0 ; j < upper ; j++){
			  idx = sortedMean.get(j).getValue();                //idx = sortedMean[j].value;
			  LeftSquares += squaredValues.get(idx);    //LeftSquares += squaredValues[idx];
			  LeftValues +=valueClass.get(idx);         //LeftValues += valueClass[idx];
			  LeftWeight +=valueWeight.get(idx);        //LeftWeight += valueWeight[idx];
			  RightSquares -=squaredValues.get(idx);    //RightSquares -= squaredValues[idx];
			  RightValues -=valueClass.get(idx);        //RightValues -= valueClass[idx];
			  RightWeight -=valueWeight.get(idx);       //RightWeight -= valueWeight[idx];
			  pLeft = LeftWeight / totalWeight;
			  variance = LeftSquares / LeftWeight - Math.pow((LeftValues / LeftWeight),2); //variance = LeftSquares / LeftWeight - sqr(LeftValues / LeftWeight);
			  if (LeftWeight > epsilon && variance > 0.0){
				estimate = pLeft * variance;
			  }
			  else
			  {
				estimate = 0.0;
			  }

			  variance = RightSquares / RightWeight - Math.pow((RightValues / RightWeight),2);   //variance = RightSquares / RightWeight - sqr(RightValues / RightWeight);
			  if (LeftWeight > epsilon && variance > 0.0){
				 estimate += ((double)1.0 - pLeft) * variance;
			  }

			  if (estimate < bestEstimate){
				 bestEstimate = estimate;
			  }
		  }
		  DiscEstimation.set(i, - bestEstimate);//DiscEstimation[i] = - bestEstimate;
	   }


	   // continuous values
	   double dVal;
	   ArrayList<SortRec> sortedAttr = new ArrayList<SortRec>();    //marray<sortRec> sortedAttr = new marray<sortRec>(TrainSize);
	   for (i = contAttrFrom ; i < contAttrTo ; i++){
		  RightWeight = RightSquares = RightValues = 0.0;
		  OKvalues = 0;
		  for (j = 0 ; j < trainSize ; j++){
			 if (NumValues.get(j).get(i)==null) //if (isNAcont(NumValues(j,i)))  int isNAcont(x) -> R_IsNA(x) ... kontroling missing value
			   continue;
                        sortedAttr.get(OKvalues).setKey(NumValues.get(j).get(i));           //sortedAttr[OKvalues].key = NumValues(j,i);
                        sortedAttr.get(OKvalues).setValue(j);                               //sortedAttr[OKvalues].value = j;
                /*???*/ //sortedAttr.set(OKvalues, new SortRec(NumValues.get(j).get(i)));        //sortedAttr[OKvalues].key = NumValues(j,i);
                /*???*/ //sortedAttr.set(OKvalues, new SortRec(j));                              //sortedAttr[OKvalues].value = j;
			 RightWeight += weight.get(j);                      //RightWeight += weight[j];
			 dVal = weight.get(j) * NumValues.get(j).get(0);    //dVal = weight[j] * NumValues(j,0); ... class value
			 RightValues += dVal;
                         dVal *= NumValues.get(j).get(0);                   //dVal *= NumValues(j,0);
			 RightSquares += dVal;
			 OKvalues++;
		  }
		  totalWeight = RightWeight;
	/*???*/	  //sortedAttr.setFilled(OKvalues);   //// sets the point to which array is filled
		  Collections.sort(sortedAttr);     //sortedAttr.qsortAsc();
		  bestEstimate = Double.MAX_VALUE;
		  LeftWeight = LeftSquares = LeftValues = 0.0;
		  j = 0;
		  while (j < OKvalues){
			 // collect cases with the same value of the attribute - we cannot split between them
			 do{
			   idx = sortedAttr.get(j).getValue();   //idx = sortedAttr[j].value;
                           dVal = weight.get(idx) * NumValues.get(idx).get(0);      //dVal = weight[idx] * NumValues(idx, 0);
			   LeftValues += dVal;
			   RightValues -= dVal;
			   dVal *= NumValues.get(idx).get(0);   //dVal *= NumValues(idx, 0);
			   LeftSquares += dVal;
			   RightSquares -= dVal;
			   LeftWeight += weight.get(idx);       //LeftWeight += weight[idx];
			   RightWeight -= weight.get(idx);      //RightWeight -= weight[idx];
			   j++;
			 } while (j < OKvalues && sortedAttr.get(j).getKey() == sortedAttr.get(j-1).getKey());//while (j < OKvalues && sortedAttr[j].key == sortedAttr[j - 1].key);
			 if (j == OKvalues)
				break;
			 pLeft = LeftWeight / totalWeight;
			 variance = LeftSquares / LeftWeight - Math.pow(LeftValues / LeftWeight,2);    //variance = LeftSquares / LeftWeight - sqr(LeftValues / LeftWeight);
			 if (LeftWeight > epsilon && variance > 0.0){
				estimate = pLeft * variance;
			 }
			 else
			 {
				estimate = 0.0;
			 }
                         variance = RightSquares / RightWeight - Math.pow(RightValues / RightWeight,2);        //variance = RightSquares / RightWeight - sqr(RightValues / RightWeight);
			 if (RightWeight > epsilon && variance > 0.0){
				estimate += (1.0 - pLeft) * variance;
			 }
			 if (estimate < bestEstimate){
				bestEstimate = estimate;
				splitPoint.set(i, (sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0);//splitPoint[i] = (sortedAttr[j].key + sortedAttr[j - 1].key) / 2.0;
			 }
		  }
		  NumEstimation.set(i, - bestEstimate);//NumEstimation[i] = - bestEstimate;
	   }
	}
    
    public ArrayList<Double> mseNumericAttr(int contAttrFrom, int contAttrTo, Instances data){
	int i, j, idx, OKvalues;
	double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight;
        int trainSize=data.numInstances();
        
        weight=calculateWeights(data);   
        //NumEstimation=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0));  //inicializacija ... napolnimo z ničlami
        NumEstimation=new ArrayList<Double>(Collections.nCopies(contAttrTo-contAttrFrom, 0.0));  //inicializacija ... napolnimo z ničlami
        splitPoint=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0));
	// continuous values
	   double dVal;
	   ArrayList<SortRec> sortedAttr = new ArrayList<SortRec>(trainSize);    //marray<sortRec> sortedAttr = new marray<sortRec>(TrainSize);
           
           sortedAttr=initArrSortRec(trainSize);

	   for (i = contAttrFrom ; i < contAttrTo ; i++){
		  RightWeight = RightSquares = RightValues = 0.0;
		  OKvalues = 0;
		  for (j = 0 ; j < trainSize ; j++){
			 if (Double.isNaN(data.instance(j).value(i))) //if (NumValues.get(j).get(i)==null) //if (isNAcont(NumValues(j,i)))  int isNAcont(x) -> R_IsNA(x) ... kontroling missing value
			   continue;
                        //sortedAttr.get(OKvalues).setKey(NumValues.get(j).get(i));           //sortedAttr[OKvalues].key = NumValues(j,i);
                        sortedAttr.get(OKvalues).setKey(data.instance(j).value(i));           //sortedAttr[OKvalues].key = NumValues(j,i);
                        sortedAttr.get(OKvalues).setValue(j);                               //sortedAttr[OKvalues].value = j;
                /*???*/ //sortedAttr.set(OKvalues, new SortRec(NumValues.get(j).get(i)));        //sortedAttr[OKvalues].key = NumValues(j,i);
                /*???*/ //sortedAttr.set(OKvalues, new SortRec(j));                              //sortedAttr[OKvalues].value = j;
			 //RightWeight += weight.get(j);                      //RightWeight += weight[j];
                         RightWeight += data.instance(j).weight();                      //RightWeight += weight[j];
			 //dVal = weight.get(j) * NumValues.get(j).get(0);    //dVal = weight[j] * NumValues(j,0); ... class value
                         dVal = data.instance(j).weight() * data.instance(j).classValue();//weight.get(j) * NumValues.get(j).get(0);    //dVal = weight[j] * NumValues(j,0); ... class value
			 RightValues += dVal;
                         //dVal *= NumValues.get(j).get(0);                   //dVal *= NumValues(j,0);
                         dVal *= data.instance(j).classValue();                   //dVal *= NumValues(j,0);
			 RightSquares += dVal;
			 OKvalues++;
		  }
		  totalWeight = RightWeight;
	/*???*/	  //sortedAttr.setFilled(OKvalues);   //// sets the point to which array is filled
		  Collections.sort(sortedAttr);     //sortedAttr.qsortAsc();
		  bestEstimate = Double.MAX_VALUE;
		  LeftWeight = LeftSquares = LeftValues = 0.0;
		  j = 0;
		  while (j < OKvalues){
			 // collect cases with the same value of the attribute - we cannot split between them
			 do{
			   idx = sortedAttr.get(j).getValue();   //idx = sortedAttr[j].value;
                           //dVal = weight.get(idx) * NumValues.get(idx).get(0);      //dVal = weight[idx] * NumValues(idx, 0);
                           dVal = data.instance(idx).weight() * data.instance(idx).classValue();      //dVal = weight[idx] * NumValues(idx, 0);
			   LeftValues += dVal;
			   RightValues -= dVal;
			   //dVal *= NumValues.get(idx).get(0);   //dVal *= NumValues(idx, 0);
                           dVal *= data.instance(idx).classValue();   //dVal *= NumValues(idx, 0);
			   LeftSquares += dVal;
			   RightSquares -= dVal;
			   //LeftWeight += weight.get(idx);       //LeftWeight += weight[idx];
                           LeftWeight += data.instance(idx).weight();       //LeftWeight += weight[idx];
                           //RightWeight -= weight.get(idx);      //RightWeight -= weight[idx];
			   RightWeight -= data.instance(idx).weight();      //RightWeight -= weight[idx];
			   j++;
			 } while (j < OKvalues && sortedAttr.get(j).getKey() == sortedAttr.get(j-1).getKey());//while (j < OKvalues && sortedAttr[j].key == sortedAttr[j - 1].key);
			 if (j == OKvalues)
				break;
			 pLeft = LeftWeight / totalWeight;
			 variance = LeftSquares / LeftWeight - Math.pow(LeftValues / LeftWeight,2);    //variance = LeftSquares / LeftWeight - sqr(LeftValues / LeftWeight);
			 if (LeftWeight > epsilon && variance > 0.0){
				estimate = pLeft * variance;
			 }
			 else
			 {
				estimate = 0.0;
			 }
                         variance = RightSquares / RightWeight - Math.pow(RightValues / RightWeight,2);        //variance = RightSquares / RightWeight - sqr(RightValues / RightWeight);
			 if (RightWeight > epsilon && variance > 0.0){
				estimate += (1.0 - pLeft) * variance;
			 }
			 if (estimate < bestEstimate){
				bestEstimate = estimate;
				splitPoint.set(i, (sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0);//splitPoint[i] = (sortedAttr[j].key + sortedAttr[j - 1].key) / 2.0;
			 }
		  }
		  NumEstimation.set(i, - bestEstimate);//NumEstimation[i] = - bestEstimate;
	   }
           //XplainAttrConstr.printArrayList(splitPoint);
           return NumEstimation;
    }
    
    public ArrayList<Double> mseDiscreteAttr(int discAttrFrom, int discAttrTo, Instances data){
        int i, j, idx, OKvalues;
	double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight, value;
        int trainSize=data.numInstances();

        weight=calculateWeights(data);
        DiscEstimation=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0)); 
        
        ArrayList<Double> valueClass = new ArrayList<Double>();      //marray<Double> valueClass = new marray<Double>(); C++
	ArrayList<Double> valueWeight = new ArrayList<Double>();     //marray<Double> valueWeight = new marray<Double>();
	ArrayList<Double> squaredValues = new ArrayList<Double>();   //marray<Double> squaredValues = new marray<Double>();
	ArrayList<SortRec> sortedMean = new ArrayList<SortRec>();          //marray<sortRec> sortedMean = new marray<sortRec>();

        discNoValues=new ArrayList<Integer>(Collections.nCopies(data.numAttributes()-1, 0));
	// estimationReg of discrete attributtes
        for (i = discAttrFrom ; i < discAttrTo ; i++){
                  discNoValues.set(i, data.numDistinctValues(i));
		  valueClass=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0));    //valueClass.create(discNoValues[i] + 1, 0.0); C++
		  valueWeight=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0));   //valueWeight.create(discNoValues[i] + 1, 0.0); C++
                  squaredValues=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0)); //squaredValues.create(discNoValues[i] + 1, 0.0); C++
                  
		  for (j = 0 ; j < trainSize ; j++){
			 idx = (int) data.instance(j).value(i);//DiscValues.get(j).get(i);    //DiscValues(j,i); ... classValue = DiscValues(j, 0) ; ... attrValue = DiscValues(j, discIdx); index of discrete attr 
			 value = data.instance(j).classValue();     //NumValues.get(j).get(0);   //NumValues(j,0);
                         //if(j==100)
                         //System.out.println("index "+idx+" "+valueClass.get(idx));
                         //System.out.println("weight "+weight.get(j) * value);
			 valueClass.set(idx, valueClass.get(idx)+ weight.get(j) * value);   //valueClass[idx] += weight[j] * value;
			 valueWeight.set(idx, valueWeight.get(idx) + weight.get(j));        //valueWeight[idx] += weight[j];
			 squaredValues.set(idx, squaredValues.get(idx)+weight.get(j)*value*value);      //squaredValues[idx] += weight[j] * sqr(value);
		  }
		  //sortedMean.create(discNoValues[i]);   //define size of sortedMean
                  sortedMean=new ArrayList<SortRec>(discNoValues.get(i));
                  sortedMean=initArrSortRec(discNoValues.get(i));
		  RightWeight = RightSquares = RightValues = 0.0;
		  OKvalues = 0;
		  //for (j = 1 ; j <= discNoValues[i]; j++){
                  for (j = 0 ; j < discNoValues.get(i); j++){
                        if (valueWeight.get(j) > epsilon){
                            //System.out.println(valueClass.get(j));
                            //System.out.println(valueWeight.get(j));

                            sortedMean.get(OKvalues).setKey(valueClass.get(j) / valueWeight.get(j));        //sortedMean[OKvalues].key = valueClass[j] / valueWeight[j];
                            sortedMean.get(OKvalues).setValue(j);                                           //sortedMean[OKvalues].value = j;
                /*???*/     //sortedMean.set(OKvalues, new SortRec(valueClass.get(j) / valueWeight.get(j)));  
                /*???*/     //sortedMean.set(OKvalues, new SortRec(j));                                       
	                    
                            OKvalues++;

                            RightWeight +=valueWeight.get(j);       //RightWeight += valueWeight[j];
                            RightSquares += squaredValues.get(j);   //RightSquares += squaredValues[j];
                            RightValues +=valueClass.get(j);        //RightValues += valueClass[j];
			 }
		  }
		  totalWeight = RightWeight;
	/*???*/	  //sortedMean.setFilled(OKvalues);       // sets the point to which array is filled
		  
                  Collections.sort(sortedMean); //sortedMean.qsortAsc();
		  bestEstimate = Double.MAX_VALUE;
		  LeftWeight = LeftSquares = LeftValues = 0.0;
		  int upper = OKvalues - 1;
		  for (j = 0 ; j < upper ; j++){
			  idx = sortedMean.get(j).getValue();                //idx = sortedMean[j].value;
			  LeftSquares += squaredValues.get(idx);    //LeftSquares += squaredValues[idx];
			  LeftValues +=valueClass.get(idx);         //LeftValues += valueClass[idx];
			  LeftWeight +=valueWeight.get(idx);        //LeftWeight += valueWeight[idx];
			  RightSquares -=squaredValues.get(idx);    //RightSquares -= squaredValues[idx];
			  RightValues -=valueClass.get(idx);        //RightValues -= valueClass[idx];
			  RightWeight -=valueWeight.get(idx);       //RightWeight -= valueWeight[idx];
			  pLeft = LeftWeight / totalWeight;
			  variance = LeftSquares / LeftWeight - Math.pow((LeftValues / LeftWeight),2); //variance = LeftSquares / LeftWeight - sqr(LeftValues / LeftWeight);
			  if (LeftWeight > epsilon && variance > 0.0){
				estimate = pLeft * variance;
			  }
			  else
			  {
				estimate = 0.0;
			  }

			  variance = RightSquares / RightWeight - Math.pow((RightValues / RightWeight),2);   //variance = RightSquares / RightWeight - sqr(RightValues / RightWeight);
			  if (LeftWeight > epsilon && variance > 0.0){
				 estimate += ((double)1.0 - pLeft) * variance;
			  }

			  if (estimate < bestEstimate){
				 bestEstimate = estimate;
			  }
		  }
		  DiscEstimation.set(i, - bestEstimate);//DiscEstimation[i] = - bestEstimate;
	   }
           return DiscEstimation;
    }
    
    public double mseDiscreteAttr(int attrIdx, Instances data){
        int j, idx, OKvalues;
	double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight, value;
        int trainSize=data.numInstances();

        weight=calculateWeights(data);
        //DiscEstimation=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0)); 
        
        ArrayList<Double> valueClass = new ArrayList<Double>();      //marray<Double> valueClass = new marray<Double>(); C++
	ArrayList<Double> valueWeight = new ArrayList<Double>();     //marray<Double> valueWeight = new marray<Double>();
	ArrayList<Double> squaredValues = new ArrayList<Double>();   //marray<Double> squaredValues = new marray<Double>();
	ArrayList<SortRec> sortedMean = new ArrayList<SortRec>();          //marray<sortRec> sortedMean = new marray<sortRec>();

        //discNoValues=new ArrayList<Integer>(Collections.nCopies(data.numDistinctValues(attrIdx), 0));
        discNoValues=new ArrayList<Integer>(Collections.nCopies(data.numAttributes()-1, 0)); //withouth class attribute
	// estimationReg of discrete attributtes

                  discNoValues.set(attrIdx, data.numDistinctValues(attrIdx));
		  valueClass=new ArrayList<Double>(Collections.nCopies(data.numDistinctValues(attrIdx), 0.0));    //valueClass.create(discNoValues[i] + 1, 0.0); C++
		  valueWeight=new ArrayList<Double>(Collections.nCopies(data.numDistinctValues(attrIdx), 0.0));   //valueWeight.create(discNoValues[i] + 1, 0.0); C++
                  squaredValues=new ArrayList<Double>(Collections.nCopies(data.numDistinctValues(attrIdx), 0.0)); //squaredValues.create(discNoValues[i] + 1, 0.0); C++
                  
		  for (j = 0 ; j < trainSize ; j++){
			 idx = (int) data.instance(j).value(attrIdx);//DiscValues.get(j).get(i);    //DiscValues(j,i); ... classValue = DiscValues(j, 0) ; ... attrValue = DiscValues(j, discIdx); index of discrete attr 
			 value = data.instance(j).classValue();     //NumValues.get(j).get(0);   //NumValues(j,0);
                         //if(j==100)
                         //System.out.println("index "+idx);
                         //System.out.println("Value class "+valueClass.get(idx));
                         //System.out.println("weight "+weight.get(j) * value);
			 valueClass.set(idx, valueClass.get(idx)+ weight.get(j) * value);   //valueClass[idx] += weight[j] * value;
			 valueWeight.set(idx, valueWeight.get(idx) + weight.get(j));        //valueWeight[idx] += weight[j];
			 squaredValues.set(idx, squaredValues.get(idx)+weight.get(j)*value*value);      //squaredValues[idx] += weight[j] * sqr(value);
		  }
		  //sortedMean.create(discNoValues[i]);   //define size of sortedMean
                  sortedMean=new ArrayList<SortRec>(discNoValues.get(attrIdx));
                  sortedMean=initArrSortRec(discNoValues.get(attrIdx));
		  RightWeight = RightSquares = RightValues = 0.0;
		  OKvalues = 0;
		  //for (j = 1 ; j <= discNoValues[i]; j++){
                  for (j = 0 ; j < discNoValues.get(attrIdx); j++){
                        if (valueWeight.get(j) > epsilon){
                            //System.out.println(valueClass.get(j));
                            //System.out.println(valueWeight.get(j));

                            sortedMean.get(OKvalues).setKey(valueClass.get(j) / valueWeight.get(j));        //sortedMean[OKvalues].key = valueClass[j] / valueWeight[j];
                            sortedMean.get(OKvalues).setValue(j);                                           //sortedMean[OKvalues].value = j;
                /*???*/     //sortedMean.set(OKvalues, new SortRec(valueClass.get(j) / valueWeight.get(j)));  
                /*???*/     //sortedMean.set(OKvalues, new SortRec(j));                                       
	                    
                            OKvalues++;

                            RightWeight +=valueWeight.get(j);       //RightWeight += valueWeight[j];
                            RightSquares += squaredValues.get(j);   //RightSquares += squaredValues[j];
                            RightValues +=valueClass.get(j);        //RightValues += valueClass[j];
			 }
		  }
		  totalWeight = RightWeight;
	/*???*/	  //sortedMean.setFilled(OKvalues);       // sets the point to which array is filled
		  
                  Collections.sort(sortedMean); //sortedMean.qsortAsc();
		  bestEstimate = Double.MAX_VALUE;
		  LeftWeight = LeftSquares = LeftValues = 0.0;
		  int upper = OKvalues - 1;
		  for (j = 0 ; j < upper ; j++){
			  idx = sortedMean.get(j).getValue();                //idx = sortedMean[j].value;
			  LeftSquares += squaredValues.get(idx);    //LeftSquares += squaredValues[idx];
			  LeftValues +=valueClass.get(idx);         //LeftValues += valueClass[idx];
			  LeftWeight +=valueWeight.get(idx);        //LeftWeight += valueWeight[idx];
			  RightSquares -=squaredValues.get(idx);    //RightSquares -= squaredValues[idx];
			  RightValues -=valueClass.get(idx);        //RightValues -= valueClass[idx];
			  RightWeight -=valueWeight.get(idx);       //RightWeight -= valueWeight[idx];
			  pLeft = LeftWeight / totalWeight;
			  variance = LeftSquares / LeftWeight - Math.pow((LeftValues / LeftWeight),2); //variance = LeftSquares / LeftWeight - sqr(LeftValues / LeftWeight);
			  if (LeftWeight > epsilon && variance > 0.0){
				estimate = pLeft * variance;
			  }
			  else
			  {
				estimate = 0.0;
			  }

			  variance = RightSquares / RightWeight - Math.pow((RightValues / RightWeight),2);   //variance = RightSquares / RightWeight - sqr(RightValues / RightWeight);
			  if (LeftWeight > epsilon && variance > 0.0){
				 estimate += ((double)1.0 - pLeft) * variance;
			  }

			  if (estimate < bestEstimate){
				 bestEstimate = estimate;
			  }
		  }
		  //DiscEstimation.set(attrIdx, - bestEstimate);//DiscEstimation[i] = - bestEstimate;
	   
           return - bestEstimate;
    }
    
    public double mseNumericAttr(int attrIdx, Instances data){
	int j, idx, OKvalues;
	double totalWeight, bestEstimate, estimate, pLeft, variance, LeftValues, LeftSquares, LeftWeight, RightValues, RightSquares, RightWeight;
        int trainSize=data.numInstances();
        
        //weight=calculateWeights(data);   
        //NumEstimation=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0));  //inicializacija ... napolnimo z ničlami
        //NumEstimation=new ArrayList<Double>(Collections.nCopies(contAttrTo-contAttrFrom, 0.0));  //inicializacija ... napolnimo z ničlami
        double splitPoint=Double.MAX_VALUE;//=new ArrayList<Double>(Collections.nCopies(data.numAttributes()-1, 0.0));
	// continuous values
	   double dVal;
           int numOfMissing=data.attributeStats(attrIdx).missingCount;
	   ArrayList<SortRec> sortedAttr = new ArrayList<SortRec>(trainSize-numOfMissing);    //marray<sortRec> sortedAttr = new marray<sortRec>(TrainSize);
           //ArrayList<SortRec> sortedAttr = new ArrayList<SortRec>(trainSize); 
           
           sortedAttr=initArrSortRec(trainSize-numOfMissing);
           //sortedAttr=initArrSortRec(trainSize);


		  RightWeight = RightSquares = RightValues = 0.0;
		  OKvalues = 0;
		  for (j = 0 ; j < trainSize ; j++){
                      if(Double.isNaN(data.instance(j).value(attrIdx))){ //if (NumValues.get(j).get(i)==null) //if (isNAcont(NumValues(j,i)))  int isNAcont(x) -> R_IsNA(x) ... kontroling missing value
                         //System.out.println("Manjka idx je: "+j+" OKvalues: "+OKvalues);
                        continue;
                        }
                        //sortedAttr.get(OKvalues).setKey(NumValues.get(j).get(i));           //sortedAttr[OKvalues].key = NumValues(j,i);
                        sortedAttr.get(OKvalues).setKey(data.instance(j).value(attrIdx));           //sortedAttr[OKvalues].key = NumValues(j,i);
                        sortedAttr.get(OKvalues).setValue(j);                               //sortedAttr[OKvalues].value = j;
			 //RightWeight += weight.get(j);                      //RightWeight += weight[j];
                         RightWeight += data.instance(j).weight();                      //RightWeight += weight[j];
			 //dVal = weight.get(j) * NumValues.get(j).get(0);    //dVal = weight[j] * NumValues(j,0); ... class value
                         dVal = data.instance(j).weight() * data.instance(j).classValue();//weight.get(j) * NumValues.get(j).get(0);    //dVal = weight[j] * NumValues(j,0); ... class value
			 RightValues += dVal;
                         //dVal *= NumValues.get(j).get(0);                   //dVal *= NumValues(j,0);
                         dVal *= data.instance(j).classValue();                   //dVal *= NumValues(j,0);
			 RightSquares += dVal;
			 OKvalues++;
		  }
		  totalWeight = RightWeight;
	/*???*/	  //sortedAttr.setFilled(OKvalues);   //// sets the point to which array is filled
		  Collections.sort(sortedAttr);     //sortedAttr.qsortAsc();
		  bestEstimate = Double.MAX_VALUE;
		  LeftWeight = LeftSquares = LeftValues = 0.0;
		  j = 0;
		  while (j < OKvalues){
			 // collect cases with the same value of the attribute - we cannot split between them
			 do{
			   idx = sortedAttr.get(j).getValue();   //idx = sortedAttr[j].value;
                           //dVal = weight.get(idx) * NumValues.get(idx).get(0);      //dVal = weight[idx] * NumValues(idx, 0);
                           dVal = data.instance(idx).weight() * data.instance(idx).classValue();      //dVal = weight[idx] * NumValues(idx, 0);
			   LeftValues += dVal;
			   RightValues -= dVal;
			   //dVal *= NumValues.get(idx).get(0);   //dVal *= NumValues(idx, 0);
                           dVal *= data.instance(idx).classValue();   //dVal *= NumValues(idx, 0);
			   LeftSquares += dVal;
			   RightSquares -= dVal;
			   //LeftWeight += weight.get(idx);       //LeftWeight += weight[idx];
                           LeftWeight += data.instance(idx).weight();       //LeftWeight += weight[idx];
                           //RightWeight -= weight.get(idx);      //RightWeight -= weight[idx];
			   RightWeight -= data.instance(idx).weight();      //RightWeight -= weight[idx];
			   j++;
			 } while (j < OKvalues && sortedAttr.get(j).getKey() == sortedAttr.get(j-1).getKey());//while (j < OKvalues && sortedAttr[j].key == sortedAttr[j - 1].key);
			 if (j == OKvalues)
				break;
			 pLeft = LeftWeight / totalWeight;
			 variance = LeftSquares / LeftWeight - Math.pow(LeftValues / LeftWeight,2);    //variance = LeftSquares / LeftWeight - sqr(LeftValues / LeftWeight);
			 if (LeftWeight > epsilon && variance > 0.0){
				estimate = pLeft * variance;
			 }
			 else
			 {
				estimate = 0.0;
			 }
                         variance = RightSquares / RightWeight - Math.pow(RightValues / RightWeight,2);        //variance = RightSquares / RightWeight - sqr(RightValues / RightWeight);
			 if (RightWeight > epsilon && variance > 0.0){
				estimate += (1.0 - pLeft) * variance;
			 }
			 if (estimate < bestEstimate){
                             //System.out.println("estimate: "+estimate);
				bestEstimate = estimate;
                                splitPoint=(sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0;
                                    //splitPoint.set(attrIdx, (sortedAttr.get(j).getKey() + sortedAttr.get(j-1).getKey()) / 2.0);//splitPoint[i] = (sortedAttr[j].key + sortedAttr[j - 1].key) / 2.0;
			 }
		  }
		  //NumEstimation.set(i, - bestEstimate);//NumEstimation[i] = - bestEstimate;
	   
           //XplainAttrConstr.printArrayList(splitPoint);
//           System.out.println("     Split point: "+splitPoint);
           return - bestEstimate;
    }
    
    public static ArrayList<Double> calculateWeights(Instances data){
    ArrayList<Double> weights = new ArrayList<Double>();
        for(int i=0;i<data.numInstances();i++)
            weights.add(data.instance(i).weight());
    return weights;
    }
    
    public static ArrayList<SortRec> initArrSortRec(int size){
    ArrayList<SortRec> tmp=new ArrayList<SortRec>(size);
        for(int i=0;i<size;i++){
            tmp.add(i, new SortRec());
        }
        return tmp;
    } 
    

}
