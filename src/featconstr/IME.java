package featconstr;

import java.util.Arrays;
import java.util.Random;
import java.util.Vector;
import jsc.combinatorics.Permutation;
import jsc.combinatorics.Permutations;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author bostjan
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public class IME{
    public static int CLASS_VALUE_TO_EXPLAIN =1;    //0 = first class value
    public static int CLASS_IDX = -1;               //default = -1 (last attribute is class attribute)
    public static Random rand = new Random();

    //equal sampling
    public static double[][] explainAllDatasetES(Instances train, Instances test, Classifier K, int numSamples, int classToExplain)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        int classIdx = CLASS_IDX;
        if (classIdx < 0 || classIdx >= test.numAttributes())
            test.setClassIndex(test.numAttributes()-1);
        else
            test.setClassIndex(classIdx);
	boolean isClassification = true;
	if (test.classAttribute().isNumeric()) 
            isClassification = false;
       
        for (int i = 0; i < test.numInstances(); i++)
            instanceExplanation[i] = explainInstanceNew(K, train, new Instances(test,i,1), numSamples, isClassification, classToExplain); //Štrumbelj's modifications
            //instanceExplanation[i] = explainInstance(K, train, new Instances(test,i,1), numSamples, isClassification, classToExplain);
        
        return instanceExplanation;
    }

    //adaptive sampling - stopping criteria is sum of samples
    public static double[][] explainAllDatasetAS(Instances train, Instances test, Classifier K, int mMin, int maxS, int classToExplain)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        int classIdx = CLASS_IDX;
        if (classIdx < 0 || classIdx >= test.numAttributes())
            test.setClassIndex(test.numAttributes()-1);
        else
            test.setClassIndex(classIdx);
	boolean isClassification = true;
	if (test.classAttribute().isNumeric()) 
            isClassification = false;
       
        for (int i = 0; i < test.numInstances(); i++)
            instanceExplanation[i]= explainInstAdapSmp(K, train, new Instances(test,i,1), isClassification, classToExplain, mMin, maxS);
        
        return instanceExplanation;
    }
    
    //adaptive sampling - stopping criteria is approxamization error for all attributes
    public static double[][] explainAllDatasetAS(Instances train, Instances test, Classifier K, int mMin, int classToExplain, double e, int pctErr)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        int classIdx = CLASS_IDX;
        if (classIdx < 0 || classIdx >= test.numAttributes())
            test.setClassIndex(test.numAttributes()-1);
        else
            test.setClassIndex(classIdx);
	boolean isClassification = true;
	if (test.classAttribute().isNumeric()) 
            isClassification = false;
       
        for (int i = 0; i < test.numInstances(); i++)
            instanceExplanation[i]= explainInstAdapSmp(K, train, new Instances(test,i,1), isClassification, classToExplain, mMin, e, pctErr);
        
        return instanceExplanation;
    }
    
    //sampling based on the approxamization error
    public static double[][] explainAllDatasetAES(Classifier K, Instances train, Instances test, boolean isClassification, int classToExplain, int mMin, double e, int pctErr)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        for (int i = 0; i < test.numInstances(); i++)
            instanceExplanation[i] = explainInstAproxErr(K, train, new Instances(test,i,1),isClassification, classToExplain, mMin, e, pctErr);
        
        return instanceExplanation;
    }
    
    //credits to Erik Štrumbelj
    //*Štrumbelj, E., & Kononenko, I. (2011, April). A general method for visualizing and explaining black-box regression models. In International Conference on Adaptive and Natural Computing Algorithms (pp. 21-30). Springer, Berlin, Heidelberg.
    public static double[] explainInstance(Classifier K, Instances D, Instances I1, int m, boolean isClassification, int classToExplain) throws Exception{
        //algorithm for calculating attribute contributions in a particular case (Algorithm 1* from the article, adapted to working on all combinations of continous/nominal attributes and classification/regression problems)
        //INPUT: prediction model, dataset, instance to explain, number of samples m, classification problem (true or false), class to explain
	//OUTPUT: vector of contributions of individual attributes for a given case	
	int numOfFeats = D.numAttributes() - 1;
   
        double[] result = new double[numOfFeats];
        Permutations permuts = new Permutations(numOfFeats);
        
	for (int i = 0; i < m; i++){	//repeat m times
            Permutation tempPermutation = permuts.randomPermutation();
            int[] intPermutation = tempPermutation.toIntArray();    //we choose a random permutation
            for (int feature = 0; feature < numOfFeats; feature++){ //for each attribute         	
            	// *** ASSEMBLE 2 EXAMPLES ****
                //here, the value of the attribute is randomly selected by randomly selecting an instance from the dataset and overwriting the value for a given attribute (separately for each attribute)
            	
                //in the first case, we randomly select the values of the attributes that are to the right of the current attribute (including the current attribute); the remaining values are taken from the instance we are explaining
            	Instances instance = new Instances(I1,0, 1);
                int featureIndex = intPermutation[feature]-1;   //because we have permutations e.g., from 1 to 6, we have to subtract 1 because we have indexes from 0 to 5
                
                for (int j = D.numAttributes() - 2; j >= feature; j--){ //D.numAttributes() returns num of attributes + class, that's why -2
                    int rndInst=rand.nextInt(D.numInstances());
                	double value = D.instance(rndInst).value(intPermutation[j]-1); //the value of a given attribute in a randomly selected case (instance)
                	instance.instance(0).setValue(intPermutation[j]-1, value);
                }
                double predictionLo = 0;                  
    			if (isClassification)
    				predictionLo = K.distributionForInstance(instance.instance(0))[classToExplain];
    			else
    				predictionLo = K.classifyInstance(instance.instance(0));		              
                
                //in the second case, we randomly select the values of the attributes that are strictly to the right of the current attribute; the remaining values are taken from the instance we are explaining
                instance = new Instances(I1,0, 1);
                for (int j = D.numAttributes() - 2; j > feature; j--){
                    int rndInst=rand.nextInt(D.numInstances());
                	double value = D.instance(rndInst).value(intPermutation[j]-1);
                	instance.instance(0).setValue(intPermutation[j]-1, value);
                }

                double predictionHi = 0;    
                if (isClassification)
                    predictionHi = K.distributionForInstance(instance.instance(0))[classToExplain];
                else
                    predictionHi = K.classifyInstance(instance.instance(0));
          
                result[featureIndex] += predictionHi-predictionLo;                                
                
		}
            }
		
            for (int featureIndex = 0; featureIndex < numOfFeats; featureIndex++) 
                result[featureIndex] /= m; //in the end, we divide by the number of samples

        return result;
    }
    
    //credits to Erik Štrumbelj    
    public static double[] explainInstanceNew(Classifier K, Instances D, Instances I1, int m, boolean isClassification, int classValueToExplain) throws Exception{
        int numOfFeats = D.numAttributes() - 1;
        Random internalRand = new Random();
        double[] result = new double[numOfFeats];
        Permutations permuts = new Permutations(numOfFeats);

        for (int i = 0; i < m; i++){ // equal sampling
            Permutation tempPermutation = permuts.randomPermutation();
            int[] intPermutation = tempPermutation.toIntArray();

            for (int feature = 0; feature < numOfFeats; feature++){
                Instances instance = new Instances(I1,0, 1);
                int featureIndex = intPermutation[feature]-1;
                
                for (int j = D.numAttributes() - 2; j > feature; j--){
                    double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[j]-1); //the value of a given attribute in a randomly selected case (instance)
                    instance.instance(0).setValue(intPermutation[j]-1, value);
                }
                
                double predictionHi = 0;
                if (isClassification)
                    predictionHi = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionHi = K.classifyInstance(instance.instance(0));

                double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[feature]-1); //the value of a given attribute in a randomly selected case (instance)
                instance.instance(0).setValue(intPermutation[feature]-1, value);
                
                double predictionLo = 0;
                if (isClassification)
                    predictionLo = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionLo = K.classifyInstance(instance.instance(0));

                result[featureIndex] += predictionHi-predictionLo;
            }
        }

	for (int featureIndex = 0; featureIndex < numOfFeats; featureIndex++)
            result[featureIndex] /= m;
	
        return result;
    }
    
    //using one sample for sampling
    public static double explainInstanceOneS(Classifier K, Instances D, Instances I1, int feat, boolean isClassification, int classValueToExplain) throws Exception{
	int numOfFeats = D.numAttributes() - 1;
        Random internalRand = new Random();  
        Permutations permuts = new Permutations(numOfFeats);
        Permutation tempPermutation = permuts.randomPermutation();
        int[] intPermutation = tempPermutation.toIntArray();    //use random permutation
            	
        Instances instance = new Instances(I1,0, 1);

        int feature=0; 
        for(int i=0;i<intPermutation.length;i++)
            if(feat+1==intPermutation[i])
                feature=i;
        
        for (int j = D.numAttributes() - 2; j > feature; j--){
            double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[j]-1);   //the value of a given attribute in a randomly selected case (instance)
            instance.instance(0).setValue(intPermutation[j]-1, value);                    
        }

        double predictionHi = 0;
        if(isClassification)
            predictionHi = K.distributionForInstance(instance.instance(0))[classValueToExplain];
        else
            predictionHi = K.classifyInstance(instance.instance(0));	              

        double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[feature]-1); //the value of a given attribute in a randomly selected case (instance)
        instance.instance(0).setValue(intPermutation[feature]-1, value);
                
        double predictionLo = 0;
        if(isClassification)
            predictionLo = K.distributionForInstance(instance.instance(0))[classValueToExplain];
        else
            predictionLo = K.classifyInstance(instance.instance(0));
	
        double finalC = +predictionHi-predictionLo;

        return finalC;
    }
    
    //credits to Erik Štrumbelj
    //*Štrumbelj, E., & Kononenko, I. (2011, April). A general method for visualizing and explaining black-box regression models. In International Conference on Adaptive and Natural Computing Algorithms (pp. 21-30). Springer, Berlin, Heidelberg.
    public static double[] explainValue(Classifier K, Instances D, int attrIdx, double valIdx, int m, boolean isNominal, double[][] extremeValues, boolean isClassification){
        //algorithm for calculating the general contribution of a value of an attribute (Algorithm 2 * from the paper, adapted to work on all combinations of continuous / nominal attributes and classification / regression)
        //INPUT: prediction model, dataset, index of an attribute, index of a value / value [depending on classification or regression], number of samples m, nominal attribute (true or false), min and max attribute values, classification problem (true or false)
        //OUTPUT: contribution of a value (psi) and stdev contribution
	
        double[] res = new double[2];
	double[] psi = new double[m];	//e.g., default value is 1000
	try{	
            for (int i = 0; i < m; i++){    //repeat m times, once for each sample
                //default
		Instances instance1 = new Instances(D,0,1); 
		Instances instance2 = new Instances(D,0,1);
					
		for (int j = 0; j < D.numAttributes()-1; j++){			
                    //randomly set attribute values ​​(same in both cases)
                    //we set the value of the attribute (which we explaining) only in the second case to the value we are interested in
                    if (D.attribute(j).isNominal()){ 
                        String value = D.attribute(j).value(rand.nextInt(D.attribute(j).numValues()));
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
                            if (j == attrIdx) 
                                instance2.instance(0).setValue(j, D.attribute(j).value((int)valIdx));
                    }
                    else{
                    //we need min / max to know from which interval to choose random values
                    //extremeValues ​​- 2D table in which are min and max attribute values ​​([j][0]-min, [j][1]-max)
                        double value = rand.nextFloat() * (extremeValues[j][1] - extremeValues[j][0]) + extremeValues[j][0]; //rand * (max-min) + min
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
                        //set the value of the attribute to the value we are interested in
			if (j == attrIdx)
                            instance2.instance(0).setValue(j, valIdx);					
                    }
		}
		double p2 = -1;
		double p1 = -1;
	
		if (isClassification){  //at classification we look at the probability of a prediction for the desired class, at regression only the prediction
                    p2 = K.distributionForInstance(instance2.instance(0))[CLASS_VALUE_TO_EXPLAIN];
                    p1 = K.distributionForInstance(instance1.instance(0))[CLASS_VALUE_TO_EXPLAIN];
		}
		else{
                    p2 = K.classifyInstance(instance2.instance(0));
                    p1 = K.classifyInstance(instance1.instance(0));	
		}
                psi[i] = p2 - p1;   //the difference between the prediction with the value and the "no" value
            }
            
            res[0] = mean(psi);     //contribution of the value (psi), mean is a method that calculates the average in a 1D table
            res[1] = Math.sqrt(var(psi,mean(psi)));     //stdev of contribution
            }
            catch(Exception e){
                e.printStackTrace();
            }
        return res;
    }
    
    //similar as explainValue method, returns only psi; explainValue method returns mean(psi) and stdev(psi) 
    public static double[] explainValueAttrImp(Classifier K, Instances D, int attrIdx, double valIdx, int m, boolean isNominal, double[][] extremeValues, boolean isClassification){
	double[] psi = new double[m];
	try{	
            for (int i = 0; i < m; i++){
		Instances instance1 = new Instances(D,0,1); 
		Instances instance2 = new Instances(D,0,1);
					
		for (int j = 0; j < D.numAttributes()-1; j++){			
                    if (D.attribute(j).isNominal()){ 
                        String value = D.attribute(j).value(rand.nextInt(D.attribute(j).numValues()));
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
                            if (j == attrIdx) 
                                instance2.instance(0).setValue(j, D.attribute(j).value((int)valIdx));
                    }
                    else{
                        double value = rand.nextFloat() * (extremeValues[j][1] - extremeValues[j][0]) + extremeValues[j][0];
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
			if (j == attrIdx)
                            instance2.instance(0).setValue(j, valIdx);					
                    }
		}
		double p2 = -1;
		double p1 = -1;
	
		if (isClassification){
                    p2 = K.distributionForInstance(instance2.instance(0))[CLASS_VALUE_TO_EXPLAIN];
                    p1 = K.distributionForInstance(instance1.instance(0))[CLASS_VALUE_TO_EXPLAIN];
		}
		else{
                    p2 = K.classifyInstance(instance2.instance(0));
                    p1 = K.classifyInstance(instance1.instance(0));	
		}
                psi[i] = p2 - p1;
            }
        }
        catch(Exception e){
            e.printStackTrace();
        }
        return psi;
    }
      
    //adaptive sampling; sum of samples (constraint) based on variance; Algorithm 2 from the paper*
    //*Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and information systems, 41(3), 647-665.
    public static double[] explainInstAdapSmp(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain, int mMin, int mMax) throws Exception{
        int numOfAttr=D.numAttributes()-1;
        int m[]=new int[numOfAttr];                     //set all values to 0 ... first, number of samples for each attribute are 0
        double psi[]=new double[numOfAttr];             //set all values to 0 ... first, all contributions are 0 
        double tmpVariance[]=new double[numOfAttr];     //set all values to 0 ... first, all contributions are 0
        double tmpMean[]=new double[numOfAttr];         //set all values to 0 ... first, all contributions are 0
        Vector v[]= new Vector[numOfAttr];
	double diff[]=new double[numOfAttr];            //for minimizing squared error
        
        for(int i=0;i<D.numAttributes()-1;i++)
            v[i]=new Vector();

        for(int i = 0 ; i < psi.length; i++ )
            psi[i] = 0;
        
        for(int i=0;i<numOfAttr;i++)
            m[i]=mMin;
        
        double tmpPsi;
        //get initial variances for each feature
        for(int i=0;i<numOfAttr;i++){
            for(int j=0;j<mMin;j++){
                tmpPsi=explainInstanceOneS(K, D, I, i, isClassification,classValueToExplain);
                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3];
            }
        }
		
	//for minimizing squared error
	for(int i=0;i<numOfAttr;i++)
            diff[i]=Math.sqrt(tmpVariance[i]/m[i]) - Math.sqrt(tmpVariance[i]/(m[i]+1));			
			
	int j;
        while(sumArr(m)<mMax){
            j=idxOfMaxValue(diff); //index of max value in array diff
            tmpPsi=explainInstanceOneS(K, D, I, j, isClassification,classValueToExplain);
            v[j].add(tmpPsi);
            tmpMean[j]=mean(v[j]);
            psi[j] += tmpPsi;
            tmpVariance[j]=incrementalVar(v[j].size(), tmpVariance[j], tmpMean[j], tmpPsi)[3];
            m[j]++;
            diff[j]=Math.sqrt(tmpVariance[j]/m[j])-Math.sqrt(tmpVariance[j]/(m[j]+1));	//we have to correct values in diff
        }

        for(int i=0;i<psi.length;i++)
            psi[i]=psi[i]/m[i];
        
        return psi;	
    }
    
    //adaptive sampling; sampling based on aproximization error, sampling while(Math.sqrt(zSquared*tmpVariance[i]/m[i]) > e); Algorithm 2 from the paper*
    //*Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and information systems, 41(3), 647-665.
    public static double[] explainInstAdapSmp(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain, int mMin, double e, int pctErr) throws Exception{
        int numOfAttr=D.numAttributes()-1;
        int m[]=new int[numOfAttr];                     //set all values to 0 ... first, number of samples for each attribute are 0
        double psi[]=new double[numOfAttr];             //set all values to 0 ... first, all contributions are 0 
        double tmpVariance[]=new double[numOfAttr];     //set all values to 0 ... first, all contributions are 0
        double tmpMean[]=new double[numOfAttr];         //set all values to 0 ... first, all contributions are 0
        Vector v[]= new Vector[numOfAttr];
	double diff[]=new double[numOfAttr];            //for minimizing squared error
        
        for(int i=0;i<D.numAttributes()-1;i++)
            v[i]=new Vector();

        for(int i = 0 ; i < psi.length; i++ )
            psi[i] = 0;
        
        for(int i=0;i<numOfAttr;i++)
            m[i]=mMin;                
        
        double tmpPsi;
        //get initial variances for each feature
        for(int i=0;i<numOfAttr;i++){
            for(int j=0;j<mMin;j++){
                tmpPsi=explainInstanceOneS(K, D, I, i, isClassification,classValueToExplain);
                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3];
            }
        }
        
        double z;
        switch(pctErr){
            case 90:z=1.285;break;  //for 90% of probability
            case 95:z=1.645;break;  //for 95% of probability
            case 99:z=2.325;break;  //for 99% of probability
            default: z=1.285;       //we set default Z for 90%
        }
 
        double zSquared=Math.pow(z, 2);        
        for(int i=0;i<numOfAttr;i++){
            while(Math.sqrt(zSquared*tmpVariance[i]/m[i]) > e){  
                tmpPsi=explainInstanceOneS(K, D, I, i, isClassification,classValueToExplain);
                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3];
                m[i]++;
            }
        }
        
        for(int i=0;i<psi.length;i++)
            psi[i]=psi[i]/m[i];
        
        return psi;	
    }
     
    //sampling based on the aproxamization error, calculate number of samples for each attribute based on mMin samples
    //see page 10 in Štrumbelj, E., & Kononenko, I. (2010). An efficient explanation of individual classifications using game theory. The Journal of Machine Learning Research, 11, 1-18.
    public static double[] explainInstAproxErr(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain, int mMin, double e, int pctErr) throws Exception{
        int numOfAttr=D.numAttributes()-1;
        int m[]=new int[numOfAttr];                     //set all values to 0 ... first, number of samples for each attribute are 0
        double psi[]=new double [numOfAttr];            //set all values to 0 ... first, all contributions are 0 
        double tmpVariance[]=new double [numOfAttr];    //set all values to 0 ... first, all contributions are 0
        double tmpMean[]=new double [numOfAttr];        //set all values to 0 ... first, all contributions are 0
        Vector v[]= new Vector[numOfAttr];
        
        for(int i=0;i<D.numAttributes()-1;i++)
            v[i]=new Vector();

        for(int i = 0 ; i < psi.length; i++ )
            psi[i] = 0;

        double tmpPsi;
        //get initial variances for each feature by using mMin samples
        for(int i=0;i<numOfAttr;i++){
            for(int j=0;j<mMin;j++){
                tmpPsi=explainInstanceOneS(K, D, I, i, isClassification,classValueToExplain);
                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3];
            }
            m[i]=mMin;
        }

        double z;
        switch(pctErr){
            case 90:z=1.285;break;  //for 90% of probability
            case 95:z=1.645;break;  //for 95% of probability
            case 99:z=2.325;break;  //for 99% of probability
            default: z=1.285;       //we set default Z for 90%
        }
 
        double zSquared=Math.pow(z, 2);
        double eSquared=Math.pow(e, 2);
		
        //calculate number of samples for each attribute
        int tmpM;
	for(int i=0;i<numOfAttr;i++){
            tmpM=(int)Math.round((zSquared*tmpVariance[i])/eSquared);
            if(tmpM>mMin)
                m[i]=tmpM;
        }
        
        for(int i=0;i<numOfAttr;i++)
            if(m[i]>mMin)
                for(int j=0;j<(m[i]-mMin);j++)                
                    psi[i] +=explainInstanceOneS(K, D, I, i, isClassification,classValueToExplain);
    
        for(int i=0;i<psi.length;i++)
            psi[i]=psi[i]/m[i];
        
        return psi;
    }
    
    //minimizing the number of samples by estimation the sample variance
    //adaptive sampling - Algorithm 2 from the paper*, test of stopping condition (m[i]<Math.round((zSquared*tmpVariance[i])/eSquared) && m[i]<mMax)
    //*Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and information systems, 41(3), 647-665.
    public static double[] explainInstBasedOnVar(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain, int mMin, int mMax, double e, int pctErr) throws Exception{
        int numOfAttr=D.numAttributes()-1;
        int m[]=new int[numOfAttr];                     //set all values to 0 ... first, number of samples for each attribute are 0
        double psi[]=new double [numOfAttr];            //set all values to 0 ... first, all contributions are 0 
        double tmpVariance[]=new double [numOfAttr];    //set all values to 0 ... first, all contributions are 0
        double tmpMean[]=new double [numOfAttr];        //set all values to 0 ... first, all contributions are 0
        Vector v[]= new Vector[numOfAttr];
        
        for(int i=0;i<D.numAttributes()-1;i++)
            v[i]=new Vector();

        for(int i = 0 ; i < psi.length; i++ )
            psi[i] = 0;
        
        for(int i=0;i<numOfAttr;i++)
            m[i]=mMin;
        
        double tmpPsi;
        //get initial variances for each feature
        for(int i=0;i<numOfAttr;i++){
            for(int j=0;j<mMin;j++){
                tmpPsi=explainInstanceOneS(K, D, I, i, isClassification,classValueToExplain);
                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3];
            }
        }

        double z;
        switch(pctErr){
            case 90:z=1.285;break;  //for 90% of probability
            case 95:z=1.645;break;  //for 95% of probability
            case 99:z=2.325;break;  //for 99% of probability
            default: z=1.285;       //we set default Z for 90%
        }
 
        double zSquared=Math.pow(z, 2);
        double eSquared=Math.pow(e, 2);

        for(int i=0;i<numOfAttr;i++){
            while(m[i]<Math.round((zSquared*tmpVariance[i])/eSquared) && m[i]<mMax){
                tmpPsi=explainInstanceOneS(K, D, I, i, isClassification,classValueToExplain);
                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3];
                m[i]++;
            }
        }
    
        for(int i=0;i<psi.length;i++)
            psi[i]=psi[i]/m[i];
        
        return psi;
    }
    
    public static double[] incrementalVar(double k, double M2, double mean, double sample){
        //k number of elements in data
	//previous variance M2 as parameter in method
	//sample - sample that we add
	k++;	
	double d = sample - mean;		
	mean += d / k;
	M2 += d * (sample - mean);
		
        double var = M2 / (k - 1);         
	//use k instead of (k-1) if want to compute the exact variance of the given data; use (k-1) if data are samples of a larger population
        
	return new double[]{k,M2,mean,var};	
    }
    
    public static int sumArr(int tab []){     //variance
        return Arrays.stream(tab).sum();
    }
    
    public static double var(Vector d){    //variance in vector
        double m1= 0;
	for (int i = 0; i < d.size(); i++){
            m1 += (Double)d.elementAt(i);
	}
		
	m1 /= d.size();
	double sum = 0;
	for (int i = 0; i < d.size(); i++) 
            sum += ((Double)d.elementAt(i) - m1) * ((Double)d.elementAt(i)- m1);
	
        return sum / d.size();
    }	
    
    public static double var(double[] d){
        double m1= 0;
        for (int i = 0; i < d.length; i++)
            m1 += d[i];
	
	m1 /= d.length;

	double sum = 0;
	for (int i = 0; i < d.length; i++) 
            sum += (d[i] - m1) * (d[i]- m1);
        
        return sum / d.length;
    }
	
    public static double var(double[] d, double m){
        double sum = 0;
        for (int i = 0; i < d.length; i++) 
            sum += (d[i] - m) * (d[i] - m);
        
        return sum / d.length;
    }
	
    public static double var(double[] d, double m, int max){
        double sum = 0;
	for (int i = 0; i < max; i++) 
            sum += (d[i] - m) * (d[i] - m);
    
        return sum /max;
    }
    
    public static double mean(Vector d){ //mean in Vector
        double sum = 0;
	for (int i = 0; i < d.size(); i++) 
            sum += (Double)d.elementAt(i);
        
        return sum / d.size();
    }
    
    public static double mean(double[] d){
        double sum = 0;
        for (int i = 0; i < d.length; i++) 
            sum += d[i];
        
        return sum / d.length;
    }
    
    public static int idxOfMaxValue(double[] array){
	int maxValAt = 0;
	for(int i = 1; i < array.length; i++)
            maxValAt = array[i] > array[maxValAt] ? i : maxValAt;
        return maxValAt;
    }
}
