/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package featconstr;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Vector;
import jsc.combinatorics.Permutation;
import jsc.combinatorics.Permutations;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
//import static featconstr.FeatConstr.mean;
//import static featconstr.FeatConstr.print1d;
//import static featconstr.FeatConstr.rnd2;
//import static featconstr.FeatConstr.var;
//import static featconstr.FeatConstr.N_SAMPLES;

/**
 *
 * @author bostjan
 */


public class IME {
    public static int CLASS_VALUE_TO_EXPLAIN =1; // 0 = first class value
    public static int CLASS_IDX = -1; // default = -1 (last attribute is class attribute)
    // some regression models:
    //public static Classifier predictionModel = new  weka.classifiers.functions.LeastMedSq();
    //public static Classifier predictionModel = new  weka.classifiers.functions.SMOreg();
    //public static Classifier predictionModel = new  weka.classifiers.functions.LinearRegression(); // linear regression
    //public static Classifier predictionModel = new  weka.classifiers.functions.MultilayerPerceptron(); // neural network
    //public static Classifier predictionModel = new  weka.classifiers.trees.M5P(); // regression tree
    //public static Classifier predictionModel = new  weka.classifiers.functions.SVMreg(); // regression tree
    //public static Classifier predictionModel = new  weka.classifiers.trees.RandomForest(); // weka 3.7 podpira tudi regresijo
    //public static Classifier predictionModel = new  weka.classifiers.functions.SimpleLinearRegression();
    //public static Classifier predictionModel = new  weka.classifiers.trees.M5P();
    public static Random rand = new Random();

    public static double[][] explainAllDataset(Instances train, Instances test, Classifier K, int numSamples, int minS, int maxS, boolean opt, int classToExplain)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        int classIdx = CLASS_IDX;
        if (classIdx < 0 || classIdx >= test.numAttributes())
            test.setClassIndex(test.numAttributes()-1);
        else
            test.setClassIndex(classIdx);
	boolean isClassification = true;
	if (test.classAttribute().isNumeric()) 
            isClassification = false;
        
        /*Random internalRand = new Random();
        System.out.println(internalRand.nextInt(100));
        System.out.println(internalRand.nextInt(100));
        System.exit(0);*/
        
        
        
        if(!opt){
            for (int i = 0; i < test.numInstances(); i++){
                //instanceExplanation[i] = explainInstance(K, train, new Instances(test,i,1), numSamples, isClassification, classToExplain);    //original by BV
                //print1d(instanceExplanation[i]);
            //System.exit(0);
                instanceExplanation[i] = explainInstanceNew(K, train, new Instances(test,i,1), numSamples, isClassification, classToExplain); //Štrumbelj's modifications
                //print1d(instanceExplanation[i]);
            //System.exit(0);
            }
        }
        else{
           for (int i = 0; i < test.numInstances(); i++){
               //instanceExplanation[i] = explainInstanceOpt(K, train, new Instances(test,i,1), isClassification, classToExplain, minS, maxS);
               instanceExplanation[i]= explainInstanceTestVar(K, train, new Instances(test,i,1), isClassification, classToExplain, maxS);
              //print1d(instanceExplanation[i]);
                //System.exit(0);
           }

        }

        //print2d(instanceExplanation);
        //System.exit(0);
        //instanceExplanation = explainInstanceExact(K, train, test, isClassification, classToExplain); //TIME COMPLEXITY!!!
        return instanceExplanation;
    }
    /*Instances(Instances source,int first,int toCopy)
        source - the set of instances from which a subset is to be created
        first - the index of the first instance to be copied
        toCopy - the number of instances to be copied*/
    
    
    public static double[][] explainAllDataset(Classifier K, Instances train, Instances test, int m[], boolean isClassification, int classToExplain)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        for (int i = 0; i < test.numInstances(); i++){
            instanceExplanation[i] = explainInstanceMaxInst(K, train, new Instances(test,i,1), m, isClassification, classToExplain);
        }
        return instanceExplanation;
    }
    
    public static double[][] explainAllDataset(Classifier K, Instances train, Instances test, boolean isClassification, int classToExplain, int mMax)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        for (int i = 0; i < test.numInstances(); i++){
            instanceExplanation[i] = explainInstanceTestVar(K, train, new Instances(test,i,1), isClassification, classToExplain, mMax);
        }
        return instanceExplanation;
    }

    public static double[][] explainAllDataset(Classifier K, Instances train, Instances test, boolean isClassification, int classToExplain,int mMin, int mMax, double e, int pctErr)throws Exception{
        double[][] instanceExplanation= new double[test.numInstances()][test.numAttributes()-1];
        for (int i = 0; i < test.numInstances(); i++){
            instanceExplanation[i] = explainInstBasedOnVar(K, train, new Instances(test,i,1),isClassification, classToExplain, mMin, mMax, e, pctErr);
        }
        return instanceExplanation;
    }
    
    
    public static double[] explainInstance(Classifier K, Instances D, Instances I1, int m, boolean isClassification, int classToExplain) throws Exception{
	// algoritem za računanje prispevkov atributov v določenem primeru (Algoritem 1 iz članka, prilagojen, da dela na vseh kombinacijah zveznih/nominalnih atributov in klasifikacije/regresije)
	// INPUT: naučen model, učna množica, primer, število vzorcev m, je klasifikacijski problem?
	// OUTPUT: vektor prispevkov posameznih atributov za podani primer	
	
	int numOfFeats = D.numAttributes() - 1;
   
        double[] result = new double[numOfFeats];
        Permutations permuts = new Permutations(numOfFeats);
        
	for (int i = 0; i < m; i++){	// ponovimo m-krat
            //System.out.println("permutation"+(i+1));
            Permutation tempPermutation = permuts.randomPermutation();
            int[] intPermutation = tempPermutation.toIntArray(); // izberemo naključno permutacijo
            /*System.out.print("permutacije indeksov atributov: ");
                    XplainAttrConstr.print1dInt(intPermutation);
           System.out.println();*/
           
            for (int feature = 0; feature < numOfFeats; feature++){		// za vsak atribut            	
            	// *** SESTAVIMO 2 PRIMERA ****
            	// tu vrednost atributa naključno izbiramo tako, da naključno izberemo primer iz učne množice in prepišemo vrednost pri danem atributu (za vsak atribut posebej)
            	
            	// pri prvem naključno izberemo vrednosti atributov, ki so desno od trenutnega atributa (vključno z); preostale vrednosti vzamemo iz primera, ki ga razlagamo
            	Instances instance = new Instances(I1,0, 1);
                int featureIndex = intPermutation[feature]-1;   // ker imamo permutacije npr. od 1 do 6, moramo odšteti 1, ker imamo indekse od 0...5
                
//                System.out.println("Original");
//                System.out.println("        "+instance.instance(0));
//                System.out.println("feature: "+feature+" featureIndex: "+featureIndex);
                for (int j = D.numAttributes() - 2; j >= feature; j--){ //D.numAttributes() returns num of attributes + class, that's why -2
                    int rndInst=rand.nextInt(D.numInstances());
                	double value = D.instance(rndInst).value(intPermutation[j]-1); // vrednost danega atributa v naključno izbranem primeru
                	instance.instance(0).setValue(intPermutation[j]-1, value);
                        
                        
                        //System.out.println("        "+instance.instance(0)+"    attr. index: "+(intPermutation[j]-1)+" rnd instance: "+rndInst+" j: "+j+" value: "+value);
                        
                }
//                System.out.println("First");
//              System.out.println("        "+instance.instance(0));
//              System.out.println("**********************");
                double predictionLo = 0;                  
    			if (isClassification)
    				predictionLo = K.distributionForInstance(instance.instance(0))[classToExplain];//predictionLo = K.distributionForInstance(instance.instance(0))[CLASS_VALUE_TO_EXPLAIN];
    			else
    				predictionLo = K.classifyInstance(instance.instance(0));		              
                
                //System.out.println("Second");
                //System.out.println("        featureIndex: "+featureIndex);
              	// pri drugem naključno izberemo vrednosti atributov, ki so strogo desno od trenutnega atributa; preostale vrednosti vzamemo iz primera, ki ga razlagamo
                instance = new Instances(I1,0, 1);
                for (int j = D.numAttributes() - 2; j > feature; j--){
                    int rndInst=rand.nextInt(D.numInstances());
                	double value = D.instance(rndInst).value(intPermutation[j]-1);
                	instance.instance(0).setValue(intPermutation[j]-1, value);

                        //System.out.println("        "+instance.instance(0)+"    attr. index: "+(intPermutation[j]-1)+" rnd instance: "+rndInst+" j: "+j+" value: "+value);  
                }
//              System.out.println("Second");
//              System.out.println("        "+instance.instance(0));
//              System.out.println("**********************");
//              System.exit(0);
                double predictionHi = 0;    
    			if (isClassification)
    				predictionHi = K.distributionForInstance(instance.instance(0))[classToExplain];//predictionHi = K.distributionForInstance(instance.instance(0))[CLASS_VALUE_TO_EXPLAIN];
    			else
    				predictionHi = K.classifyInstance(instance.instance(0));
          
                        //System.out.println("    feature index: "+featureIndex+" prediciton Hi-Lo "+rnd(predictionHi-predictionLo));
				result[featureIndex] += predictionHi-predictionLo;
                                
                                //if(feature==0 && i<11)
                                    //System.out.println("feature: "+feature+" "+" m "+i+" "+(predictionHi-predictionLo));
			}
                        //if(i>10)
                            //System.exit(0);
		}
		
		for (int featureIndex = 0; featureIndex < numOfFeats; featureIndex++) 
			result[featureIndex] /= m; // na koncu le še delimo s številom vzorcev

            return result;
    }
    
    
    public static double[] explainInstanceNew(Classifier K, Instances D, Instances I1, int m, boolean isClassification, int classValueToExplain) throws Exception
    {
        int numOfFeats = D.numAttributes() - 1;
        //Random internalRand = new Random(randomSeed);
        Random internalRand = new Random();

        double[] result = new double[numOfFeats];

        Permutations permuts = new Permutations(numOfFeats);

        for (int i = 0; i < m; i++) // equal sampling
        {
            Permutation tempPermutation = permuts.randomPermutation();
            int[] intPermutation = tempPermutation.toIntArray();

            for (int feature = 0; feature < numOfFeats; feature++)
            {
                Instances instance = new Instances(I1,0, 1);
                int featureIndex = intPermutation[feature]-1;
                
/*System.out.println("Original");
System.out.println("        "+instance.instance(0));
System.out.println("feature: "+feature+" featureIndex: "+featureIndex);*/
                
                

                for (int j = D.numAttributes() - 2; j > feature; j--){
                        double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[j]-1); // vrednost danega atributa v naključno izbranem primeru
                        instance.instance(0).setValue(intPermutation[j]-1, value);
                }
                
                
/*System.out.println("First");
System.out.println("        "+instance.instance(0));
System.out.println("**********************");*/

                double predictionHi = 0;
                if (isClassification)
                    predictionHi = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionHi = K.classifyInstance(instance.instance(0));

                    double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[feature]-1); // vrednost danega atributa v naključno izbranem primeru
                    instance.instance(0).setValue(intPermutation[feature]-1, value);
//                }

/*System.out.println("Second");
System.out.println("        "+instance.instance(0));
System.out.println("**********************");
System.exit(0);*/
                
                double predictionLo = 0;
                if (isClassification)
                    predictionLo = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionLo = K.classifyInstance(instance.instance(0));

                double finalC = +predictionHi-predictionLo;

                result[featureIndex] += predictionHi-predictionLo;

            }
        }

	for (int featureIndex = 0; featureIndex < numOfFeats; featureIndex++){
            result[featureIndex] /= m;
	}

        return result;
    }
    
    public static double explainInstanceTest(Classifier K, Instances D, Instances I1, int feat, boolean isClassification, int classValueToExplain) throws Exception{
	// algoritem za računanje prispevkov atributov v določenem primeru (Algoritem 1 iz članka, prilagojen, da dela na vseh kombinacijah zveznih/nominalnih atributov in klasifikacije/regresije)
	// INPUT: naučen model, učna množica, primer, število vzorcev m, je klasifikacijski problem?
	// OUTPUT: vektor prispevkov posameznih atributov za podani primer	
	int numOfFeats = D.numAttributes() - 1;
        Random internalRand = new Random();
        //double result =0;
     
        Permutations permuts = new Permutations(numOfFeats);
        
            //System.out.println("permutation"+(i+1));
           Permutation tempPermutation = permuts.randomPermutation();
           int[] intPermutation = tempPermutation.toIntArray(); // izberemo naključno permutacijo
           /*System.out.print("permutacije indeksov atributov: ");
                    XplainAttrConstr.print1dInt(intPermutation);
           System.out.println();*/
          
//for (int feature = 0; feature < numOfFeats; feature++){		// za vsak atribut            	
                //feature=feat;
            	// *** SESTAVIMO 2 PRIMERA ****
            	// tu vrednost atributa naključno izbiramo tako, da naključno izberemo primer iz učne množice in prepišemo vrednost pri danem atributu (za vsak atribut posebej)
            	
            	// pri prvem naključno izberemo vrednosti atributov, ki so desno od trenutnega atributa (vključno z); preostale vrednosti vzamemo iz primera, ki ga razlagamo
                Instances instance = new Instances(I1,0, 1);
                
                /*dummy???*/
                int feature=0; 
                for(int i=0;i<intPermutation.length;i++)
                    if(feat+1==intPermutation[i])
                        feature=i;
                //feature=feat;
                
                //tega sploh ne potrebujemo, potrebovali bi če bi razlagali vse atribute
                int featureIndex = intPermutation[feature]-1;   // ker imamo permutacije npr. od 1 do 6, moramo odšteti 1, ker imamo indekse od 0...5
                //System.out.println("Kontrola indeksa atributa "+featureIndex);
                
/*System.out.println("Original");
System.out.println("        "+instance.instance(0));
System.out.println("feat "+feat+" featureIndex: "+featureIndex+" feature "+feature);*/
                
                for (int j = D.numAttributes() - 2; j > feature; j--){
                    double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[j]-1); // vrednost danega atributa v naključno izbranem primeru
                    instance.instance(0).setValue(intPermutation[j]-1, value);
                    
                }

/*System.out.println("First");
System.out.println("        "+instance.instance(0));
System.out.println("**********************");*/

                double predictionHi = 0;
                if (isClassification)
                    predictionHi = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionHi = K.classifyInstance(instance.instance(0));	              
//                }
                //System.out.println("Second");
                //System.out.println("        featureIndex: "+featureIndex);
              	// pri drugem naključno izberemo vrednosti atributov, ki so strogo desno od trenutnega atributa; preostale vrednosti vzamemo iz primera, ki ga razlagamo
                //instance = new Instances(I1,0, 1);
//double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[feature]-1); // vrednost danega atributa v naključno izbranem primeru
//instance.instance(0).setValue(feat, instanceOrig.instance(0).value(feat));  //b1 i-th feature is the same as feature in instance that we explain        
                

                double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[feature]-1); // vrednost danega atributa v naključno izbranem primeru
                instance.instance(0).setValue(intPermutation[feature]-1, value);
                
                
/*System.out.println("Second");
System.out.println("        "+instance.instance(0));
System.out.println("**********************");
                System.exit(0); */               
                

                double predictionLo = 0;
                if (isClassification)
                    predictionLo = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionLo = K.classifyInstance(instance.instance(0));
	
                
                double finalC = +predictionHi-predictionLo;
                //System.out.println(finalC);
                //System.exit(0);

            return finalC;
    }
    
    public static double explainInstanceTest2(Classifier K, Instances D, Instances I1, int feat, boolean isClassification, int classValueToExplain) throws Exception{
	int numOfFeats = D.numAttributes() - 1;
        Random internalRand = new Random(0);
        double[] results = new double[numOfFeats];
        //double result =0;
     
        Permutations permuts = new Permutations(numOfFeats);
        
            Permutation tempPermutation = permuts.randomPermutation();
            int[] intPermutation = tempPermutation.toIntArray();
           
           /*System.out.print("permutacije indeksov atributov: ");
                    XplainAttrConstr.print1dInt(intPermutation);
           System.out.println();*/
            for (int feature = 0; feature < numOfFeats; feature++){
                
                Instances instance = new Instances(I1,0, 1);
                int featureIndex = intPermutation[feature]-1;   // ker imamo permutacije npr. od 1 do 6, moramo odšteti 1, ker imamo indekse od 0...5
                //if(feature==feat){
                for (int j = D.numAttributes() - 2; j > feature; j--){
                    //System.out.println("feature "+feature+" attr. index "+j);
                    double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[j]-1); // vrednost danega atributa v naključno izbranem primeru
                    instance.instance(0).setValue(intPermutation[j]-1, value);                 
                }
                /*System.out.println("First");
              System.out.println("        "+instance.instance(0));*/
                double predictionHi = 0;
                if (isClassification)
                    predictionHi = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionHi = K.classifyInstance(instance.instance(0));	              
    
             
                double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[feature]-1); // vrednost danega atributa v naključno izbranem primeru
                instance.instance(0).setValue(intPermutation[feature]-1, value);
             	/*		                System.out.println("Second");
              System.out.println("        "+instance.instance(0));*/
                
                double predictionLo = 0;
                if (isClassification)
                    predictionLo = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionLo = K.classifyInstance(instance.instance(0));
	
                results[featureIndex] = predictionHi-predictionLo;
               // }
            }
            return results[feat];
    }
    
    
    
    public static double[][] explainInstanceExact(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain) throws Exception
    {
		/*Discretize    filter;
		filter = new Discretize();
		filter.setInputFormat(D);
		filter.setOptions(weka.core.Utils.splitOptions("-B 50 -R first-last"));    
		D = Filter.useFilter(D, filter);*/

		//System.out.println(D.toSummaryString());
		//System.exit(0)
		int N_bins = 10;
		
		int numOfFeats = D.numAttributes() - 1;
		int N = numOfFeats;
        int[] bitCounter = new int[numOfFeats];
        int nComb = (int)Math.pow(2, numOfFeats);
        int[] N_values = new int[numOfFeats];
        int N_combinations = 1;
        for (int i = 0; i < numOfFeats; i++)
        {		           
            if (!D.attribute(i).isNominal())
            	N_values[i] = N_bins;
            else
            	N_values[i] = D.attribute(i).numValues();	
            N_combinations *= N_values[i];
        }
        //double[][] result = new double[N][I.numInstances()];
        double[][] result = new double[I.numInstances()][N];//changed by bv
        for (int i = 0; i < nComb; i++)
        {
                int N_remaining = 0;
                for (int j = 0; j < N; j++) N_remaining += bitCounter[j];

                long combFactor1 = choose(N-1,N-N_remaining-1);
                long combFactor2 = choose(N-1,N-N_remaining);

                for (int j = 0; j < I.numInstances(); j++)
                {

                    //
                    int[] combCounter = new int[N];
                    // go over all combinations of unknown values and get mean prediction[classIndex]
                    double prediction = 0;

                    for (int i1 = 0; i1 < N_combinations; i1++)
                    {
                        Instances tempTest = new Instances(I, j, 1);

                        for (int i2 = 0; i2 < N; i2++)
                        {
                           if (bitCounter[i2] != 1)
                           {
                        	    if (D.attribute(i2).isNumeric())
                        	    {
                        	    	double step = 1.0 / (N_bins - 1);
                        	    	tempTest.instance(0).setValue(i2, combCounter[i2] * step);              	    	
                        	    }
                        	    else
                        	    {
                        	    	//System.out.println(D.attribute(i2).value(combCounter[i2]));
                        	    	tempTest.instance(0).setValue(i2, D.attribute(i2).value(combCounter[i2]));
                        	    }
                           }
                            
                        }
                        
            			if (isClassification)
            			      prediction += K.distributionForInstance(tempTest.instance(0))[classValueToExplain];
            			else
            				prediction = K.classifyInstance(tempTest.instance(0));
                     
                         combCounter = increaseCounter(combCounter, N_values);
                    }

                    prediction /= N_combinations;

                    for (int k = 0; k < N; k++)
                    {
                        if (bitCounter[k] == 1)
                            result[k][k] += (prediction / (N * combFactor2));
                        else
                            result[k][k] -= (prediction / (N * combFactor1));
                    }
                }

            bitCounter = increaseCounter(bitCounter);
        }        

        return result;
    }
    
    
    
    public static double[] explainValue(Classifier K, Instances D, int attrIdx, double valIdx, int m, boolean isNominal, double[][] extremeValues, boolean isClassification){
	// algoritem za računanje splošnega prispevka neke vrednosti nekega atributa (Algoritem 2 iz članka, prilagojen, da dela na vseh kombinacijah zveznih/nominalnih atributov in klasifikacije/regresije)
	// INPUT: naučen model, učna množica, indeks atributa, indeks vrednosti/ vrednost [odvisno od tega, ali gre za klasifikacijo ali regresijo], število vzorcev m, je atribut nominalen?, minimalne in maksimalne vrednosti atributov, <tega ne uporabljamo>, je klasifikacijski problem?
	// OUTPUT: prispevek vrednosti (psi) in std. dev. prispevka
	
        double[] res = new double[2];
	double[] psi = new double[m];	//default vrednost je nastavljena na 1000
		//double[] psi = new double[m*D.numInstances()];	//default vrednost je nastavljena na 1000
	try{	
            for (int i = 0; i < m; i++){ // ponovimo m-krat, enkrat za vsak vzorec
                //default
		Instances instance1 = new Instances(D,0,1); 
		Instances instance2 = new Instances(D,0,1);
			
		//for(int k=0;k<D.numInstances();k++)
					
		for (int j = 0; j < D.numAttributes()-1; j++){			
                    // naključno nastavimo vrednosti atributov (pri obeh primerih enako), 
                    // le pri drugem primeru nastavimo vrednost atributa, ki ga razlagamo, na vrednost, ki nas zanima
                    if (D.attribute(j).isNominal()){ 
                        String value = D.attribute(j).value(rand.nextInt(D.attribute(j).numValues()));
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
                            if (j == attrIdx) 
                                instance2.instance(0).setValue(j, D.attribute(j).value((int)valIdx));
                    }
                    else{
                    // zato potrebujemo min/max, da vemo na katerem intervalu izbirati naključne vrednosti
                    //extremeValues - 2D tabela v kateri sta min in max vrednost atributa ([j][0]-min, [j][1]-max)
                        double value = rand.nextFloat() * (extremeValues[j][1] - extremeValues[j][0]) + extremeValues[j][0]; //rand * (max-min) + min
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
                        //System.out.println("Attr. "+j+" value "+value+" valIdx "+valIdx);
			//vrednost atributa nastavimo na vrednost ki nas zanim
			if (j == attrIdx)
                            instance2.instance(0).setValue(j, valIdx);					
                    }
		}
		double p2 = -1;
		double p1 = -1;
	
		if (isClassification){ // pri klasifikaciji gledamo verjetnostno napoved za željeni razred, pri regresiji samo napoved
                    p2 = K.distributionForInstance(instance2.instance(0))[CLASS_VALUE_TO_EXPLAIN];
                    p1 = K.distributionForInstance(instance1.instance(0))[CLASS_VALUE_TO_EXPLAIN];
		}
		else{
                    p2 = K.classifyInstance(instance2.instance(0));
                    p1 = K.classifyInstance(instance1.instance(0));	
		}
                psi[i] = p2 - p1; // razlika med napovedjo z vrednostjo in "brez" vrednosti
                /*System.out.println("p1 "+p1+"p2 "+p2);
                System.out.println(instance1.instance(0));
                System.out.println(instance2.instance(0));*/

                /*(printPsi)
                    psiFile.print(rnd(psi[i])+",");*/
                    //System.out.print(psi[i]+",");
            }
            
            res[0] = mean(psi);		//prispevek vrednosti (psi), mean je metoda, ki izračuna povprečje v 1D tabeli
            res[1] = Math.sqrt(var(psi,mean(psi)));		// std. dev. prispevka 
                /*if(printPsi){
                    print1d(psi);  // izpis vseh prispevkov v tabeli
                System.out.println();
                System.out.println("mean psi"+res[0]);
                System.out.println("stdev psi"+res[1]);
                }*/        
            }
            catch(Exception e){
                    //e.printStackTrace();
            }
            return res;
    }
    //spremenjena metoda na podlagi članka  Explaining prediction models and individual predictions with feature contributions. Formula 14 in razlage Štrumblja (mail)
    public static double[] explainValueAttrImp(Classifier K, Instances D, int attrIdx, double valIdx, int m, boolean isNominal, double[][] extremeValues, boolean isClassification){
	// algoritem za računanje splošnega prispevka neke vrednosti nekega atributa (Algoritem 2 iz članka, prilagojen, da dela na vseh kombinacijah zveznih/nominalnih atributov in klasifikacije/regresije)
	// INPUT: naučen model, učna množica, indeks atributa, indeks vrednosti/ vrednost [odvisno od tega, ali gre za klasifikacijo ali regresijo], število vzorcev m, je atribut nominalen?, minimalne in maksimalne vrednosti atributov, <tega ne uporabljamo>, je klasifikacijski problem?
	// OUTPUT: prispevek vrednosti (psi) in std. dev. prispevka
	
        double[] res = new double[2];
	double[] psi = new double[m];	//default vrednost je nastavljena na 1000
		//double[] psi = new double[m*D.numInstances()];	//default vrednost je nastavljena na 1000
	try{	
            for (int i = 0; i < m; i++){ // ponovimo m-krat, enkrat za vsak vzorec
                //default
		Instances instance1 = new Instances(D,0,1); 
		Instances instance2 = new Instances(D,0,1);
			
		//for(int k=0;k<D.numInstances();k++)
					
		for (int j = 0; j < D.numAttributes()-1; j++){			
                    // naključno nastavimo vrednosti atributov (pri obeh primerih enako), 
                    // le pri drugem primeru nastavimo vrednost atributa, ki ga razlagamo, na vrednost, ki nas zanima
                    if (D.attribute(j).isNominal()){ 
                        String value = D.attribute(j).value(rand.nextInt(D.attribute(j).numValues()));
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
                            if (j == attrIdx) 
                                instance2.instance(0).setValue(j, D.attribute(j).value((int)valIdx));
                    }
                    else{
                    // zato potrebujemo min/max, da vemo na katerem intervalu izbirati naključne vrednosti
                    //extremeValues - 2D tabela v kateri sta min in max vrednost atributa ([j][0]-min, [j][1]-max)
                        double value = rand.nextFloat() * (extremeValues[j][1] - extremeValues[j][0]) + extremeValues[j][0]; //rand * (max-min) + min
			instance1.instance(0).setValue(j, value);
			instance2.instance(0).setValue(j, value);
                        //System.out.println("Attr. "+j+" value "+value+" valIdx "+valIdx);
			//vrednost atributa nastavimo na vrednost ki nas zanima
			if (j == attrIdx)
                            instance2.instance(0).setValue(j, valIdx);					
                    }
		}
		double p2 = -1;
		double p1 = -1;
	
		if (isClassification){ // pri klasifikaciji gledamo verjetnostno napoved za željeni razred, pri regresiji samo napoved
                    p2 = K.distributionForInstance(instance2.instance(0))[CLASS_VALUE_TO_EXPLAIN];
                    p1 = K.distributionForInstance(instance1.instance(0))[CLASS_VALUE_TO_EXPLAIN];
		}
		else{
                    p2 = K.classifyInstance(instance2.instance(0));
                    p1 = K.classifyInstance(instance1.instance(0));	
		}
                psi[i] = p2 - p1; // razlika med napovedjo z vrednostjo in "brez" vrednosti
                //System.out.println("psi[i]: "+psi[i]);
                /*System.out.println("p1 "+p1+"p2 "+p2);
                System.out.println(instance1.instance(0));
                System.out.println(instance2.instance(0));*/

                /*(printPsi)
                    psiFile.print(rnd(psi[i])+",");*/
                    //System.out.print(psi[i]+",");
            }
                
            //res[0] = mean(psi);		//prispevek vrednosti (psi), mean je metoda, ki izračuna povprečje v 1D tabeli
            //res[1] = Math.sqrt(var(psi,mean(psi)));		// std. dev. prispevka 
                /*if(printPsi){
                    print1d(psi);  // izpis vseh prispevkov v tabeli
                System.out.println();
                System.out.println("mean psi"+res[0]);
                System.out.println("stdev psi"+res[1]);
                }*/
                
                //System.out.println("Number of samples: "+m);
            }
            catch(Exception e){
                    //e.printStackTrace();
            }
            return psi;
    }
    public static ArrayList<Double> attributeSelection(Classifier predictionModel, Instances data, int N_SAMPLES, boolean isClassification){
    double meanStdPsi=0, absWeight=0,tempValue;
    Instance tempInst;
    double[] res1;
    ArrayList<Double> allAttrPsi=new ArrayList<>();
    //double allAttrPsi[]=new double[data.numAttributes()-1]; //we don't need psi for target parameter
    Attribute tempAttr;

    // get range (min,max) for numeric attributes
    // gremo skozi vse učne primere in za vsak atribut določimo min in max vrednost
    		// find min and max for numeric attributes
    try{
    double[][] minMaxForNumericAttributes = new double[data.numAttributes()][2];
        for (int i = 0; i < data.numAttributes(); i++){
            minMaxForNumericAttributes[i][0] = Double.MAX_VALUE;	//to store min value of an attribute
            minMaxForNumericAttributes[i][1] = Double.MIN_VALUE;	//to store max value of an attribute
    }
     
    
    for (int i = 0; i < data.numInstances(); i++){
            tempInst = data.instance(i);
            for (int j = 0; j < data.numAttributes(); j++){
                tempAttr = data.attribute(j);
		if (tempAttr.isNumeric()){
                    if (tempInst.value(j) < minMaxForNumericAttributes[j][0]) 
                        minMaxForNumericAttributes[j][0] = tempInst.value(j);
                            if (tempInst.value(j) > minMaxForNumericAttributes[j][1]) 
                                minMaxForNumericAttributes[j][1] = tempInst.value(j);
			}
            }
	}    
        
  
    for (int i = 0; i < data.numAttributes() - 1; i++){
        tempAttr = data.attribute(i);
	
	// če gre za numerični atribut, izračunamo prispevke vrednosti med min in max vrednostjo 
	// (enakomerno razporejeno, število je odvisno od ločljivosti, ki jo želimo za vizualizacijo)
	if (tempAttr.isNumeric()){

	tempValue = minMaxForNumericAttributes[i][0] + Math.random()*(minMaxForNumericAttributes[i][1] - minMaxForNumericAttributes[i][0]);
	/*if(printPsi)
            psiFile.println("Psi for attribute "+data.attribute(i).name());*/
        res1 = explainValue(predictionModel, data, i, tempValue, N_SAMPLES, false, minMaxForNumericAttributes, isClassification);
        
        absWeight += Math.abs(res1[0]);
        //psiValue +=res1[0];
	meanStdPsi += res1[1];
        
        allAttrPsi.add(res1[1]);
        /*if(printPsi)
            System.out.println("Psi of attribute "+data.attribute(i).name()+" = "+rnd(res1[1]));*/
        //psiValue=0;
        meanStdPsi=0;
	absWeight=0;				
	}
    }
    
    }
    catch(Exception e){
        System.out.println("Napaka v metodi attributeSelection");
        e.printStackTrace();
    }
    
    return allAttrPsi;
    }
    

    
   
    
    public static double explainFeatInInstNoPermut(Classifier K, Instances D, Instances I1, int feature, boolean isClassification, int classToExplain) throws Exception{ //just one iteration, just one sample
	// algoritem za računanje prispevkov atributov v določenem primeru (Algoritem 1 iz članka, prilagojen, da dela na vseh kombinacijah zveznih/nominalnih atributov in klasifikacije/regresije)
	// INPUT: naučen model, učna množica, primer, število vzorcev m, je klasifikacijski problem?
	// OUTPUT: vektor prispevkov posameznih atributov za podani primer	
	
	int numOfFeats = D.numAttributes() - 1;
        double result=0;
            	Instances instance = new Instances(I1,0, 1);

//System.out.println("Feature idx "+feature);
                /*System.out.println("Original");
                System.out.println("        "+instance.instance(0));
                System.out.println("feature: "+feature+" featureIndex: "+featureIndex+" i: "+i);*/
                //B2 IN ARTICLE
//System.out.println("B2 orig. instance x "+instance.instance(0));
                for (int j = D.numAttributes() - 2; j >= feature; j--){ //D.numAttributes() returns num of attributes + class, that's why -2
                    int rndInst=rand.nextInt(D.numInstances());

//System.out.println("B2 Random instance w "+D.instance(rndInst));
                	double value = D.instance(rndInst).value(j);// double value = D.instance(rndInst).value(intPermutation[j]-1); // vrednost danega atributa v naključno izbranem primeru
                	instance.instance(0).setValue(j, value);//instance.instance(0).setValue(intPermutation[j]-1, value);
//System.out.println("B2 Changed instance x "+instance.instance(0));
                        
                        //System.out.println("        "+instance.instance(0)+"    attr. index: "+(intPermutation[j]-1)+" rnd instance: "+rndInst+" j: "+j+" value: "+value);
                }
                /*System.out.println("First");
              System.out.println("        "+instance.instance(0));
              System.out.println("**********************");*/
                double predictionLo = 0;                  
    			if (isClassification)
    				predictionLo = K.distributionForInstance(instance.instance(0))[classToExplain];//predictionLo = K.distributionForInstance(instance.instance(0))[CLASS_VALUE_TO_EXPLAIN];
    			else
    				predictionLo = K.classifyInstance(instance.instance(0));		              
                
                //System.out.println("Second");
                //System.out.println("        featureIndex: "+featureIndex);
              	// pri drugem naključno izberemo vrednosti atributov, ki so strogo desno od trenutnega atributa; preostale vrednosti vzamemo iz primera, ki ga razlagamo
                instance = new Instances(I1,0, 1);
//System.out.println("B1 orig. instance x "+instance.instance(0));
                for (int j = D.numAttributes() - 2; j > feature; j--){  //B1 IN ARTICLE
                    int rndInst=rand.nextInt(D.numInstances());

//System.out.println("B1 Random instance w "+D.instance(rndInst));
                    //System.out.println("instance x"+instance.instance(0).index(0)+" random instance w "+rndInst);
                	double value = D.instance(rndInst).value(j); // double value = D.instance(rndInst).value(intPermutation[j]-1);
                	instance.instance(0).setValue(j, value);// instance.instance(0).setValue(intPermutation[j]-1, value);
//System.out.println("B1 Changed instance x "+instance.instance(0));
                        //System.out.println("        "+instance.instance(0)+"    attr. index: "+(intPermutation[j]-1)+" rnd instance: "+rndInst+" j: "+j+" value: "+value);    
                }
                 /*System.out.println("Second");
              System.out.println("        "+instance.instance(0));
              System.out.println("**********************");*/
                double predictionHi = 0;    
    			if (isClassification)
    				predictionHi = K.distributionForInstance(instance.instance(0))[classToExplain];//predictionHi = K.distributionForInstance(instance.instance(0))[CLASS_VALUE_TO_EXPLAIN];
    			else
    				predictionHi = K.classifyInstance(instance.instance(0));

            result+= predictionHi-predictionLo;

		
            return result;
    }
    
    public static double explainFeatInInstNew(Classifier K, Instances D, Instances I1, int feature, boolean isClassification, long randomSeed, int classValueToExplain, boolean equiprobableContext) throws Exception{
        int numOfFeats = D.numAttributes() - 1;
        Random internalRand = new Random(randomSeed);
        // get extremes for nominal
        double[][] extremeValues = new double[D.numAttributes()][2];
        for (int i = 0; i < D.numAttributes(); i++)
        {
            extremeValues[i][0] = Double.MAX_VALUE;
            extremeValues[i][1] = Double.MIN_VALUE;
        }

        // get range for numeric attributes
        for (int i = 0; i < D.numInstances(); i++)
        {
            Instance tempInst = D.instance(i);
            //System.out.println(predictionModel.distributionForInstance(tempInst)[0]);
            for (int j = 0; j < D.numAttributes(); j++)
            {
                Attribute tempAttr = D.attribute(j);
                if (tempAttr.isNumeric())
                {
                    if (tempInst.value(j) < extremeValues[j][0]) extremeValues[j][0] = tempInst.value(j);
                    if (tempInst.value(j) > extremeValues[j][1]) extremeValues[j][1] = tempInst.value(j);
                }
            }
        }


         double result = 0;
        

        Permutations permuts = new Permutations(numOfFeats);


            Permutation tempPermutation = permuts.randomPermutation();
            int[] intPermutation = tempPermutation.toIntArray();


                Instances instance = new Instances(I1,0, 1);
                int featureIndex = intPermutation[feature]-1;

                for (int j = D.numAttributes() - 2; j > feature; j--)
                {
                    
                        double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[j]-1); // vrednost danega atributa v naključno izbranem primeru
                        instance.instance(0).setValue(intPermutation[j]-1, value);
                    //}
                }

                double predictionHi = 0;
                if (isClassification)
                    predictionHi = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionHi = K.classifyInstance(instance.instance(0));
                
                
                if (equiprobableContext)
                {
                    if (D.attribute(intPermutation[feature]-1).isNominal())
                    {
                        int numValues = D.attribute(intPermutation[feature]-1).numValues();
                        instance.instance(0).setValue(intPermutation[feature]-1, D.attribute(intPermutation[feature]-1).value(internalRand.nextInt(numValues)));
                    }
                    else
                    {
                        double value = internalRand.nextFloat() * (extremeValues[feature][1] - extremeValues[feature][0]) + extremeValues[feature][0];
                        instance.instance(0).setValue(intPermutation[feature]-1, value);
                    }
                }
                else
                {
                    double value = D.instance(internalRand.nextInt(D.numInstances())).value(intPermutation[feature]-1); // vrednost danega atributa v naključno izbranem primeru
                    instance.instance(0).setValue(intPermutation[feature]-1, value);
                }

                double predictionLo = 0;
                if (isClassification)
                    predictionLo = K.distributionForInstance(instance.instance(0))[classValueToExplain];
                else
                    predictionLo = K.classifyInstance(instance.instance(0));

                //double finalC = +predictionHi-predictionLo;
                

                result= predictionHi-predictionLo;


        return result;
    }
    
    public static double[] explainInstanceOpt(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain, int mMin, int mMax) throws Exception{
        int m[]=new int[D.numAttributes()-1];  //set all values to 0 ... first number of samples for each attribute are 0
        double psi[]=new double [D.numAttributes()-1]; //set all values to 0 ... first all contributions are 0 
        double tmpVariance[]=new double [D.numAttributes()-1]; //set all values to 0 ... first all contributions are 0
        double tmpMean[]=new double [D.numAttributes()-1]; //set all values to 0 ... first all contributions are 0
        Vector v[]= new Vector[D.numAttributes()-1];
        double maxDiff=0;
        double tmpDiffVar;
        int j=0;
        
        for(int i=0;i<D.numAttributes()-1;i++)
            v[i]=new Vector();

        for(int i = 0 ; i < psi.length; i++ ){
            psi[i] = 0;
        }
        
        while(sumArr(m) < mMax){       
                if(allHigher(m,mMin)){ //get max diff of variance
                    maxDiff=0;
                    for(int i=0;i<tmpVariance.length;i++){
                        tmpDiffVar=(Math.sqrt(tmpVariance[i]/m[i])) - Math.sqrt(tmpVariance[i]/(m[i]+1));
                        //tmpDiffVar=((tmpVariance[i]/m[i])) - (tmpVariance[i]/(m[i]+1));
                        if(tmpDiffVar>maxDiff){
                            maxDiff=tmpDiffVar;
                            j=i;
                        }
                    }
                }
                else{
                    j=getIdx(m,mMin);
                }

                double tmpPsi;
                //tmpPsi=explainFeatInInstNoPermut(K, D, I, j, isClassification,classValueToExplain);
                tmpPsi=explainInstanceTest(K, D, I, j, isClassification,classValueToExplain);

                v[j].add(tmpPsi);
                tmpMean[j]=mean(v[j]);
                psi[j] += tmpPsi;
                tmpVariance[j]=incrementalVar(v[j].size(), tmpVariance[j], tmpMean[j], tmpPsi)[3]; //return new double[]{k,M2,mean,var};*/
                m[j]++;

                //if(allVarBelowThr(tmpVariance, varThr) && allHigher(m,mMin)){
                                /*print1d(tmpVariance);
                                System.out.println();
                                XplainAttrConstr.print1dInt(m);
                System.out.println();*/
                //    break;
                //}
//                print1d(tmpVariance);
//                                System.out.println();
        }
        
        for(int i=0;i<psi.length;i++)
            psi[i]=psi[i]/m[i];
        
        
        return psi;
    }
    
    public static double[] explainInstanceTestVar(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain, int mMax) throws Exception{
        int m[]=new int[D.numAttributes()-1];  //set all values to 0 ... first number of samples for each attribute are 0
        double psi[]=new double [D.numAttributes()-1]; //set all values to 0 ... first all contributions are 0 
        double tmpVariance[]=new double [D.numAttributes()-1]; //set all values to 0 ... first all contributions are 0
        double tmpMean[]=new double [D.numAttributes()-1]; //set all values to 0 ... first all contributions are 0
        Vector v[]= new Vector[D.numAttributes()-1];
        
        for(int i=0;i<D.numAttributes()-1;i++)
            v[i]=new Vector();

        for(int i = 0 ; i < psi.length; i++ ){
            psi[i] = 0;
        }
        for(int i=0;i<D.numAttributes()-1;i++){
            for(int j=0;j<mMax;j++){       
                double tmpPsi;
                //tmpPsi=explainFeatInInstNoPermut(K, D, I, j, isClassification,classValueToExplain);
                tmpPsi=explainInstanceTest(K, D, I, i, isClassification,classValueToExplain);

                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3]; //return new double[]{k,M2,mean,var};*/
                //m[j]++;
            }
        }
        
        
                //if(allVarBelowThr(tmpVariance, varThr) && allHigher(m,mMin)){
                                /*print1d(tmpVariance);
                                System.out.println();
                                XplainAttrConstr.print1dInt(m);
                System.out.println();*/
                //    break;
                //}
//                print1d(tmpVariance);
//                                System.out.println();
        
        //changed "variance" \sqrt{ \sigma_i^2 / m_i)}
        for(int i=0;i<D.numAttributes()-1;i++)
            tmpVariance[i]=Math.sqrt(tmpVariance[i]/mMax);
        
//        print1d(tmpVariance);
//        System.out.println();
//        
        return psi;
    }
    
    public static double[] explainInstBasedOnVar(Classifier K, Instances D, Instances I, boolean isClassification, int classValueToExplain, int mMin, int mMax, double e, int pctErr) throws Exception{
        int numOfAttr=D.numAttributes()-1;
        int m[]=new int[numOfAttr];  //set all values to 0 ... first number of samples for each attribute are 0
        double psi[]=new double [numOfAttr]; //set all values to 0 ... first all contributions are 0 
        double tmpVariance[]=new double [numOfAttr]; //set all values to 0 ... first all contributions are 0
        double tmpMean[]=new double [numOfAttr]; //set all values to 0 ... first all contributions are 0
        Vector v[]= new Vector[numOfAttr];
        //int j=0;
        
        for(int i=0;i<D.numAttributes()-1;i++)
            v[i]=new Vector();

        for(int i = 0 ; i < psi.length; i++ ){
            psi[i] = 0;
        }
        
        for(int i=0;i<numOfAttr;i++)
            m[i]=mMin;
        
        double tmpPsi;
        //get initial variances for each feature
        for(int i=0;i<numOfAttr;i++){
            for(int j=0;j<mMin;j++){
                tmpPsi=explainInstanceTest(K, D, I, i, isClassification,classValueToExplain);
                v[i].add(tmpPsi);
                tmpMean[i]=mean(v[i]);
                psi[i] += tmpPsi;
                tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3]; //return new double[]{k,M2,mean,var};*/
                //m[i]++;
        }
        }

    double z;
    switch(pctErr){
        case 90:z=1.285;break;    //for 90% of probability
        case 95:z=1.645;break;    //for 95% of probability
        case 99:z=2.325;break;    //for 99% of probability
        default: z=1.285;   //we set default Z for 90%
    }
 
//    for(int i=0; i<numOfAttr;i++)
//        numOfSamples[i]=(int)((Math.pow(z, 2)*tmpVariance[i])/Math.pow(e, 2));
//System.out.println(z);
//System.out.println(Math.pow(z, 2));
//System.out.println(tmpVariance[3]);
//System.out.println(e);
//System.out.println(Math.pow(e, 2));
//
//System.out.println((int)((Math.pow(z, 2)*tmpVariance[3])/Math.pow(e, 2)));
//System.exit(0);
double zSquared=Math.pow(z, 2);
double eSquared=Math.pow(e, 2);

    for(int i=0;i<numOfAttr;i++){
        while(m[i]<Math.round((zSquared*tmpVariance[i])/eSquared) && m[i]<mMax){
            tmpPsi=explainInstanceTest(K, D, I, i, isClassification,classValueToExplain);
            v[i].add(tmpPsi);
            tmpMean[i]=mean(v[i]);
            psi[i] += tmpPsi;
            tmpVariance[i]=incrementalVar(v[i].size(), tmpVariance[i], tmpMean[i], tmpPsi)[3]; //return new double[]{k,M2,mean,var};*/
            m[i]++;
        }
    }
    
//        print1d(m);
//            System.out.println();
//                                                                                                                                                                    for(int i=0;i<m.length;i++){
//                                                                                                                                                                        FeatConstr.samplesStat.print(m[i]+"\t");
//                                                                                                                                                                    }
//                                                                                                                                                                    FeatConstr.samplesStat.println("max: "+Arrays.stream(m).max().getAsInt());

        for(int i=0;i<psi.length;i++)
            psi[i]=psi[i]/m[i];
        
        
        return psi;
    }
    
    
    public static double[] explainInstanceMaxInst(Classifier K, Instances D, Instances I, int m[], boolean isClassification, int classValueToExplain) throws Exception{
        double psi[]=new double [D.numAttributes()-1]; //set all values to 0 ... first all contributions are 0 
        for(int i=0; i<m.length;i++){   //length of m is the same as number of attributes
            if(m[i]==0)
                continue;
            for(int j=0;j<m[i];j++){
                psi[i]+=explainInstanceTest(K, D, I, i, isClassification,classValueToExplain);
            }
	}
				
	for(int i=0;i<psi.length;i++){
            if(m[i]==0)
                continue;
            psi[i]=psi[i]/m[i];
	}
				
        return psi;
    }
    
    
    
    
    
    
//                                                                                                                                                                        public static void print2dToFile(double tab [][], File fileName) throws FileNotFoundException{
//                                                                                                                                                                           //PrintWriter pw=new PrintWriter(fileName);	
//                                                                                                                                                                           PrintWriter pw = new PrintWriter(new FileOutputStream(fileName, true)); 
//                                                                                                                                                                           //pw.println("Instance 1");
//                                                                                                                                                                               for(int i=0;i<tab.length;i++){
//                                                                                                                                                                                   for (int j=0;j<tab[i].length;j++){
//                                                                                                                                                                                       pw.print(FeatConstr.rnd2(Math.abs(tab[i][j]))+"\t");
//                                                                                                                                                                                       //System.out.print(tab[i][j]+" ");
//                                                                                                                                                                                   }
//                                                                                                                                                                               pw.println();
//                                                                                                                                                                               //pw.println("Instance "+(i+2));
//                                                                                                                                                                               //System.out.println();
//                                                                                                                                                                           }
//                                                                                                                                                                           pw.close();
//                                                                                                                                                                       }
    
    public static void print2d(double tab [][]) throws FileNotFoundException{
            for(int i=0;i<tab.length;i++){
                for (int j=0;j<tab[i].length;j++){
                    System.out.print(tab[i][j]+" ");
		}
            System.out.println();
	}
    }
    
    
    
    private static long choose( int n, int k){
    // return combinations
            return fact(n) / (fact(k) * fact(n-k));
    }
    private static long fact( int n ){
    // return factorial
        if( n <= 1 )     // base case
            return 1;
        else
            return n * fact( n - 1 );
    }
    private static int[] increaseCounter(int[] counter, int[] maxValue){
    // increase combination counter by one
        counter[0]++;
        for (int i = 0; i < counter.length-1; i++)
        {
            if (counter[i] >= maxValue[i])
            {
                counter[i]=0;
                counter[i+1]++;
            }
        }
        return counter;
    }
    
    private static int[] increaseCounter(int[] counter){
    // increase bit counter by one
        counter[0]++;    
        for (int i = 0; i < counter.length-1; i++)
        {
            if (counter[i] == 2)
            {
                counter[i]=0;
                counter[i+1]++;              
            }
        }
        return counter;
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
        //double var = M2 / k;
            
	//use k instead of (k-1) if want to compute the exact variance of the given data
	//use (k-1) if data are samples of a larger population
	//System.out.println(k + "\t" + M2 + "\t" + mean + "\t" + sample);
	return new double[]{k,M2,mean,var};	
    }
    
    public static int sumArr(int tab []){     //variance
        return Arrays.stream(tab).sum();
    }
    
    static double var(Vector d){    //variance in vector
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
    
    
    static double var(double[] d){
    double m1= 0;
        for (int i = 0; i < d.length; i++) {
            m1 += d[i];
	}
	m1 /= d.length;


	double sum = 0;
	for (int i = 0; i < d.length; i++) sum += (d[i] - m1) * (d[i]- m1);
    return sum / d.length;
    }	
	
	
    static double var(double[] d, double m){
    double sum = 0;
        for (int i = 0; i < d.length; i++) sum += (d[i] - m) * (d[i] - m);
    return sum / d.length;
    }
	
    static double var(double[] d, double m, int max){
    double sum = 0;
	for (int i = 0; i < max; i++) sum += (d[i] - m) * (d[i] - m);
    return sum /max;
    }
    
    static double mean(Vector d){ //mean in Vector
        double sum = 0;
	for (int i = 0; i < d.size(); i++) 
            sum += (Double)d.elementAt(i);
        return sum / d.size();
    }
    
    static double mean(double[] d){
		double sum = 0;
		for (int i = 0; i < d.length; i++) sum += d[i];
		return sum / d.length;
    }
    
    static boolean allHigher(int samples[], int mMin){ //mean in Vector
        for (int i=0;i<samples.length;i++)
            if(samples[i] < mMin)
                return false;
        return true;    
    }
    
    static boolean allVarBelowThr(double tabVar[], double thr){ //mean in Vector
        for (int i=0;i<tabVar.length;i++)
            if(tabVar[i] > thr)
                return false;
        return true;    
    }
    
    
    static int getIdx(int m[],int mMin){
        for (int i=0;i<m.length;i++)
            if(m[i] < mMin)
                return i;
        
        return -1;
    }
//                                                                                                                                                                                                            public static void print1d(double tab []){
//                                                                                                                                                                                                                for(int i=0;i<tab.length;i++){
//                                                                                                                                                                                                                    System.out.print(FeatConstr.rnd4(tab[i])+"\t");
//                                                                                                                                                                                                                }
//                                                                                                                                                                                                            }
    public static void print1d(int tab []){
        for(int i=0;i<tab.length;i++){
            System.out.print(tab[i]+"\t");
        }
    }
}
