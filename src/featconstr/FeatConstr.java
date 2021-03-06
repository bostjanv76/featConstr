package featconstr;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import weka.core.Instances;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import weka.core.*;
import weka.classifiers.*;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import java.util.Map;
import java.util.Random;
import weka.filters.unsupervised.attribute.Add;
import java.util.Collections; 
import java.util.HashSet;
import org.apache.commons.lang3.ArrayUtils; //commons-lang3-3.11.jar
import weka.classifiers.trees.J48;
import java.util.stream.Collectors;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import java.io.IOException;
import java.awt.Image;
import java.awt.image.RenderedImage;
import java.io.OutputStream;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.Set;
import java.util.TreeMap;
import javax.imageio.ImageIO;
import org.ghost4j.converter.PDFConverter;      //ghost4j-1.0.1.jar ... for converting svg to pdf
import org.ghost4j.document.PDFDocument;
import org.ghost4j.document.PSDocument;
import org.ghost4j.document.PaperSize;
import org.ghost4j.renderer.SimpleRenderer;
import org.paukov.combinatorics3.Generator;
import com.github.rcaller.rstuff.RCaller;   //RCaller-3.0.2.jar ... for calling MDL for discrete and contionous features
import com.github.rcaller.rstuff.RCode;
import java.net.URL;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.rules.Rule;
import weka.core.converters.ConverterUtils.DataSink;
import weka.classifiers.rules.FURIA;    //fuzzyUnorderedRuleInduction.jar
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.instance.RemoveRange;
import weka.filters.unsupervised.instance.RemoveWithValues;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.java.Booster;
import java.util.HashMap;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 *
 * @author bostjan
 */
@SuppressWarnings({"rawtypes", "unchecked", "serial"})
public class FeatConstr {
    /**************************** main EFC parameters *********************************/
    public static boolean justExplain=true;     //just explain datasets, construct features and evaluate them
    public static boolean visualisation=true;  //visualisation of explanations using IME method
    public static boolean exhaustive=true;     //try exhaustive search ... all combinations between attributes
    public static boolean jakulin=true;        //try exhaustive search, calculate interaction information between all comb. of attributes; Jakulin, A. (2005). Machine learning based on attribute interactions [Doctoral dissertation, University of Ljubljana]. ePrints.FRI. https://bit.ly/3eiJ18x
    /*****************************************************************************/
    public static boolean groupsByThrStat=true;        //print statistics about groups (identified by EFC) by thresholds
    public static boolean writeAccByFoldsInFile=true;  //for analysing results of statistical tests
    public static boolean saveConstructs=true; //save generated features with attributes into new dataset ("dataset name"-origPlusRen1stLFeat-"time-date".arff)
    public static boolean renameGenFeat=true;  //rename generated features (e.g., F1, F2 ...), available only if saveConstructs=true, for potential generation of 2nd level features; input dataset for generating 2nd level feat must contain "origPlusRen1stLFeat" string
    /**************************** IME parameters *********************************/
    public enum IMEver{equalSampling, adaptiveSamplingSS, adaptiveSamplingAE, aproxErrSampling};
    //equalSampling - each attribute has same num. of samples, Algorithm 1 in in ??trumbelj, Erik, and Igor Kononenko. "An efficient explanation of individual classifications using game theory." The Journal of Machine Learning Research 11 (2010): 1-18.
    //adaptiveSampling - Algorithm 2 in ??trumbelj, Erik, and Igor Kononenko. "Explaining prediction models and individual predictions with feature contributions." Knowledge and information systems 41.3 (2014): 647-665.
    //adaptiveSamplingSS - stopping criteria is sum of samples
    //adaptiveSamplingAE - stopping criteria is approxamization error for all attributes
    //aproxErrSampling - we calculate samples for each attribute mi=(<1-alpha, e>) (article 2010) in ??trumbelj, Erik, and Igor Kononenko. "An efficient explanation of individual classifications using game theory." The Journal of Machine Learning Research 11 (2010): 1-18.
    public static IMEver method=IMEver.adaptiveSamplingSS; //selected IME method
    public static int N_SAMPLES=100;    //if we use equalSampling ... number of samples, we choose random value from interval min-max N_SAMPLE times  
    public static int minS=10;          //min samples ... if we use sumOfSamples and diffSampling ... to obtain an approximate estimate of the variance
    public static int sumOfSmp=2000;    //sum of samples ... if we use adaptive sampling ... sumOfSmp >= n*minS ... n is number of attributes
    public static int pctErr=95;        //90, 95 or 99;
    public static double error=0.01;
    /**************************** XGBoost parameters *********************************/
    public static int numOfRounds=100;          //XGBoost parameter - number of decision trees 
    public static int maxDepth=3;               //XGBoost parameter - size of decision trees
    public static double eta=0.3;               //XGBoost parameter - shrinkage
    public static double gamma=1;               //XGBoost parameter - gamma
    /**************************** visualisation parameters *********************************/
    public static int visFrom=1, visTo=10;     //visualize instances from visFrom to visTo
    public static int drawLimit=20;    //we draw (max.) 20 the most important attributes
    public static int topHigh=10;       //visualise features with highest contributions (instance explanation)
    public static int numOfImpt=6;      //visualise features with highest contributions ... 
    public static int RESOLUTION=100;   // density for model visualisation
    public static boolean pdfPng=true;          //besided eps, print also pdf and png
    /**************************** additional EFC parameters *********************************/
    public static boolean treeSHAP=true;
    public static boolean explAllData=false; 
    public static boolean explAllClasses=false;
    public static boolean numerFeat=true;  //generate numerical features   
    public static double attrImpThrs[]={0,0.25,0.5};    //{0,0.2,0.4,0.6,0.7};//{0,0.1,0.2,0.3,0.4,0.5}; ... used only in paramSearch method
    public static double thrL=0.1;          //weight threshold - lower 
    public static double thrU=0.8;          //weight threshold - upper
    public static double step=0.1;          //step for traversing all thresholds from thrL to thrU
    public static double NOISE=1;           //(default 1) "lower" groups that have less than noiseThr% groups are removed ... noiseThr=0 (we take all groups); noiseThr=(numInst*NOISE)/100.0;
    public static int minNoise=3;           //(default 3) minimum number of groups at noiseThr
    public static int minMinNoise=1;        //(default 1) if numInst<minExplInst then minNoise=minMinNoise
    public static int minExplInst=50;       //if numInst<minExplInst then minNoise=minMinNoise
    public static int maxToExplain=500;     //(default 500) max instances to explain if we have more than 500 instances to explain from the class 
    public static int instThr=10;           //e.g. 10% ... we explain minority class, if minority class has at least percent of instThr instances
    /**************************** evaluation parameters *********************************/
    public static int folds=10;         //for generating models, folds=1 means no CV and using split in ratio listed below
    public static int splitTrain=5;     //5 ... 80%:20%, 4 ... 75%25%, 3 ... 66%:33%; useful only when folds are set to 1, meaning no CV and using split
    /**************************** FURIA parameters *********************************/
    public static double cf=0.5;                    //confidence factor (FURIA)
    public static double pci=0.9;                   //percentage of covered instances (FURIA)
    public static boolean covering=false;           //covering=true -> if all instances are covered by features generated by FURIA we stop construction with FURIA
    public static boolean featFromExplClass=true;   //for generate FURIA features ... for ablation study featFromExplClass=false this means that we take features from all classes
    /*****************************************************************************/
    public static int classToExplain=1;             //default is second class but this value is changed due to heuristic - explain minority class if class has at least instThr pct instances
    public static int timeLimit=10800000;           //10800000ms = 3h
    public static int numInst;                      //number of instances in explained class; is set when classToExplain is defined
    public static String datasetName;
    public static String tmpDir;
    public static List<String> listOfConcepts;
    public static String fileName;
    public static long modelBuildTime[];
    public static double accOrigModelByFolds[][];
    public static double accExplAlgInt[];       //for internal evaluation of the explanation alg
    public static double accExplAlgTest[];      //for evaluation of the explanation alg
    public static double oobRF[];               //for internal evaluation of the RF explanation alg - out of bag
    public static double accuracyByFolds[][];
    public static double accuracyByFoldsPS[][];
    public static double accuracyByFoldsFuriaThr[][];
    public static double accuracyByFoldsEvalAtt1 [][];
    public static double accuracyByFoldsEvalAtt2 [][];
    public static double accuracyByFoldsEvalAtt3 [][];
    public static double accByFoldsLF[][];              //for measuring accuracy for method Logical features
    public static double accByFoldsCP[][];              //for Cartesian product
    public static double accByFoldsRE[][];              //for relational features
    public static double accByFoldsNum[][];             //for numerical features
    public static double featByFoldsPS[][][];
    public static double numberOfFeatByFolds[][];       //0-logical, 1-threshold, 2-FURIA ... 5-numerical
    public static double numOfFeatByFoldsLF[];          //number of logical features per folds (for Logical features method)
    public static double numFeatByFoldsCP[];            //number of features from Cartesian product
    public static double numFeatByFoldsRE[];            //number of relational features
    public static double numFeatByFoldsNum[];           //number of numerical features
    public static double numFeatByFoldsEvalFeat1[][], numFeatByFoldsEvalFeat2[][], numFeatByFoldsEvalFeat3[][];
    public static double numberOfTreeByFoldsPS[][];     //0-tree size, 1-num of leaves, 2-sum of terms  
    public static double numOfTreeByFoldsLF[][];        //0-tree size, 1-num of leaves, 2-sum of terms (for Logical features method)
    public static double numOfTreeByFoldsCP[][];        //0-tree size, 1-num of leaves, 2-sum of terms (for Cartesian product)
    public static double numOfTreeByFoldsRE[][];        //0-tree size, 1-num of leaves, 2-sum of terms (for relational feat)
    public static double numOfTreeByFoldsNum[][];       //0-tree size, 1-num of leaves, 2-sum of terms (for numerical feat)
    public static double numCartFeatInTreeFS[][];       //0-number of Cartesian features in tree, 1 sum of constructs
    public static double numTreeByFoldsFuriaThr[][];
    public static double numberOfTreeByFolds[][];       //0-tree size, 1-num of leaves, 2-sum of terms 
    public static double numTreeByFoldsEvalFeat1[][],numTreeByFoldsEvalFeat2[][], numTreeByFoldsEvalFeat3[][];
    public static double numberOfUnImpFeatByFolds[][];  //0 - or, 1 - equ, 2 - xor, 3 - impl, 4 - and, 5 - lessthan, 6 - relational, 7 - Cartesian
    public static double numFeatByFoldsFuriaThr[][];
    public static long exlpTime[], allFCTime[], allFCTimeLF[], numericalFCTime[], cartesianFCTime[], relationalFCTime[], furiaThrTime[],attrEval1Time[], attrEval2Time[], attrEval3Time[];
    public static long learnAllFCTime[][], learnAllFCTimeLF[][], learnAllFCTimeNum[][], learnAllFCTimeCP[][], learnAllFCTimeRE[][], learnFuriaThrTime[][], learnAttrEval1Time[][], learnAttrEval2Time[][], learnAttrEval3Time[][];
    public static double numOfExplainedInst[];
    public static double numOfRulesByFolds[];
    public static double numOfCartesian[];          //number of Cartesian features in tree - when we generate just Cartesian
    public static double numOfRelational[];         //number of relational features in tree - when we generate just relational feat
    public static double numOfNumerical[];          //number of numerical features in tree - when we generate just numerical feat
    public static double numOfCartFAll[];           //number of Cartesian features in tree - when we generate all features
    public static double numOfRelInTreeAll[];       //number of relational features in tree - when we generate all features      
    public static double numOfNumInTreeAll[];       //number of numerical features in tree - when we generate all features    
    public static double sumOfConstrCart[];         //sum of constructs (from Cartesian) in tree - when we generate just Cartesian
    public static double sumOfConstrRel[];          //sum of constructs (from relational feat) in tree - when we generate just relatioan feat
    public static double sumOfConstrNum[];          //sum of constructs (from numerical feat) in tree - when we generate just numerical feat
    public static double sumOfConstrRelAll[];       //sum of constructs (from relational feat) in tree - when we generate all feat
    public static double sumOfConstrCartAll[];      //sum of constructs (from Cartesian) in tree - when we generate all features
    public static double numOfLogicalInTree[][];    //number of logical features in tree - when we generate just logical
    public static double numOfLogInTreeAll[];       //number of logical features in tree - when we generate all features
    public static double sumOfConstrLFAll[];
    public static double sumOfConstrNumAll[];       //sum of constructs (from numerical) in tree - when we generate all features
    public static double numLogFeatInTreeFS[][];    //number of logical features in trees (feature selection) and constructs
    public static double numNumFeatInTreeFS[][];    //number of numerical features in trees (feature selection) and constructs
    public static double numRelFeatInTreeFS[][];    //number of relational features in trees (feature selection) and constructs
    public static double numOfRulesByFoldsLF[];     //for method Logical features
    public static double numOfTermsByFoldsLF[];     //for method Logical features
    public static double numOfRatioByFoldsLF[];     //for method Logical features
    public static double numOfRulesByFoldsCP[];     //for Cartesian product
    public static double numOfTermsByFoldsCP[];     //for Cartesian product
    public static double numOfRatioByFoldsCP[];     //for Cartesian product
    public static double numOfRulesByFoldsRE[];     //for relational feat
    public static double numOfRulesByFoldsNum[];    //for numerical feat
    public static double numOfTermsByFoldsRE[];
    public static double numOfRatioByFoldsRE[];   
    public static double numOfTermsByFoldsNum[];
    public static double numOfRatioByFoldsNum[];  
      
    public static double numOfTermsByFoldsF[];      //number of terms of constructs in FURIA features    
    public static double numOfRatioByFoldsF[];
    public static double complexityOfFuriaPS[][];
    
    public static double numOfFuriaThrInTreeByFolds[][];    //0-num of FURIA feat, 1-sum of terms of FURIA feat, 2-num of thr feat, 3-sum of terms in thr feat
    public static double numOfFuriaThrInTreeByFoldsF[][];   //0-num of FURIA feat, 1-sum of terms of FURIA feat, 2-num of thr feat, 3-sum of terms in thr feat
    public static double numOfFuriaThrInTreeByFoldsM[][];   //0-num of FURIA feat, 1-sum of terms of FURIA feat, 2-num of thr feat, 3-sum of terms in thr feat
    public static double numOfFuriaThrInTreeByFoldsP[][];   //0-num of FURIA feat, 1-sum of terms of FURIA feat, 2-num of thr feat, 3-sum of terms in thr feat
    public static long learnAllTime[][];
    public static double treeSize[], numOfLeaves[], sumOfTerms[], ratioTermsNodes[], numOfRules[], numOfTerms[], numConstructsPerRule[];

    public static long paramSearchTime[][], paramSLearnT[][];
    public static double complexityOfFuria[][];
    public static double complexityOfFuriaEF1[][], complexityOfFuriaEF2[][], complexityOfFuriaEF3[][];
        
    public static double maxGroupOfConstructs[];
    public static double numOfGroupsOfFeatConstr[];
    public static double avgTermsPerGroup[];
    public static int avgTermsPerFold[];
    public static Set unInfFeatures = new HashSet(); //for controlling informative features
        
    public static PrintWriter logFile, impGroups, impGroupsKD, attrImpListMDL_KD, discIntervalsKD, attrImpListMDL, bestParamPerFold, samplesStat, discIntervals, accByFolds,groupsStat;
    public enum OperationLog{EQU,AND,XOR,IMPL, OR};         //logical operators  - for composing new features
    public enum OperationRel{LESSTHAN,DIFF};                //relational operators - for composing new features
    public enum OperationNum{ADD,SUBTRACT,DIVIDE,ABSDIFF};  //numeric operators  - for composing new features

    public static ArrayList<Double>[] dotsA;
    public static ArrayList<Double>[] dotsB;
    public static String nThHigh;
    public static int processors;
    public static void main(String[] args) throws IOException, Exception {
        /**************************** check the correct setting of EFC *********************************/
        if((justExplain==false && visualisation==false && exhaustive==false && jakulin==true)){
            System.out.println("\u001B[31mYou must set the correct values of the parameters from the following list of settings and run the program again!\u001B[0m");
                      
            System.out.println("\t1) \u001B[34mEFC\033[0m \t\t\t\t\t(justExplain=false, visualisation=false, exhaustive=false, jakulin=false)");
            System.out.println("\t2) \u001B[34mFC based on exhaustive search\033[0m \t(justExplain=false, visualisation=false, exhaustive=true, jakulin=false)");
            System.out.println("\t3) \u001B[34mFC based on interaction information\033[0m \t(justExplain=false, visualisation=false, exhaustive=true, jakulin=true)");
            System.out.println("\t4) \u001B[34mKnowledge discovery\033[0m \t\t\t(justExplain=true, visualisation=false)");
            System.out.println("\t5) \u001B[34mVisualisation\033[0m \t\t\t(justExplain=false, visualisation=true)");
            
            URL myURL = new URL("https://github.com/bostjanv76/featConstr");
            System.out.println("For more instructions, please see "+myURL); 
            System.exit(0);        
        }
        
        tmpDir = System.getProperty("java.io.tmpdir");
        deleteXGBdll();         //if xgboost4j dll exists in temp folder, delete it
        
        String folderName="logs/";
       
        if((justExplain==false && visualisation==false && exhaustive==false && jakulin==false))
            folderName="logs/efc/";
        if((justExplain==false && visualisation==false && exhaustive==true && jakulin==false))
            folderName="logs/exhaustive/";
        if((justExplain==false && visualisation==false && exhaustive==true && jakulin==true))
            folderName="logs/jakulin/";
        
        String lg = new SimpleDateFormat("HH.mm.ss-dd.MM.yyyy").format(new Date());
        if(!justExplain && !visualisation){
            logFile= new PrintWriter(new FileWriter(folderName+"report-"+lg+".log"));
            if(!exhaustive && !jakulin){
                bestParamPerFold= new PrintWriter(new FileWriter(folderName+"params-"+lg+".dat"));
                if(groupsByThrStat)
                    groupsStat = new PrintWriter(new FileWriter(folderName+"groupsStat-"+lg+".csv",true));    //number of identified groups by EFC for each threshold and each fold
            }
        }
        if(!treeSHAP)
            samplesStat= new PrintWriter(new FileWriter(folderName+"samplesStat-"+lg+".dat"));

        if((justExplain && visualisation) || justExplain){
            impGroupsKD = new PrintWriter(new FileWriter(folderName+"kd/impGroups-"+lg+".log"));
            attrImpListMDL_KD = new PrintWriter(new FileWriter(folderName+"kd/attrImpListMDL-"+lg+".dat"));        
            discIntervalsKD = new PrintWriter(new FileWriter(folderName+"kd/discretizationIntervals-"+lg+".dat"));
        }
        else if(!visualisation && !justExplain){            
            impGroups = new PrintWriter(new FileWriter(folderName+"impGroups-"+lg+".log"));
            if(!jakulin)
                attrImpListMDL = new PrintWriter(new FileWriter(folderName+"attrImpListMDL-"+lg+".dat"));
            discIntervals = new PrintWriter(new FileWriter(folderName+"discretizationIntervals-"+lg+".dat"));
        }
             
        File folder;
        Timer t1;
        Timer tTotal=new Timer();
        double [] classDistr;
        
        RCaller rCaller = RCaller.create(); //open RCaller only once and close it at the end of the program
        RCode code = RCode.create();
        
        boolean isClassification=true;
        
        //classification datasets

        /*****demo datasets*****/
        folder = new File("datasets/demo");
    
        /*****artificial datasets*****/ 
        //folder = new File("datasets/artificial");
    
        /*****UCI datasets*****/
        //folder = new File("datasets/uci");
    
        /*****real dataset - credit score*****/       
        //folder = new File("datasets/real");
                     
        File[] listOfFiles = folder.listFiles();

        boolean noFiles=true;
        for(File file : listOfFiles){   
            if(file.isFile()){
                noFiles=false;
                break;
            }
        }
        
        if(noFiles) //check if analysed folder is empty
            System.out.println("\u001B[31mPut dataset(s) in the selected folder and run the program again! The currently selected folder is "+folder.getName()+".\u001B[0m");
           
        loopExplanationVisualisation:
        for(File file : listOfFiles){
            loopExhaustiveTooLong:      
            if(file.isFile()){
                tTotal.start();
                fileName=file.getName();
                System.out.println("dataset: "+fileName);
                if((justExplain && visualisation) || justExplain){
                    impGroupsKD.println("dataset: "+fileName);
                    attrImpListMDL_KD.println("dataset: "+fileName); 
                    discIntervalsKD.println("dataset: "+fileName);
                }      
                else if(!visualisation && !justExplain){ 
                    logFile.println("dataset: "+fileName);
                    if(!exhaustive && !jakulin){
                        bestParamPerFold.println("dataset: "+fileName);
                        if(groupsByThrStat)
                            groupsStat.println("dataset: "+fileName);
                    }
                    if(!treeSHAP)
                        samplesStat.println("dataset: "+fileName);
                
                    impGroups.println("dataset: "+fileName);
                    if(exhaustive && !jakulin)
                        impGroups.println("Exhaustive search");
                    if((!exhaustive && !jakulin) || (exhaustive && !jakulin))   
                        attrImpListMDL.println("dataset: "+fileName);
                    discIntervals.println("dataset: "+fileName);                
                }
                
                Classifier clsTab[]=null;
                String lab[];

                processors = Runtime.getRuntime().availableProcessors();

                //classification
                NaiveBayes nb=new NaiveBayes();
                J48 j48=new J48();
                FURIA furia=new FURIA(); //in WEKA API ruleset is given for whole dataset, we will have rulesets for every fold
                MultilayerPerceptron mp=new MultilayerPerceptron(); //hiddenLayers=(attribs + classes) / 2 ... one hidden layer with (attribs + classes) / 2 units (neurons)
                mp.setOptions(weka.core.Utils.splitOptions("-H 10"));

                RandomForest rf=new RandomForest();
                    rf.setNumExecutionSlots(processors);
                    rf.setCalcOutOfBag(true);

                Instances data = new Instances(new BufferedReader(new FileReader(file)));        
                data.setClassIndex(data.numAttributes()-1); 

                System.gc();

                //just for any case
                String oldName, newName;
                for(int i=0;i<data.numAttributes();i++){
                    oldName=data.attribute(i).name();
                    if(oldName.toUpperCase().contains("==")){
                        newName=oldName.toUpperCase().replace("==", "-IS-");
                    data.renameAttribute(i, newName);
                    }
                }

                if(justExplain){
                    numberOfUnImpFeatByFolds=new double[8][folds]; 

                    ReplaceMissingValues rwm=new ReplaceMissingValues();
                    rwm.setInputFormat(data);
                    data=Filter.useFilter(data, rwm);

//                    attrImpListMDL_KD.println("MDL - before CI");
//                        mdlCORElearn(data, rCaller, code);
                        
                    Instances dataWithNewFeat=justExplainAndConstructFeat(data, rf,true, rCaller, code); //knowledge discovery

                    if(saveConstructs){
                        Instances origNewfeat=null;
                        origNewfeat=new Instances(dataWithNewFeat);
                        String fName1, fName2, fName3, origName=fileName.substring(0, fileName.indexOf('.'));
                        folderName="logs/kd/";
                        String searchString1="origPlusRen", searchString2="LFeat";
                        int featLevel;

                        if(fileName.contains(searchString1) && fileName.contains(searchString2)){
                            int startIdx=fileName.indexOf(searchString1);
                            int endIdx=fileName.indexOf(searchString2)+searchString2.length();
                            String tmpString=fileName.substring(startIdx, endIdx);
                            featLevel=Integer.parseInt(tmpString.substring(searchString1.length(),searchString1.length()+1))+1; //increase feature level
                            fName1="origPlus"+featLevel+searchString2;
                            fName2="names-"+featLevel+"-level-feat";
                            fName3=searchString1+featLevel+searchString2;
                            origName=fileName.substring(0, fileName.indexOf("-"+searchString1));
                        }
                        else{
                            featLevel=1;
                            fName1 ="origPlus1LFeat";
                            fName2 ="names-1-level-feat";
                            fName3 ="origPlusRen1LFeat";
                        }

                        DataSink.write(folderName+origName+"-"+fName1+"-"+lg+".arff", origNewfeat);
                        if(renameGenFeat){
                            int iName=1;
                            String tmpAttrName;
                            int oldNumAttr=data.numAttributes()-1;
                            PrintWriter attNames= new PrintWriter(new FileWriter(folderName+fName2+"-"+lg+".dat"));
                            attNames.println("NEW \t OLD");
                            for(int i=oldNumAttr;i<dataWithNewFeat.numAttributes()-1;i++){
                                tmpAttrName="F"+iName+"L"+featLevel;
                                attNames.println(tmpAttrName+"\t"+dataWithNewFeat.attribute(i).name());    
                                dataWithNewFeat.renameAttribute(i, tmpAttrName);
                                iName++;
                            }
                            attNames.close();
                            DataSink.write(folderName+origName+"-"+fName3+"-"+lg+".arff", dataWithNewFeat); //save the dataset with renamed features (F1L1, F2L1 ...) for the potential generating 2nd level features
                        }
                    }
                    continue loopExplanationVisualisation;                                     
                }

                if(!visualisation){
                    System.out.println("Number of folds for testing (CV): "+folds);
                        logFile.println("Number of folds for testing (CV): "+folds);
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");
                }
                
                if(data.classAttribute().isNumeric())
                    isClassification=false;

                if(isClassification)
                    clsTab=new Classifier[]{j48, nb, furia, rf};          

                accuracyByFolds=new double[clsTab.length][folds];
                accuracyByFoldsPS=new double[clsTab.length][folds];
                accuracyByFoldsFuriaThr=new double[clsTab.length][folds];
                accuracyByFoldsEvalAtt1=new double[clsTab.length][folds];
                accuracyByFoldsEvalAtt2=new double[clsTab.length][folds];
                accuracyByFoldsEvalAtt3=new double[clsTab.length][folds];
                accByFoldsLF=new double[clsTab.length][folds];
                accByFoldsCP=new double[clsTab.length][folds];
                accByFoldsRE=new double[clsTab.length][folds];
                accByFoldsNum=new double[clsTab.length][folds];

                numberOfFeatByFolds=new double[6][folds];
                numOfFeatByFoldsLF=new double[folds];
                numFeatByFoldsCP=new double[folds];
                numFeatByFoldsRE=new double[folds];
                numFeatByFoldsNum=new double[folds];
                numFeatByFoldsFuriaThr=new double[2][folds];
                numFeatByFoldsEvalFeat1=new double[3][folds];numFeatByFoldsEvalFeat2=new double[3][folds];numFeatByFoldsEvalFeat3=new double[3][folds];
                numTreeByFoldsEvalFeat1=new double[3][folds];numTreeByFoldsEvalFeat2=new double[4][folds];numTreeByFoldsEvalFeat3=new double[3][folds];
                exlpTime=new long[folds]; allFCTime=new long[folds]; allFCTimeLF=new long[folds]; cartesianFCTime=new long[folds];relationalFCTime=new long[folds];
                numericalFCTime=new long[folds]; furiaThrTime=new long[folds]; attrEval1Time=new long[folds]; attrEval2Time=new long[folds]; attrEval3Time=new long[folds];
                paramSearchTime=new long[clsTab.length][folds]; paramSLearnT=new long[clsTab.length][folds];
                learnAllFCTime=new long[clsTab.length][folds]; learnAllFCTimeLF=new long[clsTab.length][folds];
                learnAllFCTimeCP=new long[clsTab.length][folds];
                learnAllFCTimeRE=new long[clsTab.length][folds];
                learnAllFCTimeNum=new long[clsTab.length][folds];
                learnFuriaThrTime=new long[clsTab.length][folds]; learnAttrEval1Time=new long[clsTab.length][folds]; learnAttrEval2Time=new long[clsTab.length][folds]; learnAttrEval3Time=new long[clsTab.length][folds];
                featByFoldsPS=new double[6][folds][clsTab.length];
                numberOfTreeByFolds=new double[4][folds];
                numOfTreeByFoldsLF=new double[4][folds];  
                numOfTreeByFoldsCP=new double[4][folds];
                numOfTreeByFoldsRE=new double[4][folds];
                numOfTreeByFoldsNum=new double[4][folds];
                numTreeByFoldsFuriaThr=new double[4][folds];
                numberOfTreeByFoldsPS=new double[4][folds];    
                numCartFeatInTreeFS=new double[2][folds]; 
                numberOfUnImpFeatByFolds=new double[8][folds];  //OR, EQU, XOR, IMPL, AND, LESSTHAN, DIFF, CARTESIAN
                maxGroupOfConstructs=new double[folds];
                numOfGroupsOfFeatConstr=new double[folds];
                numOfExplainedInst=new double[folds];
                avgTermsPerGroup=new double[folds];
                avgTermsPerFold=new int[folds];
                numOfRulesByFolds=new double[folds]; numOfTermsByFoldsF=new double[folds];
                numOfRulesByFoldsLF=new double[folds]; numOfTermsByFoldsLF=new double[folds]; numOfRatioByFoldsLF=new double[folds];        //for logical features
                numOfRulesByFoldsCP=new double[folds]; numOfTermsByFoldsCP=new double[folds]; numOfRatioByFoldsCP=new double[folds];        //for Cartesian features
                numOfRulesByFoldsRE=new double[folds]; numOfTermsByFoldsRE=new double[folds]; numOfRatioByFoldsRE=new double[folds];        //for relational features
                numOfRulesByFoldsNum=new double[folds]; numOfTermsByFoldsNum=new double[folds]; numOfRatioByFoldsNum=new double[folds];     //for numerical features
                complexityOfFuriaPS=new double[3][folds];
                numOfFuriaThrInTreeByFolds=new double[4][folds];
                numOfFuriaThrInTreeByFoldsF=new double[4][folds];
                numOfFuriaThrInTreeByFoldsM=new double[4][folds];
                numOfFuriaThrInTreeByFoldsP=new double[4][folds];
                learnAllTime=new long[clsTab.length][folds];
                accOrigModelByFolds=new double[clsTab.length][folds];
                accExplAlgInt=new double[folds]; accExplAlgTest=new double[folds];
                complexityOfFuria=new double[3][folds];
                complexityOfFuriaEF1=new double[3][folds]; complexityOfFuriaEF2=new double[3][folds];complexityOfFuriaEF3=new double[3][folds];
                treeSize=new double[folds]; numOfLeaves=new double[folds]; sumOfTerms=new double[folds]; ratioTermsNodes=new double[folds]; numOfRules=new double[folds]; 
                numOfTerms=new double[folds]; numConstructsPerRule=new double[folds]; numOfRatioByFoldsF=new double[folds];
                modelBuildTime=new long[folds];
                oobRF=new double[folds];
                numOfCartesian=new double[folds];
                numOfRelational=new double[folds];
                numOfNumerical=new double[folds];
                numOfCartFAll=new double[folds];              
                numLogFeatInTreeFS=new double[2][folds];    //0-number of log features, 1-sum of constructs  
                numNumFeatInTreeFS=new double[2][folds];    //0-number of numerical features, 1-sum of constructs  
                numRelFeatInTreeFS=new double[2][folds];    //0-number of rel features, 1-sum of constructs  
                numOfLogicalInTree=new double[2][folds];
                numOfLogInTreeAll=new double[folds];
                numOfNumInTreeAll=new double[folds];
                numOfRelInTreeAll=new double[folds];
                sumOfConstrCart=new double[folds];
                sumOfConstrRel=new double[folds];
                sumOfConstrNum=new double[folds];
                sumOfConstrCartAll=new double[folds];
                sumOfConstrLFAll=new double[folds];
                sumOfConstrNumAll=new double[folds];
                sumOfConstrRelAll=new double[folds];

                if(!visualisation){
                    System.out.println("number of instances: "+data.numInstances());
                        logFile.println("number of instances: "+data.numInstances());
                    System.out.println("number of attributes: "+(data.numAttributes()-1));
                        logFile.println("number of attributes: "+(data.numAttributes()-1));
                    System.out.println("number of classes: "+(data.numClasses()));
                        logFile.println("number of classes: "+(data.numClasses()));
                }
                data.setClassIndex(data.numAttributes()-1);

                //HEURISTICS OF CLASS SELECTION FOR EXPLANATION
                //array of frequencies for each class - how many instances occur in a particular class
                classDistr=Arrays.stream(data.attributeStats(data.classIndex()).nominalCounts).asDoubleStream().toArray(); //we convert because we need in log2Multinomial as parameter double array

                for(int i=0;i<minIndexClassifiers(classDistr).length;i++){
                    if(minIndexClassifiers(classDistr)[i].v>=Math.ceil(data.numInstances()*instThr/100.00)){ //we choose class to explain - class has to have at least instThr pct of whole instances
                        classToExplain=minIndexClassifiers(classDistr)[i].i;
                        break;
                    }
                }

                Classifier predictionModel=rf;
                if(!visualisation){
                    System.out.println("---------------------------------------------------------------------------------");
                        logFile.println("---------------------------------------------------------------------------------"); 
                }

                //SPLIT TO TRAIN AND TEST - N fold CV or split in ratio, depends on what number of folds has been chosen
                ReplaceMissingValues rwm;
                Random rand = new Random(1);
                Instances randData=new Instances(data);
                Instances test;
                randData.randomize(rand);
                if(isClassification && folds>1)
                    randData.stratify(folds); //for imbalanced datasets before splitting the dataset into train and test set - same as WEKA GUI

                /******************* VISUALISATION *********************************/
                if(visualisation){        
                    System.out.println("Drawing ...");
                    visualizeModelInstances(rf, data, true, RESOLUTION, numOfImpt, visFrom, visTo);  //visualise explanations from e.g., 50th to 60 instance
                    System.out.println("Drawing is finished!");
                    continue loopExplanationVisualisation;  
                }     

                /*************************************  FOLDS  *********************************************************************/        
                for (int f = 0; f < folds; f++){
                    unInfFeatures.clear();      //clear set for each fold
                    int minN=minNoise;
                    if(!jakulin){
                        attrImpListMDL.println("\t\t\t\t\t\t\t\t--------------"); 
                        attrImpListMDL.printf("\t\t\t\t\t\t\t\t\tFold %2d\n",(f+1));
                        attrImpListMDL.println("\t\t\t\t\t\t\t\t--------------");
                    }

                    if(folds==1){
                        StratifiedRemoveFolds fold;
                        fold = new StratifiedRemoveFolds();
                        fold.setInputFormat(randData);
                        fold.setSeed(1);
                        fold.setNumFolds(splitTrain);
                        fold.setFold(splitTrain);
                        fold.setInvertSelection(true);  //because we invert selection we take all folds except the "split" one
                        data = Filter.useFilter(randData,fold); 

                        fold = new StratifiedRemoveFolds();
                        fold.setInputFormat(randData);
                        fold.setSeed(1);
                        fold.setNumFolds(splitTrain);
                        fold.setFold(splitTrain);
                        fold.setInvertSelection(false);
                        test = Filter.useFilter(randData,fold); 
                    }
                    else{
                        data = randData.trainCV(folds, f,rand);  //same as WEKA GUI
                        test= randData.testCV(folds, f);
                    }

                        rwm=new ReplaceMissingValues();
                        rwm.setInputFormat(data);
                        data=Filter.useFilter(data, rwm);
                        test=Filter.useFilter(test, rwm); //insert mean values from train dataset
       
                        discIntervals.println("\t\t\t\t\t\t\t\t--------------"); 
                        discIntervals.printf("\t\t\t\t\t\t\t\t\tFold %2d\n",(f+1));
                        discIntervals.println("\t\t\t\t\t\t\t\t--------------"); 
                        namesOfDiscAttr(data);  //save discretization intervals
             
                    ModelAndAcc ma; 
                    Classifier model;
                    
                        for (int m=0;m<clsTab.length;m++){            
                            model=clsTab[m];
                            t1=new Timer();
                            t1.start();
                                ma=evaluateModel(data, test, model);
                            t1.stop();
                            accOrigModelByFolds[m][f]=ma.getAcc();
                            learnAllTime[m][f]=t1.diff();

                            model=ma.getClassifier();
                            if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                j48=new J48(); 
                                j48=(J48)(model);
                                treeSize[f]=j48.measureTreeSize();
                                numOfLeaves[f]=j48.measureNumLeaves();
                                sumOfTerms[f]=sumOfTermsInConstrInTree(data,data.numAttributes()-1, j48);
                                ratioTermsNodes[f]=(treeSize[f]-numOfLeaves[f])==0 ? 0 : sumOfTerms[f]/(treeSize[f]-numOfLeaves[f]);
                            }
                            if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                FURIA fu=new FURIA();
                                fu=(FURIA)(model);
                                numOfRules[f]= fu.getRuleset().size();
                                numOfTerms[f]=sumOfTermsInConstrInRule(fu.getRuleset(),data);
                                numConstructsPerRule[f]=numOfRules[f]==0 ? 0 : (numOfTerms[f]/numOfRules[f]);
                            }
                        } 
                         
                    if(!exhaustive){
                        double allExplanations[][]=null;
                        double allWeights[][]=null;
                        float allExplanationsSHAP[][];
                        float allWeightsSHAP[][]=null;

                        List<String>impInter=null;
                        Set<String> attrGroups= new LinkedHashSet<>();  //we want to keep the insertion order, we don't want duplicates - use of LinkedHashSet

                        int numClasses=1; //1 - just one iteration, we explain minority class, otherwise numClasses=classDistr.length; we explain all classes          

                        if(explAllClasses)
                            numClasses=classDistr.length;

                        /*SHAP*/  
                        if(treeSHAP){
                        /*XGBOOST*/
                            DMatrix trainMat = wekaInstancesToDMatrix(data);
                            DMatrix testMat =wekaInstancesToDMatrix(test);
                            float tmpContrib[][];
                            int numOfClasses=data.numClasses();
                            HashMap<String, Object> params = new HashMap<>();
                                params.put("eta", eta); //"eta - learning_rate ("shrinkage" parameter)": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ]  It is advised to have small values of eta in the range of 0.1 to 0.3 because of overfitting
                                params.put("max_depth", maxDepth);
                                params.put("silent", 1);    //print 
                                params.put("nthread", processors);
                                params.put("gamma", gamma);   //"gamma-min_split_loss": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ], gamma works by regularising using "across trees" information

                            if(numOfClasses==2){    //for binary examples
                                params.put("objective", "binary:logistic");  //binary:logistic ??? logistic regression for binary classification, returns predicted probability (not class)
                                params.put("eval_metric", "error");
                            }
                            else{                   //multi class problems
                                params.put("objective", "multi:softmax");  //multi:softprob multi:softmax
                                params.put("eval_metric", "merror");
                                params.put("num_class", (numOfClasses));    
                            }

                            Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {{
                                put("train", trainMat);
                                put("test", testMat);
                            }
                            };    

                           //building model
                           t1=new Timer();
                           t1.start();                
                                Booster booster = XGBoost.train(trainMat, params, numOfRounds, watches, null, null);
                           t1.stop();
                           modelBuildTime[f]=t1.diff();
                           String evalNameTest[]={"test"};
                           String evalNameTrain[]={"train"};
                           DMatrix [] testMatArr={testMat};
                           DMatrix [] trainMatArr={trainMat};

                           String accTrain=booster.evalSet(trainMatArr, evalNameTrain,0); 
                           String accTest=booster.evalSet(testMatArr, evalNameTest,0);
                           testMatArr=null; 
                           trainMatArr=null;
                           
                           accExplAlgInt[f]=(1-Double.parseDouble(accTrain.split(":")[1]))*100; //internal evaluation of the model
                           accExplAlgTest[f]=(1-Double.parseDouble(accTest.split(":")[1]))*100; //evaluation on the test dataset

                           //explaining model
                           t1=new Timer();
                           t1.start();            

                            for(int c=0;c<numClasses;c++){//numClasses=1; (we explain minority class), numClasses=classDistr.length; (we explain all classes) 
                                if(explAllClasses)
                                    classToExplain=c;

                                Instances explainData=new Instances(data);
                                RemoveWithValues filter = new RemoveWithValues();
                                filter.setAttributeIndex("last") ;  //class
                                filter.setNominalIndices((classToExplain+1)+""); //what we remove ... +1 because indexes go from 0, we need indexes from 1 for method setNominalIndices
                                filter.setInvertSelection(true); //if we invert selection than we keep selected data ... 
                                filter.setInputFormat(explainData);
                                explainData = Filter.useFilter(explainData, filter);

                                if(f==0){   //print this info only once        
                                    System.out.println("Alg. for searching concepts: TreeSHAP (XGBOOST) parameters: numOfRounds->"+numOfRounds+" maxDepth->"+maxDepth+" eta->"+eta+" gamma->"+gamma);
                                        logFile.println("Alg. for searching concepts: TreeSHAP (XGBOOST) parameters: numOfRounds->"+numOfRounds+" maxDepth->"+maxDepth+" eta->"+eta+" gamma->"+gamma);
                                    System.out.println("Explaining class: "+data.classAttribute().value(classToExplain)+" explaining whole dataset: "+(explAllData?"YES":"NO"));
                                        logFile.println("Explaining class: "+data.classAttribute().value(classToExplain)+" explaining whole dataset: "+(explAllData?"YES":"NO"));
                                    System.out.println("---------------------------------------------------------------------------------");
                                        logFile.println("---------------------------------------------------------------------------------");
                                }

                                numInst=data.attributeStats(data.classIndex()).nominalCounts[classToExplain]; //number of instances in explained class
                                if(numInst==0)
                                    continue;   //class has no instances e.g., class -3 in dataset autos

                                //we are explaining just instances from the explained class and not also from other classes
                                //if we have more than maxToExplain (e.g. 500) instances we take only maxToExplain instances
                                if(!explAllData){
                                    if(numInst>maxToExplain){
                                        System.out.println("We take only "+maxToExplain+" instances out of "+numInst+".");
                                            impGroups.println("We take only "+maxToExplain+" instances out of "+numInst+".");
                                        explainData.randomize(rand);
                                        explainData = new Instances(explainData, 0, maxToExplain);
                                        numInst=explainData.attributeStats(explainData.classIndex()).nominalCounts[classToExplain]; //for correct print on output
                                    }
                                }

                                DMatrix explainMat = wekaInstancesToDMatrix(explainData);
                                if(explAllData)
                                    tmpContrib=booster.predictContrib(trainMat, 0);   //Tree SHAP ... for each feature, and last for bias matrix of size (???nsample, nfeats + 1) ... feature contributions (SHAP???  xgboost predict)
                                else
                                    tmpContrib=booster.predictContrib(explainMat, 0); //tree limit - Limit number of trees in the prediction; defaults to 0 (use all trees).
                                
                                explainMat.dispose();
                                testMat.dispose();
                                trainMat.dispose();
                                booster.dispose();
                                t1.stop();

                                //Note that shap_values for the two classes are additive inverses for a binary classification problem!!!
                                //The variant of SHAP which deals with trees (TreeSHAP) calculates exact Shapley values and does it fast.
                                if(numOfClasses==2){
                                    allExplanationsSHAP=removeCol(tmpContrib, tmpContrib[0].length-1);  //we remove last column, because we do not need column with bias
                                }
                                else{
                                int idxQArr[]=new int[data.numAttributes()-1];

                                    if(classToExplain==0)
                                        for(int i=0;i<idxQArr.length;i++)
                                            idxQArr[i]=i;
                                    else{
                                        int start=(classToExplain*data.numAttributes()-1)+1;
                                        int j=0;
                                        for(int i=start;i<=idxQArr.length*(classToExplain+1);i++){
                                            idxQArr[j]=i;
                                            j++;
                                        }
                                    }                
                                    allExplanationsSHAP=someColumns(tmpContrib, idxQArr);  //we take just columns of attributes from the class that we explain
                                }

                                if(numInst<minExplInst)
                                    minN=minMinNoise;

                                double noiseThr=(numInst*NOISE)/100.0; //we take number of noise threshold from the number of explained instances
                                int usedNoise=Math.max((int)Math.ceil(noiseThr),minN);  //makes sense only if NOISE=0

                                System.out.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Tree SHAP num of expl. inst. "+(explAllData ? data.numInstances() : numInst)+" (fold "+(f+1)+").");
                                    impGroups.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Tree SHAP num of expl. inst. "+(explAllData ? data.numInstances() : numInst)+" (fold "+(f+1)+").");
                                    impGroups.println("Lower threshold thrL: "+thrL+" upper threshold thrU: "+thrU+" with step: "+step);

                                impGroups.println("\t\t\t\t\t\t\t\t--------------"); 
                                impGroups.printf("\t\t\t\t\t\t\t\t\tFold %2d\n",(f+1));
                                impGroups.println("\t\t\t\t\t\t\t\t--------------"); 

                                if(groupsByThrStat && !visualisation && !justExplain && !exhaustive && !jakulin && f==0){
                                    DecimalFormat df = new DecimalFormat("0.0");
                                    for(double q=thrL;q<=thrU;q=q+step)
                                        groupsStat.write(df.format(q)+";");
                                    groupsStat.println();
                                }
                                for(double q=thrL;q<=thrU;q=q+step){
                                    impGroups.println("--------------"); 
                                    impGroups.printf("Threshold: %2.2f\n",round(q,1));
                                    impGroups.println("--------------"); 

                                    allWeightsSHAP=setWeights(data,allExplanationsSHAP,round(q,1));
                                    impInter=(getMostFqSubsets(allWeightsSHAP,data,usedNoise));
                                    attrGroups.addAll(impInter);
                                    if(groupsByThrStat && !visualisation && !justExplain && !exhaustive && !jakulin)
                                        groupsStat.write(impInter.size()+";");
                                }
                                if(groupsByThrStat && !visualisation && !justExplain && !exhaustive && !jakulin)
                                    groupsStat.println();
                            }//loop explain (all) class(es)
                        }
                        else{
                            System.out.println("Building model ...");
                            t1=new Timer();
                            t1.start();
                                predictionModel.buildClassifier(data);                  
                            t1.stop();
                            System.out.println("Prediction model created.");
                            modelBuildTime[f]=t1.diff();

                            if(excludeUppers(predictionModel.getClass().getSimpleName()).equals("RF")){ //OOB
                                rf=(RandomForest)predictionModel;
                                oobRF[f]=(1-rf.measureOutOfBagError())*100;
                            }

                            //evaluate prediction model on test data
                            Evaluation eval = new Evaluation(data);
                            eval.evaluateModel(predictionModel, test);

                            accExplAlgTest[f]=(eval.correct())/(eval.incorrect()+eval.correct())*100.00;    //same as 1-eval.errorRate())*100.0
                            if(!treeSHAP){
                                samplesStat.println("\t\t\t\t\t\t\t\t--------------"); 
                                samplesStat.printf("\t\t\t\t\t\t\t\t\tFold %2d\n",(f+1));
                                samplesStat.println("\t\t\t\t\t\t\t\t--------------"); 
                            }

                            /*IME*/            
                            for(int i=0;i<numClasses;i++){  //numClasses=1; (we explain minority class), numClasses=classDistr.length; (we explain all classes) 
                                if(explAllClasses)
                                    classToExplain=i;
                                if(f==0){   //print this info only once  
                                    System.out.println("IME (explanation), "+method.name()+", "+(method.name().equals("adaptiveSampling") ? "min samples: "+minS+", sum of samples: "+sumOfSmp : method.name().equals("diffSampling")?" min samples: "+minS:" N_SAMPLES: "+N_SAMPLES)+" - alg. for searching concepts: "+predictionModel.getClass().getSimpleName());
                                        logFile.println("IME (explanation), "+method.name()+", "+(method.name().equals("adaptiveSampling") ? "min samples: "+minS+", sum of samples: "+sumOfSmp : method.name().equals("diffSampling")?" min samples: "+minS:" N_SAMPLES: "+N_SAMPLES)+" - alg. for searching concepts: "+predictionModel.getClass().getSimpleName());  
                                    System.out.println("Explaining class: "+data.classAttribute().value(classToExplain)+", explaining whole dataset: "+(explAllData?"YES":"NO"));
                                        logFile.println("Explaining class: "+data.classAttribute().value(classToExplain)+", explaining all dataset: "+(explAllData?"YES":"NO"));
                                    System.out.println("---------------------------------------------------------------------------------");
                                        logFile.println("---------------------------------------------------------------------------------");

                                    switch(method){
                                        case aproxErrSampling:
                                            System.out.println("Sampling based on mi=(<1-alpha, e>), pctErr = "+pctErr+" error = "+error+".");
                                                logFile.println("Sampling based on mi=(<1-alpha, e>), pctErr = "+pctErr+" error = "+error+".");
                                            System.out.println("---------------------------------------------------------------------------------");
                                                logFile.println("---------------------------------------------------------------------------------");                                    
                                        break;
                                    }                
                                }

                                numInst=data.attributeStats(data.classIndex()).nominalCounts[classToExplain]; //number of instances in explained class
                                if(numInst==0)
                                    continue;   //class has no instances e.g., class -3 in dataset autos

                                Instances explainData=new Instances(data);
                                RemoveWithValues filter = new RemoveWithValues();
                                filter.setAttributeIndex("last") ;                  //class
                                filter.setNominalIndices((classToExplain+1)+"");    //what we remove ... +1 because indexes go from 0, we need indexes from 1 for method setNominalIndices
                                filter.setInvertSelection(true);                    //if we invert selection then we keep selected data ... 
                                filter.setInputFormat(explainData);
                                explainData = Filter.useFilter(explainData, filter);

                                //we are explaining just instances from the explained class and not also instances from other classes
                                //if we have more than maxToExplain (e.g. 500) instances we take only maxToExplain instances
                                if(!explAllData){
                                    if(numInst>maxToExplain){
                                        System.out.println("We take only "+maxToExplain+" instances out of "+numInst+".");
                                            impGroups.println("We take only "+maxToExplain+" instances out of "+numInst+".");

                                        explainData.randomize(rand);
                                        explainData = new Instances(explainData, 0, maxToExplain);
                                        numInst=explainData.attributeStats(explainData.classIndex()).nominalCounts[classToExplain]; //for correct print on output
                                    }
                                    t1=new Timer();
                                    t1.start();
                                        switch (method){
                                            case equalSampling: 
                                                allExplanations=IME.explainAllDatasetES(data,explainData,predictionModel,N_SAMPLES, classToExplain);    //equal sampling
                                                break;
                                            case adaptiveSamplingSS:  
                                                allExplanations=IME.explainAllDatasetAS(data,explainData,predictionModel, minS, sumOfSmp, classToExplain);     //we need sumOfSmp (sum of samples) for additive sampling
                                                break;
                                            case adaptiveSamplingAE: 
                                                allExplanations=IME.explainAllDatasetAS(data,explainData,predictionModel, minS, classToExplain, error, pctErr);
                                                break;
                                            case aproxErrSampling:  
                                                allExplanations=IME.explainAllDatasetAES(predictionModel, data, explainData,true, classToExplain, minS, error, pctErr);                   
                                                break;
                                        }
                                    t1.stop();
                                }
                                else{
                                    t1=new Timer();
                                    t1.start();
                                        switch (method){
                                            case equalSampling: 
                                                allExplanations=IME.explainAllDatasetES(data,data,predictionModel,N_SAMPLES, classToExplain); //equal sampling
                                                break;
                                            case adaptiveSamplingSS: 
                                                allExplanations=IME.explainAllDatasetAS(data,data,predictionModel, minS, sumOfSmp, classToExplain);//we need sumOfSmp (sum of samples) for additive sampling
                                                break;
                                             case adaptiveSamplingAE:   
                                                allExplanations=IME.explainAllDatasetAS(data,data,predictionModel, minS, classToExplain, error, pctErr);
                                                break;                                            
                                            case aproxErrSampling: 
                                                allExplanations=IME.explainAllDatasetAES(predictionModel, data, data, true, classToExplain, minS, error,pctErr);
                                                break;
                                        }
                                    t1.stop();
                                }   

                                if(numInst<minExplInst)
                                    minN=minMinNoise;

                                double noiseThr=(numInst*NOISE)/100.0;//we take number of noise threshold from the number of explained instances
                                int usedNoise=Math.max((int)Math.ceil(noiseThr),minN);  //makes sense only if NOISE=0 or num of explained instances is very low

                                System.out.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Number of instances from class ("+explainData.classAttribute().value(classToExplain)+") is "+numInst+" (fold "+(f+1)+").");
                                    impGroups.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Number of instances from class ("+explainData.classAttribute().value(classToExplain)+") is "+numInst+" (fold "+(f+1)+").");
                                    impGroups.println("Lower threshold thrL: "+thrL+" upper threshold thrU: "+thrU+" with step: "+step);

                                impGroups.println("\t\t\t\t\t\t\t\t--------------"); 
                                impGroups.printf("\t\t\t\t\t\t\t\t\tFold %2d\n",(f+1));
                                impGroups.println("\t\t\t\t\t\t\t\t--------------"); 

                                for(double q=thrL;q<=thrU;q=q+step){
                                    impGroups.println("--------------"); 
                                    impGroups.printf("Threshold: %2.2f\n",round(q,1));
                                    impGroups.println("--------------"); 

                                    allWeights=setWeights(data,allExplanations,round(q,1));
                                    impInter=(getMostFqSubsets(allWeights,data,usedNoise));
                                    attrGroups.addAll(impInter);
                                }     
                            }   //loop explain (all) class(es)                
                        }   //condition SHAP or IME

                        exlpTime[f]=t1.diff();
                        numOfExplainedInst[f]=numInst;
                        listOfConcepts = new ArrayList<>(attrGroups);
                        
                        if(listOfConcepts.size()==0){//if we didn't find any concepts in this fold we take results from the fold that doesn't consist of CI
                            //take ACC and learning time from orig. dataset
                            for (int i=0;i<clsTab.length;i++){
                                //logical
                                accByFoldsLF[i][f]=accOrigModelByFolds[i][f];
                                learnAllFCTimeLF[i][f]=learnAllTime[i][f];
                                //numerical
                                accByFoldsNum[i][f]=accOrigModelByFolds[i][f];
                                learnAllFCTimeNum[i][f]=learnAllTime[i][f]; 
                                //Cartesian product    
                                accByFoldsCP[i][f]=accOrigModelByFolds[i][f];
                                learnAllFCTimeCP[i][f]=learnAllTime[i][f];    
                                //relational
                                accByFoldsRE[i][f]=accOrigModelByFolds[i][f];
                                learnAllFCTimeRE[i][f]=learnAllTime[i][f]; 
                                //FURIA and thr    
                                accuracyByFoldsFuriaThr[i][f]=accOrigModelByFolds[i][f];
                                learnFuriaThrTime[i][f]=learnAllTime[i][f];
                                //All features
                                accuracyByFolds[i][f]=accOrigModelByFolds[i][f];
                                learnAllFCTime[i][f]=learnAllTime[i][f];  
                                //FS on validation dataset    
                                accuracyByFoldsPS[i][f]=accOrigModelByFolds[i][f];
                                paramSLearnT[i][f]=learnAllTime[i][f];
                            }     
                            //take treeSize from orig. dataset
                            //logical
                            numOfTreeByFoldsLF[0][f]=treeSize[f];
                            //numerical
                            numOfTreeByFoldsNum[0][f]=treeSize[f];
                            //Cartesian product
                            numOfTreeByFoldsCP[0][f]=treeSize[f];
                            //relational
                            numOfTreeByFoldsRE[0][f]=treeSize[f];
                            //FURIA and thr   
                            numTreeByFoldsFuriaThr[0][f]=treeSize[f];                    
                            //All features
                            numberOfTreeByFolds[0][f]=treeSize[f];
                            //FS on validation dataset 
                            numberOfTreeByFoldsPS[0][f]=treeSize[f];
                            //take numOfLeaves from orig. dataset
                            //logical
                            numOfTreeByFoldsLF[1][f]=numOfLeaves[f]; 
                            //numerical
                            numOfTreeByFoldsNum[1][f]=numOfLeaves[f];
                            //Cartesian product
                            numOfTreeByFoldsCP[1][f]=numOfLeaves[f];
                            //relational
                            numOfTreeByFoldsRE[1][f]=numOfLeaves[f];
                            //FURIA and thr  
                            numTreeByFoldsFuriaThr[1][f]=numOfLeaves[f]; //numOfLeaves
                            //All features
                            numberOfTreeByFolds[1][f]=numOfLeaves[f]; //numOfLeaves
                            //FS on validation dataset
                            numberOfTreeByFoldsPS[1][f]=numOfLeaves[f];
                            //take sumOfTerms from orig. dataset
                            //logical
                            numOfTreeByFoldsLF[2][f]=sumOfTerms[f];
                            //numerical
                            numOfTreeByFoldsNum[2][f]=sumOfTerms[f];
                            //Cartesian product
                            numOfTreeByFoldsCP[2][f]=sumOfTerms[f];
                            //relational
                            numOfTreeByFoldsRE[2][f]=sumOfTerms[f];
                            //FURIA and thr
                            numTreeByFoldsFuriaThr[2][f]=sumOfTerms[f]; 
                            //All features
                            numberOfTreeByFolds[2][f]=sumOfTerms[f]; 
                            //FS on validation dataset
                            numberOfTreeByFoldsPS[2][f]=sumOfTerms[f];                        
                            //ratio between sumOfTerms and nodes in original dataset is 1 - one node equals one attribute it can be also 0, just one leave!!!
                            numOfTreeByFoldsLF[3][f]=ratioTermsNodes[f];        //logical
                            numOfTreeByFoldsNum[3][f]=ratioTermsNodes[f];       //numerical                    
                            numOfTreeByFoldsCP[3][f]=ratioTermsNodes[f];        //Cartesian product
                            numOfTreeByFoldsRE[3][f]=ratioTermsNodes[f];        //relational feat
                            numTreeByFoldsFuriaThr[3][f]=ratioTermsNodes[f];    //FURIA and thr
                            numberOfTreeByFolds[3][f]=ratioTermsNodes[f];       //All features
                            numberOfTreeByFoldsPS[3][f]=ratioTermsNodes[f];     //FS                   
                            //take numOfRules and numOfTerms from orig. dataset    
                            //logical
                            numOfRulesByFoldsLF[f]=numOfRules[f];
                            numOfTermsByFoldsLF[f]=numOfTerms[f];
                            numOfRatioByFoldsLF[f]=numConstructsPerRule[f];   
                            //numerical
                            numOfRulesByFoldsNum[f]=numOfRules[f];
                            numOfTermsByFoldsNum[f]=numOfTerms[f];
                            numOfRatioByFoldsNum[f]=numConstructsPerRule[f];
                            //Cartesian product
                            numOfRulesByFoldsCP[f]=numOfRules[f];
                            numOfTermsByFoldsCP[f]=numOfTerms[f];
                            numOfRatioByFoldsCP[f]=numConstructsPerRule[f];
                            //relational
                            numOfRulesByFoldsRE[f]=numOfRules[f];
                            numOfTermsByFoldsRE[f]=numOfTerms[f];
                            numOfRatioByFoldsRE[f]=numConstructsPerRule[f];
                            //FURIA and thr
                            complexityOfFuria[0][f]=numOfRules[f];
                            complexityOfFuria[1][f]=numOfTerms[f];
                            complexityOfFuria[2][f]=numConstructsPerRule[f];
                            //All features
                            numOfRulesByFolds[f]=numOfRules[f];
                            numOfTermsByFoldsF[f]=numOfTerms[f];
                            numOfRatioByFoldsF[f]=numConstructsPerRule[f];
                            //FS on validation dataset
                            complexityOfFuriaPS[0][f]=numOfRules[f];
                            complexityOfFuriaPS[1][f]=numOfTerms[f];   
                            complexityOfFuriaPS[2][f]=numConstructsPerRule[f];
                        }
                    }        
                    else{
                        String idxOfAttr="";
                        for (int i=0;i<data.numAttributes()-1;i++)
                            if(i<data.numAttributes()-2)
                                idxOfAttr+=i+",";
                            else
                                idxOfAttr+=i;
                        System.out.println(idxOfAttr);
                        listOfConcepts = new ArrayList<>();    
                        listOfConcepts.add(idxOfAttr);  //we try all combinations
                    }

                    if(listOfConcepts.size()!=0 && !jakulin){
                        impGroups.println("*********************************************************************************"); 
                        impGroups.println("All potential concepts based on thresholds");
                        impGroups.print("\t"); printFqAttrOneRow(listOfConcepts,data);
                        impGroups.println("\n*********************************************************************************");
                    }

                    int sumMax[]=printMaxConstructLength(listOfConcepts); //sumMax[0] number of all constructs (attributes) in all groups
                    if(listOfConcepts.size()!=0)
                        avgTermsPerGroup[f]=sumMax[0]/(double)listOfConcepts.size(); //average number of constructs per group
                    avgTermsPerFold[f]=sumMax[0];       //number of all attributes in groups in one fold
                    maxGroupOfConstructs[f]=sumMax[1];  //length of the longest construct     
                    t1=new Timer();
                    t1.start();

                    numOfGroupsOfFeatConstr[f]=listOfConcepts.size();

                    if(numOfGroupsOfFeatConstr[f]==0){
                        System.out.println("We didn't find any concepts in fold "+(f+1)+" above threshold max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(numInst*NOISE/100.0)+ ", minNoise="+minN+" we remove groups of size "+Math.max((int)Math.ceil(numInst*NOISE/100.0),minN));  
                        logFile.println("We didn't find any concepts in fold "+(f+1)+" above threshold max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(numInst*NOISE/100.0)+ ", minNoise="+minN+" we remove groups of size "+Math.max((int)Math.ceil(numInst*NOISE/100.0),minN));   
                        continue; //we skip constructive induction if we don't find any concepts
                    }
                
                    Instances trainFold= new Instances(data);   //for logical and All features
                    Instances testFold=new Instances(test);     //for logical and All features

                    Instances trainFoldNum= new Instances(data);    //for numerical features (e.g., dataset Credit score)
                    Instances testFoldNum=new Instances(test);      //for numerical features (e.g., dataset Credit score)

                    Instances trainFoldRE= new Instances(data); //for relational features
                    Instances testFoldRE=new Instances(test);   //for relational features

                    Instances trainFoldCP= new Instances(data); //for Cartesian product
                    Instances testFoldCP=new Instances(test);   //for Cartesian product

                    Instances trainFoldFU= new Instances(data); //for FURIA and thr
                    Instances testFoldFU=new Instances(test);   //for FURIA and thr
                    int numOfOrigAttr=data.numAttributes()-1;
                    unInfFeatures.clear();
                    //logical features
                    int tmp[];
                    int nC[]; //for counting Cartesian or relational or numerical features in tree

                    unInfFeatures.clear();
                    if(exhaustive){
                        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("dd. MM. yyyy HH:mm:ss");  
                        LocalDateTime now = LocalDateTime.now();  
                        System.out.println("Starting FC (exhaustive search, All features method): "+dtf.format(now)+" fold: "+(f+1)); 
                            logFile.println("Starting FC (exhaustive search, All features method): "+dtf.format(now)+" fold: "+(f+1));
                    }

            /**************** INTERACTION INFORMATION BY JAKULIN ******************/                  
                    if(jakulin){
                        boolean  allDiscrete=true;
                        for(int i=0;i<trainFoldCP.numAttributes();i++)
                            if(trainFoldCP.attribute(i).isNumeric()){   //check if attribute is numeric
                                allDiscrete=false; 
                                System.out.println("We found continuous attribute!");
                                break;
                            }

                        if(!allDiscrete){
                        //discretization    
                            weka.filters.supervised.attribute.Discretize filter;    //because of same class name in different packages
                            //setup filter
                            filter = new weka.filters.supervised.attribute.Discretize();
                            //Discretization is by Fayyad & Irani's MDL method (the default).
                            trainFoldCP.setClassIndex(trainFoldCP.numAttributes()-1); //we need class index for Fayyad & Irani's MDL
                            testFoldCP.setClassIndex(testFoldCP.numAttributes()-1);
                            filter.setInputFormat(trainFoldCP);
                            //apply filter
                            trainFoldCP = Filter.useFilter(trainFoldCP, filter);
                            testFoldCP = Filter.useFilter(testFoldCP, filter);  //we have to apply discretization on test dataset based on info from train dataset
                        }

                        List allCombSecOrdv2=allCombOfOrderN(listOfConcepts,2); //create groups for second ordered features
                        //calculates interaction information between all combinations of attributes - in some cases it can be too expensive
                        if(f==(folds-1)){
                            logFile.println("---------------------------------------------------------------------------------");
                            System.out.println("Number of all combinations: "+allCombSecOrdv2.size());
                                logFile.println("Number of all combinations: "+allCombSecOrdv2.size());
                        }

                        t1=new Timer();
                        t1.start();
                        List combInfInter=interInfoJakulin(trainFoldCP,allCombSecOrdv2, 4);    //we take 4 best interaction combinations, for more info see A. Jakulin. Machine learning based on attribute interactions. PhD thesis, University of Ljubljana, Faculty of Computer and Information Science, 2005.

                        System.out.println("Important combinations based on intraction information (Jakulin) in DESC order");
                        impGroups.println("*********************************** Fold "+(f+1)+" - list of combinations in DESC order ***********************************");
                        printAttrNamesIntInf(trainFoldCP, combInfInter);
                        unInfFeatures.clear();

                        if(!allDiscrete){
                            trainFoldCP=addCartFeat(data, trainFoldCP,combInfInter,false,f,2,false);    //Jakulin's method accepts all features
                            testFoldCP=addCartFeat(test, testFoldCP,combInfInter,false,f,2,false);      //Jakulin's method accepts all features
                            }
                        else{
                            trainFoldCP=addCartFeat(trainFoldCP,combInfInter,false,f,2,false);      //Jakulin's method accepts all features
                            testFoldCP=addCartFeat(testFoldCP,combInfInter,false,f,2,false);        //Jakulin's method accepts all features
                        }

                        t1.stop();
                        cartesianFCTime[f]=t1.diff();

                        if(cartesianFCTime[f]>timeLimit){ //10800000ms = 3h
                            System.out.println("Time exceeds the limit ("+timeLimit +" [ms]) of FC - Jakulin's method for one fold!"); 
                                logFile.println("Time exceeds the limit ("+timeLimit +" [ms]) of FC - Jakulin's method for one fold!");
                            break loopExhaustiveTooLong;
                        }

                        tmp= numOfFeat(trainFoldCP, data.numAttributes()-1);
                        numFeatByFoldsCP[f]=tmp[3];     //Cartesian

                        for(int c=0;c<clsTab.length;c++){
                            model=clsTab[c];
                            t1.start();
                            ma=evaluateModel(trainFoldCP,testFoldCP,model);
                            t1.stop();
                            accByFoldsCP[c][f]=ma.getAcc();
                            learnAllFCTimeCP[c][f]=t1.diff();
                            model=ma.getClassifier();
                            if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                j48=(J48)(model);
                                numOfTreeByFoldsCP[0][f]=(int)j48.measureTreeSize(); //treeSize
                                numOfTreeByFoldsCP[1][f]=(int)j48.measureNumLeaves(); //numOfLeave
                                numOfTreeByFoldsCP[2][f]=sumOfTermsInConstrInTree(trainFoldCP, data.numAttributes()-1, j48); //sumOfTerms
                                numOfTreeByFoldsCP[3][f]=(numOfTreeByFoldsCP[0][f]-numOfTreeByFoldsCP[1][f])==0 ? 0 : numOfTreeByFoldsCP[2][f]/(numOfTreeByFoldsCP[0][f]-numOfTreeByFoldsCP[1][f]); //sum of terms of constr DIV num of nodes
                                nC=numOfCartFeatInTree(trainFoldCP, data.numAttributes()-1, j48);
                                numOfCartesian[f]=nC[0]; //number of Cartesian features in tree
                                sumOfConstrCart[f]=nC[1]; //sum of constructs (Cartesian features) in tree
                            }
                            if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                FURIA fu=(FURIA)(model);
                                numOfRulesByFoldsCP[f]=fu.getRuleset().size(); //System.out.println("All features "+fu.getRuleset().size());
                                numOfTermsByFoldsCP[f]=sumOfTermsInConstrInRule(fu.getRuleset(),trainFoldCP);//System.out.println("All features "+countTermsOfConstructsFuria(fu.getRuleset(),trainFold));
                                numOfRatioByFoldsCP[f]=numOfRulesByFoldsCP[f]==0 ? 0 : (numOfTermsByFoldsCP[f]/numOfRulesByFoldsCP[f]);
                            }
                        }

                        if((f+1)==folds){
                            impGroups.println("---------------------------------------------------------------------------------");
                        }
                    }  
                    else{
                        /****************************************************************/ 
                        /**************** LOGICAL FEATURES ******************/    
                        t1=new Timer();
                        t1.start();    
                        int N2=2;
                        List allCombSecOrd=allCombOfOrderN(listOfConcepts,N2);  //create groups for second ordered features  
                        trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.AND, false, f, N2);   
                        testFold= addLogFeatDepth(data, testFold, allCombSecOrd,OperationLog.AND, false, f, N2);
                        
                        trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.OR, false, f, N2);        
                        testFold= addLogFeatDepth(data, testFold, allCombSecOrd,OperationLog.OR, false, f, N2);

                        unInfFeatures.clear();
                        int N3=3;
                        List allCombThirdOrd=allCombOfOrderN(listOfConcepts,N3);    //create groups for third ordered features  
                        trainFold= addLogFeatDepth(trainFold, allCombThirdOrd,OperationLog.AND, false, f, N3);        
                        testFold= addLogFeatDepth(data, testFold, allCombThirdOrd,OperationLog.AND, false, f, N3);

                        trainFold= addLogFeatDepth(trainFold, allCombThirdOrd,OperationLog.OR, false, f, N3);        
                        testFold= addLogFeatDepth(data, testFold, allCombThirdOrd,OperationLog.OR, false, f, N3);

                        trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.EQU, false, f, N2); 
                        trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.XOR, false, f, N2); 
                        trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.IMPL, false, f, N2); 

                        testFold= addLogFeatDepth(data, testFold, allCombSecOrd,OperationLog.EQU, false, f, N2);
                        testFold= addLogFeatDepth(data, testFold, allCombSecOrd,OperationLog.XOR, false, f, N2);
                        testFold= addLogFeatDepth(data, testFold, allCombSecOrd,OperationLog.IMPL, false, f, N2);

                        unInfFeatures.clear();
                        t1.stop();
                        allFCTimeLF[f]=t1.diff(); //time for constructing logical features
                        if(exhaustive){
                            System.out.println("FC ended (exhaustive search, All features method) fold: "+(f+1)+" time for FC: "+allFCTimeLF[f]); 
                            if(allFCTimeLF[f]>timeLimit){ //10800000ms = 3h
                                System.out.println("Time exceeds the limit ("+timeLimit +" [ms]) of FC for one fold!"); 
                                    logFile.println("Time exceeds the limit ("+timeLimit +" [ms]) of FC for one fold!");
                                break loopExhaustiveTooLong;
                            }
                        }

//                        if(!jakulin){
//                            attrImpListMDL.println("Feature evaluation: MDL (Logical features) - After CI");                
//                            mdlCORElearn(trainFold, rCaller, code);
//                        }
                        
                        tmp= numOfFeat(trainFold, data.numAttributes()-1);
                        numOfFeatByFoldsLF[f]=tmp[0];    //we count just logical features
                        
                        for(int c=0;c<clsTab.length;c++){
                            model=clsTab[c];
                            t1.start(); 
                            ma=evaluateModel(trainFold,testFold,model);
                            t1.stop();
                            accByFoldsLF[c][f]=ma.getAcc();
                            learnAllFCTimeLF[c][f]=t1.diff();
                            model=ma.getClassifier();
                            if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                j48=(J48)(model);
                                numOfTreeByFoldsLF[0][f]=(int)j48.measureTreeSize(); //treeSize
                                numOfTreeByFoldsLF[1][f]=(int)j48.measureNumLeaves(); //numOfLeaves
                                numOfTreeByFoldsLF[2][f]=sumOfTermsInConstrInTree(trainFold, data.numAttributes()-1, j48); //sumOfTerms
                                numOfTreeByFoldsLF[3][f]=(numOfTreeByFoldsLF[0][f]-numOfTreeByFoldsLF[1][f])==0 ? 0 : numOfTreeByFoldsLF[2][f]/(numOfTreeByFoldsLF[0][f]-numOfTreeByFoldsLF[1][f]); //sum of terms of constructs DIV num of nodes

                                numOfLogicalInTree[0][f]=numOfLogFeatInTree(trainFold, data.numAttributes()-1, j48);                            
                                numOfLogicalInTree[1][f]=sumOfLFTermsInConstrInTree(trainFold, data.numAttributes()-1, j48);
                            }
                            if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                FURIA fu=(FURIA)(model);
                                numOfRulesByFoldsLF[f]=fu.getRuleset().size();
                                numOfTermsByFoldsLF[f]=sumOfTermsInConstrInRule(fu.getRuleset(),trainFold);
                                numOfRatioByFoldsLF[f]=numOfRulesByFoldsLF[f]==0 ? 0 : (numOfTermsByFoldsLF[f]/numOfRulesByFoldsLF[f]);
                            }
                        }
                        /****************************************************************/   
                        /**************** NUMERICAL FEATURES ******************/
                        //numerical operators: /, -, +  ... e.g. for Credit score dataset
                        if(numerFeat){
                            t1.start();
                            trainFoldNum=addNumFeat(trainFoldNum, OperationNum.DIVIDE, allCombSecOrd);
                            testFoldNum=addNumFeat(testFoldNum, OperationNum.DIVIDE, allCombSecOrd);

                            trainFoldNum=addNumFeat(trainFoldNum, OperationNum.SUBTRACT, allCombSecOrd);
                            testFoldNum=addNumFeat(testFoldNum, OperationNum.SUBTRACT, allCombSecOrd);  

                            trainFoldNum=addNumFeat(trainFoldNum, OperationNum.ADD, allCombSecOrd);
                            testFoldNum=addNumFeat(testFoldNum, OperationNum.ADD, allCombSecOrd);  
                            t1.stop();
                            numericalFCTime[f]=t1.diff();
                            
//                            if(!jakulin){
//                                attrImpListMDL.println("Feature evaluation: MDL (Numerical features) - After CI");                
//                                mdlCORElearn(trainFoldNum, rCaller, code);
//                            }

                            tmp= numOfFeat(trainFoldNum, data.numAttributes()-1);
                            numFeatByFoldsNum[f]=tmp[5]; //numerical

                            for(int c=0;c<clsTab.length;c++){
                                model=clsTab[c];
                                t1.start();
                                ma=evaluateModel(trainFoldNum,testFoldNum,model);
                                t1.stop();
                                accByFoldsNum[c][f]=ma.getAcc();
                                learnAllFCTimeNum[c][f]=t1.diff();
                                model=ma.getClassifier();
                                if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                    j48=(J48)(model);
                                    numOfTreeByFoldsNum[0][f]=(int)j48.measureTreeSize(); //treeSize
                                    numOfTreeByFoldsNum[1][f]=(int)j48.measureNumLeaves(); //numOfLeave
                                    numOfTreeByFoldsNum[2][f]=sumOfTermsInConstrInTree(trainFoldNum, data.numAttributes()-1, j48); //sumOfTerms
                                    numOfTreeByFoldsNum[3][f]=(numOfTreeByFoldsNum[0][f]-numOfTreeByFoldsNum[1][f])==0 ? 0 : numOfTreeByFoldsNum[2][f]/(numOfTreeByFoldsNum[0][f]-numOfTreeByFoldsNum[1][f]); //sum of terms of constructs DIV num of nodes
                                    nC=numOfNumFeatInTree(trainFoldNum, data.numAttributes()-1, j48);
                                    numOfNumerical[f]=nC[0]; //number of numerical features in tree
                                    sumOfConstrNum[f]=nC[1]; //sum of constructs (numerical features) in tree
                                }
                                if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                    FURIA fu=(FURIA)(model);
                                    numOfRulesByFoldsNum[f]=fu.getRuleset().size();
                                    numOfTermsByFoldsNum[f]=sumOfTermsInConstrInRule(fu.getRuleset(),trainFoldNum);
                                    numOfRatioByFoldsNum[f]=numOfRulesByFoldsNum[f]==0 ? 0 : (numOfTermsByFoldsNum[f]/numOfRulesByFoldsNum[f]);
                                }
                            }                    
                        } 
                        /****************************************************************/
                        /**************** RELATIONAL FEATURES ******************/    
                        t1.start();
                        trainFoldRE=addRelFeat(trainFoldRE,allCombSecOrd,OperationRel.LESSTHAN,true,f); //true ... we count and remove uninformative features
                        testFoldRE=addRelFeat(testFoldRE,allCombSecOrd,OperationRel.LESSTHAN,false,f);  //false .. we just skip uninformative features

                        trainFoldRE=addRelFeat(trainFoldRE,allCombSecOrd,OperationRel.DIFF,true,f); //true ... we count and remove uninformative features
                        testFoldRE=addRelFeat(testFoldRE,allCombSecOrd,OperationRel.DIFF,false,f);  //false .. we just skip uninformative features

                        t1.stop();
                        relationalFCTime[f]=t1.diff();
                        
//                        if(!jakulin){
//                            attrImpListMDL.println("Feature evaluation: MDL (Relational features) - After CI");                
//                            mdlCORElearn(trainFoldRE, rCaller, code);
//                        }

                        tmp= numOfFeat(trainFoldRE, data.numAttributes()-1);
                        numFeatByFoldsRE[f]=tmp[4]; //relational

                        for(int c=0;c<clsTab.length;c++){
                            model=clsTab[c];
                            t1.start();
                            ma=evaluateModel(trainFoldRE,testFoldRE,model);
                            t1.stop();
                            accByFoldsRE[c][f]=ma.getAcc();
                            learnAllFCTimeRE[c][f]=t1.diff();
                            model=ma.getClassifier();
                            if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                j48=(J48)(model);
                                numOfTreeByFoldsRE[0][f]=(int)j48.measureTreeSize(); //treeSize
                                numOfTreeByFoldsRE[1][f]=(int)j48.measureNumLeaves(); //numOfLeave
                                numOfTreeByFoldsRE[2][f]=sumOfTermsInConstrInTree(trainFoldRE, data.numAttributes()-1, j48); //sumOfTerms
                                numOfTreeByFoldsRE[3][f]=(numOfTreeByFoldsRE[0][f]-numOfTreeByFoldsRE[1][f])==0 ? 0 : numOfTreeByFoldsRE[2][f]/(numOfTreeByFoldsRE[0][f]-numOfTreeByFoldsRE[1][f]); //sum of terms of constr DIV num of nodes
                                nC=numOfRelFeatInTree(trainFoldRE, data.numAttributes()-1, j48);
                                numOfRelational[f]=nC[0]; //number of relational features in tree
                                sumOfConstrRel[f]=nC[1]; //sum of constructs (relational features) in tree
                            }
                            if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                FURIA fu=(FURIA)(model);
                                numOfRulesByFoldsRE[f]=fu.getRuleset().size();
                                numOfTermsByFoldsRE[f]=sumOfTermsInConstrInRule(fu.getRuleset(),trainFoldRE);
                                numOfRatioByFoldsRE[f]=numOfRulesByFoldsRE[f]==0 ? 0 : (numOfTermsByFoldsRE[f]/numOfRulesByFoldsRE[f]);
                            }
                        }                
                        /****************************************************************/
                        /**************** CARTESIAN PRODUCT ******************/ 
                        boolean  allDiscrete=true;
                        for(int i=0;i<trainFoldCP.numAttributes();i++)
                            if(trainFoldCP.attribute(i).isNumeric()){    //check if attribute is numeric
                                allDiscrete=false; 
                                System.out.println("We found continuous attribute!");
                                break;
                            }

                        t1.start();
                        if(!allDiscrete){
                        //discretization    
                            weka.filters.supervised.attribute.Discretize filter;    //because of same class name in different packages
                            // setup filter
                            filter = new weka.filters.supervised.attribute.Discretize();
                            //Discretization is by Fayyad & Irani's MDL method (the default).
                            //filter.
                            trainFoldCP.setClassIndex(trainFoldCP.numAttributes()-1); //we need class index for Fayyad & Irani's MDL
                            testFoldCP.setClassIndex(testFoldCP.numAttributes()-1);
                            //filter.setUseBinNumbers(true); //eg BXofY ... B1of1
                            filter.setInputFormat(trainFoldCP);
                            //apply filter
                            trainFoldCP = Filter.useFilter(trainFoldCP, filter);
                            testFoldCP = Filter.useFilter(testFoldCP, filter); //we have to apply discretization on test dataset based on info from train dataset
                        }
                        //N2=2;
                        allCombSecOrd=allCombOfOrderN(listOfConcepts,N2); //create groups for second ordered features  
                        unInfFeatures.clear();//just for any case ... it should be empty
                        if(!allDiscrete){
                            trainFoldCP=addCartFeat(data, trainFoldCP,allCombSecOrd,false,f,2,true);
                            testFoldCP=addCartFeat(test, testFoldCP,allCombSecOrd,false,f,2,false); //we don't apply uninformative features
                            }
                        else{
                            trainFoldCP=addCartFeat(trainFoldCP,allCombSecOrd,false,f,2,true);
                            testFoldCP=addCartFeat(testFoldCP,allCombSecOrd,false,f,2,false); //we don't apply uninformative features
                        }

                        t1.stop();
                        cartesianFCTime[f]=t1.diff();
                        
//                        if(!jakulin){
//                            attrImpListMDL.println("Feature evaluation: MDL (Cartesian features) - After CI");                
//                            mdlCORElearn(trainFoldCP, rCaller, code);
//                        }

                        tmp= numOfFeat(trainFoldCP, data.numAttributes()-1);
                        numFeatByFoldsCP[f]=tmp[3]; //Cartesian

                        for(int c=0;c<clsTab.length;c++){
                            model=clsTab[c];
                            t1.start();
                            ma=evaluateModel(trainFoldCP,testFoldCP,model);
                            t1.stop();
                            accByFoldsCP[c][f]=ma.getAcc();
                            learnAllFCTimeCP[c][f]=t1.diff();
                            model=ma.getClassifier();
                            if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                j48=(J48)(model);
                                numOfTreeByFoldsCP[0][f]=(int)j48.measureTreeSize(); //treeSize
                                numOfTreeByFoldsCP[1][f]=(int)j48.measureNumLeaves(); //numOfLeave
                                numOfTreeByFoldsCP[2][f]=sumOfTermsInConstrInTree(trainFoldCP, data.numAttributes()-1, j48); //sumOfTerms
                                numOfTreeByFoldsCP[3][f]=(numOfTreeByFoldsCP[0][f]-numOfTreeByFoldsCP[1][f])==0 ? 0 : numOfTreeByFoldsCP[2][f]/(numOfTreeByFoldsCP[0][f]-numOfTreeByFoldsCP[1][f]); //sum of terms of constr DIV num of nodes
                                nC=numOfCartFeatInTree(trainFoldCP, data.numAttributes()-1, j48);
                                numOfCartesian[f]=nC[0]; //number of Cartesian features in tree
                                sumOfConstrCart[f]=nC[1]; //sum of constructs (Cartesian features) in tree
                            }
                            if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                FURIA fu=(FURIA)(model);
                                numOfRulesByFoldsCP[f]=fu.getRuleset().size();
                                numOfTermsByFoldsCP[f]=sumOfTermsInConstrInRule(fu.getRuleset(),trainFoldCP);
                                numOfRatioByFoldsCP[f]=numOfRulesByFoldsCP[f]==0 ? 0 : (numOfTermsByFoldsCP[f]/numOfRulesByFoldsCP[f]);
                                }
                        }        
                        /****************************************************************/                   
                        /**************** FURIA AND THRESHOLD FEATURES ******************/                                    
                        t1.start();
                        List<String> listOfFeat;
                        listOfFeat=genFeatFromFuria(trainFoldFU, (ArrayList<String>) listOfConcepts, classToExplain, cf, pci,covering, featFromExplClass);

                        trainFoldFU=addFeatures(trainFoldFU, (ArrayList<String>) listOfFeat); //add features from Furia
                        testFoldFU=addFeatures(testFoldFU, (ArrayList<String>) listOfFeat); //add features from Furia

                        //num-of-N features ... we are counting true conditions from rules
                        trainFoldFU=addFeatNumOfN(trainFoldFU, (ArrayList<String>) listOfFeat); //add num-Of-N features for evaluation
                        testFoldFU=addFeatNumOfN(testFoldFU, (ArrayList<String>) listOfFeat); //add num-Of-N features for evaluation
                        t1.stop();
                        furiaThrTime[f]=t1.diff();

//                        if(!jakulin){
//                            attrImpListMDL.println("Feature evaluation: MDL (FURIA and threshold features) - After CI");
//                            mdlCORElearn(trainFoldFU, rCaller, code);
//                        }

                        tmp= numOfFeat(trainFoldFU, data.numAttributes()-1);
                        numFeatByFoldsFuriaThr[0][f]=tmp[1]; //threshold
                        numFeatByFoldsFuriaThr[1][f]=tmp[2]; //FURIA

                        for(int c=0;c<clsTab.length;c++){
                            model=clsTab[c];
                            t1.start();
                            ma=evaluateModel(trainFoldFU,testFoldFU,model);
                            t1.stop();
                            accuracyByFoldsFuriaThr[c][f]=ma.getAcc();
                            learnFuriaThrTime[c][f]=t1.diff(); 

                            model=ma.getClassifier();
                            if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                j48=(J48)(model);
                                numTreeByFoldsFuriaThr[0][f]=(int)j48.measureTreeSize(); //treeSize
                                numTreeByFoldsFuriaThr[1][f]=(int)j48.measureNumLeaves(); //numOfLeaves
                                numTreeByFoldsFuriaThr[2][f]=sumOfTermsInConstrInTree(trainFoldFU, data.numAttributes()-1, j48); //sumOfTerms
                                numTreeByFoldsFuriaThr[3][f]=(numTreeByFoldsFuriaThr[0][f]-numTreeByFoldsFuriaThr[1][f])==0 ? 0 : numTreeByFoldsFuriaThr[2][f]/(numTreeByFoldsFuriaThr[0][f]-numTreeByFoldsFuriaThr[1][f]);

                                int furiaThrC[]=numOfDrThrFeatInTree(trainFoldFU, data.numAttributes()-1, j48);

                                numOfFuriaThrInTreeByFoldsF[0][f]=furiaThrC[0];
                                numOfFuriaThrInTreeByFoldsF[1][f]=furiaThrC[1];
                                numOfFuriaThrInTreeByFoldsF[2][f]=furiaThrC[2];
                                numOfFuriaThrInTreeByFoldsF[3][f]=furiaThrC[3];
                            }
                            if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                FURIA fu=(FURIA)(model);
                                complexityOfFuria[0][f]=fu.getRuleset().size();
                                complexityOfFuria[1][f]=sumOfTermsInConstrInRule(fu.getRuleset(),trainFoldFU);
                                complexityOfFuria[2][f]=complexityOfFuria[0][f]==0 ? 0 : (complexityOfFuria[1][f]/complexityOfFuria[0][f]);
                            }
                        }              
                        /****************************************************************/ 
                        /**************** ALL FEATURES ******************/
                        /*Merge Logical, Furia and Thr features*/
                        /*get everything except class from Logical features train fold*/
                        Remove remove= new Remove();
                        remove.setAttributeIndices("last");
                        remove.setInputFormat(trainFold);
                        trainFold = Filter.useFilter(trainFold, remove); 

                        remove.setAttributeIndices("last");
                        remove.setInputFormat(testFold);
                        testFold = Filter.useFilter(testFold, remove); 

                        /*get all features from numerical features, without class -  train fold*/
                        if(numerFeat){
                            if(!(numOfOrigAttr==trainFoldNum.numAttributes()-1)){ //if we don't get any feature from numerical then we skip merge with numerical feat
                                remove.setAttributeIndices((numOfOrigAttr+1)+"-"+(trainFoldNum.numAttributes()-1));
                                remove.setInvertSelection(true);
                                remove.setInputFormat(trainFoldNum);
                                trainFoldNum = Filter.useFilter(trainFoldNum, remove); 

                                remove.setAttributeIndices((numOfOrigAttr+1)+"-"+(testFoldNum.numAttributes()-1));
                                remove.setInvertSelection(true);
                                remove.setInputFormat(testFoldNum);
                                testFoldNum = Filter.useFilter(testFoldNum, remove); 
                                /*merge logical features and Cartesian product features*/
                                trainFold=Instances.mergeInstances(trainFold,trainFoldNum);              
                                testFold=Instances.mergeInstances(testFold,testFoldNum);
                            }                                
                        }    

                        /*get all features from relational features, without class -  train fold*/
                        if(!(numOfOrigAttr==trainFoldRE.numAttributes()-1)){ //if we don't get any feature from Relational then we skip merge with relational feat
                            remove.setAttributeIndices((numOfOrigAttr+1)+"-"+(trainFoldRE.numAttributes()-1));
                            remove.setInvertSelection(true);
                            remove.setInputFormat(trainFoldRE);
                            trainFoldRE = Filter.useFilter(trainFoldRE, remove); 

                            remove.setAttributeIndices((numOfOrigAttr+1)+"-"+(testFoldRE.numAttributes()-1));
                            remove.setInvertSelection(true);
                            remove.setInputFormat(testFoldRE);
                            testFoldRE = Filter.useFilter(testFoldRE, remove); 
                            /*merge logical features and Cartesian product features*/
                            trainFold=Instances.mergeInstances(trainFold,trainFoldRE);              
                            testFold=Instances.mergeInstances(testFold,testFoldRE);
                        }

                        /*get all features from Cartesian product features, without class -  train fold*/
                        if(!(numOfOrigAttr==trainFoldCP.numAttributes()-1)){ //if we don't get any feature from CP then we skip merge with Cartesian
                            remove.setAttributeIndices((numOfOrigAttr+1)+"-"+(trainFoldCP.numAttributes()-1));
                            remove.setInvertSelection(true);
                            remove.setInputFormat(trainFoldCP);
                            trainFoldCP = Filter.useFilter(trainFoldCP, remove); 

                            remove.setAttributeIndices((numOfOrigAttr+1)+"-"+(testFoldCP.numAttributes()-1));
                            remove.setInvertSelection(true);
                            remove.setInputFormat(testFoldCP);
                            testFoldCP = Filter.useFilter(testFoldCP, remove); 
                            /*merge logical features and Cartesian product features*/
                            trainFold=Instances.mergeInstances(trainFold,trainFoldCP);              
                            testFold=Instances.mergeInstances(testFold,testFoldCP);
                        }

                        /*get just features+class from Furia and thr fold*/
                        remove.setAttributeIndices((numOfOrigAttr+1)+"-last");
                        remove.setInvertSelection(true);
                        remove.setInputFormat(trainFoldFU);
                        trainFoldFU = Filter.useFilter(trainFoldFU, remove); 
                        remove.setAttributeIndices((numOfOrigAttr+1)+"-last");
                        remove.setInvertSelection(true);
                        remove.setInputFormat(testFoldFU);
                        testFoldFU = Filter.useFilter(testFoldFU, remove); 
                        trainFold=Instances.mergeInstances(trainFold,trainFoldFU);            
                        testFold=Instances.mergeInstances(testFold,testFoldFU);

                        trainFold.setClassIndex(trainFold.numAttributes()-1);
                        testFold.setClassIndex(testFold.numAttributes()-1);

                        allFCTime[f]=numerFeat ? (allFCTimeLF[f]+numericalFCTime[f]+relationalFCTime[f]+cartesianFCTime[f]+furiaThrTime[f]) : (allFCTimeLF[f]+relationalFCTime[f]+cartesianFCTime[f]+furiaThrTime[f]);
                        if(exhaustive){
                            System.out.println("FC time (exhaustive search): "+allFCTime[f]+" All features method, fold: "+(f+1));
                                logFile.println("FC time (exhaustive search): "+allFCTime[f]+" All features method, fold: "+(f+1));
                        }

                        if(!jakulin){
                            attrImpListMDL.println("Feature evaluation: MDL (All features dataset) - After CI");                
                            mdlCORElearn(trainFold, rCaller, code);
                        }
                        
                        tmp = numOfFeat(trainFold, data.numAttributes()-1);
                        numberOfFeatByFolds[0][f]=tmp[0];    //logical
                        numberOfFeatByFolds[1][f]=tmp[1];    //threshold
                        numberOfFeatByFolds[2][f]=tmp[2];    //decision rule (FURIA)
                        numberOfFeatByFolds[3][f]=tmp[3];    //Cartesian
                        numberOfFeatByFolds[4][f]=tmp[4];    //relational
                        numberOfFeatByFolds[5][f]=tmp[5];    //numerical

                        for(int c=0;c<clsTab.length;c++){
                            model=clsTab[c];
                            t1.start(); 
                                ma=evaluateModel(trainFold,testFold,model);
                            t1.stop();

                            accuracyByFolds[c][f]=ma.getAcc();
                            learnAllFCTime[c][f]=t1.diff();
                            model=ma.getClassifier();
                            if(excludeUppers(model.getClass().getSimpleName()).equals("J48")){
                                j48=(J48)(model);
                                numberOfTreeByFolds[0][f]=(int)j48.measureTreeSize(); //treeSize
                                numberOfTreeByFolds[1][f]=(int)j48.measureNumLeaves(); //numOfLeaves
                                numberOfTreeByFolds[2][f]=sumOfTermsInConstrInTree(trainFold, data.numAttributes()-1, j48); //sumOfTerms
                                numberOfTreeByFolds[3][f]=(numberOfTreeByFolds[0][f]-numberOfTreeByFolds[1][f])==0 ? 0 : numberOfTreeByFolds[2][f]/(numberOfTreeByFolds[0][f]-numberOfTreeByFolds[1][f]); //sum of terms of constr DIV num of nodes
                                numOfLogInTreeAll[f]=numOfLogFeatInTree(trainFold, data.numAttributes()-1, j48);
                                sumOfConstrLFAll[f]=sumOfLFTermsInConstrInTree(trainFold, data.numAttributes()-1, j48);

                                nC=numOfNumFeatInTree(trainFold, data.numAttributes()-1, j48);
                                numOfNumInTreeAll[f]=nC[0];
                                sumOfConstrNumAll[f]=nC[1];

                                nC=numOfRelFeatInTree(trainFold, data.numAttributes()-1, j48);
                                numOfRelInTreeAll[f]=nC[0]; //number of relational feat in tree
                                sumOfConstrRelAll[f]=nC[1]; //sum of constructs (relational features) in tree

                                nC=numOfCartFeatInTree(trainFold, data.numAttributes()-1, j48);
                                numOfCartFAll[f]=nC[0]; //number of Cartesian features in tree
                                sumOfConstrCartAll[f]=nC[1]; //sum of constructs (Cartesian features) in tree

                                int furiaThrC[]=numOfDrThrFeatInTree(trainFold, data.numAttributes()-1, j48);
                                //if we don't get any constructs than we take 0 from originally initialized dataset - we don't take value from original dataset because there is no constructs
                                numOfFuriaThrInTreeByFolds[0][f]=furiaThrC[0];
                                numOfFuriaThrInTreeByFolds[1][f]=furiaThrC[1];
                                numOfFuriaThrInTreeByFolds[2][f]=furiaThrC[2];
                                numOfFuriaThrInTreeByFolds[3][f]=furiaThrC[3];
                            }

                            if(excludeUppers(model.getClass().getSimpleName()).equals("FURIA")){
                                FURIA fu=(FURIA)(model);
                                numOfRulesByFolds[f]=fu.getRuleset().size();
                                numOfTermsByFoldsF[f]=sumOfTermsInConstrInRule(fu.getRuleset(),trainFold);
                                numOfRatioByFoldsF[f]=numOfRulesByFolds[f]==0 ? 0 : (numOfTermsByFoldsF[f]/numOfRulesByFolds[f]);
                            }
                        }

                        if(!exhaustive){
                            if(f==0){
                                logFile.print("Feature selection on validation dataset");
                                bestParamPerFold.println("Feature selection on validation dataset");
                            }
                            int split=4;    //5 ... 80%:20%, 4 ... 75%25%, 3 ... 66%:33%

                            if(f==0)
                                switch(split){
                                    case 5: logFile.println("Percentage of split (subTrain:validation) 80%:20%"); break;
                                    case 4: logFile.println("Percentage of split (subTrain:validation) 75%:25%"); break;
                                    case 3: logFile.println("Percentage of split (subTrain:validation) 66%:33%"); break;
                                    default: System.out.println("Number of split is not right!!!"); logFile.println("Number of split is not right!!!");
                                }
                            for(int i=0;i<clsTab.length;i++){
                                ParamSearchEval pse;
                                pse=paramSearch(trainFold, testFold, clsTab[i],data.numAttributes()-1,split, rCaller, code);
                                accuracyByFoldsPS[i][f]=pse.getAcc();

                                //we need info for different classifiers
                                featByFoldsPS[0][f][i]=pse.getFeat()[0];    //logical
                                featByFoldsPS[1][f][i]=pse.getFeat()[1];    //threshold
                                featByFoldsPS[2][f][i]=pse.getFeat()[2];    //decision rule (FURIA)
                                featByFoldsPS[3][f][i]=pse.getFeat()[3];    //Cartesian
                                featByFoldsPS[4][f][i]=pse.getFeat()[4];    //relational
                                featByFoldsPS[5][f][i]=pse.getFeat()[5];    //numerical

                                paramSearchTime[i][f]=pse.getTime()[0];
                                paramSLearnT[i][f]=pse.getTime()[1];

                                if(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("J48")){ //0-tree size, 1-number of leaves, 3-sum of constructs
                                    numberOfTreeByFoldsPS[0][f]=pse.getTree()[0];
                                    numberOfTreeByFoldsPS[1][f]=pse.getTree()[1];
                                    numberOfTreeByFoldsPS[2][f]=pse.getTree()[2];
                                    numberOfTreeByFoldsPS[3][f]=numberOfTreeByFoldsPS[2][f]/(numberOfTreeByFoldsPS[0][f]-numberOfTreeByFoldsPS[1][f]);

                                    numLogFeatInTreeFS[0][f]=pse.getNumLogFeatInTree()[0];
                                    numLogFeatInTreeFS[1][f]=pse.getNumLogFeatInTree()[1];

                                    if(numerFeat){
                                        numNumFeatInTreeFS[0][f]=pse.getNumFeatInTree()[0];
                                        numNumFeatInTreeFS[1][f]=pse.getNumFeatInTree()[1];
                                    }

                                    numRelFeatInTreeFS[0][f]=pse.getRelFeatInTree()[0];
                                    numRelFeatInTreeFS[1][f]=pse.getRelFeatInTree()[1];

                                    numCartFeatInTreeFS[0][f]=pse.getCartFeatInTree()[0];
                                    numCartFeatInTreeFS[1][f]=pse.getCartFeatInTree()[1];

                                    numOfFuriaThrInTreeByFoldsP[0][f]=pse.getFuriaThrComplx()[0];
                                    numOfFuriaThrInTreeByFoldsP[1][f]=pse.getFuriaThrComplx()[1];
                                    numOfFuriaThrInTreeByFoldsP[2][f]=pse.getFuriaThrComplx()[2];
                                    numOfFuriaThrInTreeByFoldsP[3][f]=pse.getFuriaThrComplx()[3];
                                }
                                if(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")){
                                    complexityOfFuriaPS[0][f]=pse.getComplexityFuria()[0];
                                    complexityOfFuriaPS[1][f]=pse.getComplexityFuria()[1];
                                    complexityOfFuriaPS[2][f]=complexityOfFuriaPS[0][f]==0 ? 0 : (complexityOfFuriaPS[1][f]/complexityOfFuriaPS[0][f]);
                                }
                            }    
                        }
                    }   //Jakulin's method or EFC
                }   //end FOR loop for folds

                DecimalFormat df = new DecimalFormat("0.00");

                /**************** Write ACC by folds into files ******************/
                if(writeAccByFoldsInFile && (!visualisation && !justExplain && !exhaustive && !jakulin)){
                    for(int i=0;i<clsTab.length;i++){
                        accByFolds=new PrintWriter(new FileWriter(folderName+clsTab[i].getClass().getSimpleName()+"-byFolds-"+lg+".csv"));
                        if(numerFeat)
                            accByFolds.println("FoldNo;Base;Log;Num;Rel;Cart;DrThr;All;FS");
                        else
                            accByFolds.println("FoldNo;Base;Log;Rel;Cart;DrThr;All;FS");
                        for(int f=0;f<folds;f++){
                            if(numerFeat)
                                accByFolds.println("Fold"+(f+1)+";"+df.format(accOrigModelByFolds[i][f])+";"+df.format(accByFoldsLF[i][f])+";"+ df.format(accByFoldsNum[i][f])+";"+df.format(accByFoldsRE[i][f])+";"+df.format(accByFoldsCP[i][f])+";"+df.format(accuracyByFoldsFuriaThr[i][f])+";"+df.format(accuracyByFolds[i][f])+";"+df.format(accuracyByFoldsPS[i][f]));                    
                            else   
                                accByFolds.println("Fold"+(f+1)+";"+df.format(accOrigModelByFolds[i][f])+";"+df.format(accByFoldsLF[i][f])+";"+df.format(accByFoldsRE[i][f])+";"+df.format(accByFoldsCP[i][f])+";"+df.format(accuracyByFoldsFuriaThr[i][f])+";"+df.format(accuracyByFolds[i][f])+";"+df.format(accuracyByFoldsPS[i][f]));
                        }
                        accByFolds.close();    
                    }
                }

                /****************************************************************/
                if(!jakulin){
                    if(!exhaustive){
                        System.out.println("---------------------------------------------------------------------------------");
                            logFile.println("---------------------------------------------------------------------------------");      
                    
                        System.out.println("Avg. explanation time: "+df.format(mean(exlpTime))+" [ms] (stdev "+ df.format(Math.sqrt(var(exlpTime,mean(exlpTime))))+")");  
                            logFile.println("Avg. explanation time: "+df.format(mean(exlpTime))+" [ms] (stdev "+ df.format(Math.sqrt(var(exlpTime,mean(exlpTime))))+")");
                        System.out.println("Avg. number of instances that we explain: "+df.format(mean(numOfExplainedInst))+" (stdev "+df.format(Math.sqrt(var(numOfExplainedInst,mean(numOfExplainedInst))))+")");
                            logFile.println("Avg. number of instances that we explain: "+df.format(mean(numOfExplainedInst))+" (stdev "+df.format(Math.sqrt(var(numOfExplainedInst,mean(numOfExplainedInst))))+")");
                    }

                    System.out.println("---------------------------------------------------------------------------------");
                        logFile.println("---------------------------------------------------------------------------------"); 
                    if(!exhaustive){
                        if(treeSHAP){
                            System.out.println("Internal (during building) accuracy of explanation model: "+df.format(mean(accExplAlgInt))+" (stdev "+df.format(Math.sqrt(var(accExplAlgInt,mean(accExplAlgInt))))+") ACC on the test dataset: "+df.format(mean(accExplAlgTest))+" (stdev "+df.format(Math.sqrt(var(accExplAlgTest,mean(accExplAlgTest))))+")");
                                logFile.println("Internal (during building) accuracy of explanation model: "+df.format(mean(accExplAlgInt))+" (stdev "+df.format(Math.sqrt(var(accExplAlgInt,mean(accExplAlgInt))))+") ACC on the test dataset: "+df.format(mean(accExplAlgTest))+" (stdev "+df.format(Math.sqrt(var(accExplAlgTest,mean(accExplAlgTest))))+")"); 
                        }
                        else{
                            System.out.println("ACC on the test dataset: "+df.format(mean(accExplAlgTest))+" stdev "+df.format(Math.sqrt(var(accExplAlgTest,mean(accExplAlgTest))))+(excludeUppers(predictionModel.getClass().getSimpleName()).equals("RF")?" ACC RF OOB: "+df.format(mean(oobRF))+" stdev "+df.format(Math.sqrt(var(oobRF,mean(oobRF)))):""));
                                logFile.println("ACC on the test dataset: "+df.format(mean(accExplAlgTest))+" stdev "+df.format(Math.sqrt(var(accExplAlgTest,mean(accExplAlgTest))))+(excludeUppers(predictionModel.getClass().getSimpleName()).equals("RF")?" ACC RF OOB: "+df.format(mean(oobRF))+" stdev "+df.format(Math.sqrt(var(oobRF,mean(oobRF)))):""));
                        }
                    }
                    
                    if(!exhaustive){
                        System.out.println("---------------------------------------------------------------------------------");
                            logFile.println("---------------------------------------------------------------------------------");
                          
                        System.out.println("Avg. model building time: "+df.format(mean(modelBuildTime))+" [ms] (stdev "+ df.format(Math.sqrt(var(modelBuildTime,mean(modelBuildTime))))+")");
                            logFile.println("Avg. model building time: "+df.format(mean(modelBuildTime))+" [ms] (stdev "+ df.format(Math.sqrt(var(modelBuildTime,mean(modelBuildTime))))+")");
                    }
                    
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");      
                    System.out.println("Original dataset");
                        logFile.println("Original dataset");
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");    
                    System.out.println("Avg. tree size: "+df.format(mean(treeSize))+" (stdev "+df.format(Math.sqrt(var(treeSize,mean(treeSize))))+")"+" avg. number of leaves: "+df.format(mean(numOfLeaves))+" (stdev "+df.format(Math.sqrt(var(numOfLeaves,mean(numOfLeaves))))+")"+" avg. number of nodes: "+df.format(mean(sumOfTerms))+" (stdev "+df.format(Math.sqrt(var(sumOfTerms,mean(sumOfTerms))))+")");
                        logFile.println("Avg. tree size: "+df.format(mean(treeSize))+" (stdev "+df.format(Math.sqrt(var(treeSize,mean(treeSize))))+")"+" avg. number of leaves: "+df.format(mean(numOfLeaves))+" (stdev "+df.format(Math.sqrt(var(numOfLeaves,mean(numOfLeaves))))+")"+" avg. number of nodes: "+df.format(mean(sumOfTerms))+" (stdev "+df.format(Math.sqrt(var(sumOfTerms,mean(sumOfTerms))))+")");      
                    System.out.println("Avg. ruleset size: "+df.format(mean(numOfRules))+" (stdev "+df.format(Math.sqrt(var(numOfRules,mean(numOfRules))))+")"+" avg. number of attributes in rules: "+ df.format(mean(numOfTerms))+" (stdev "+df.format(Math.sqrt(var(numOfTerms,mean(numOfTerms))))+")"+" avg. number of attributes per rule: "+df.format(mean(numConstructsPerRule)) +" (stdev "+df.format(Math.sqrt(var(numConstructsPerRule,mean(numConstructsPerRule))))+")");
                        logFile.println("Avg. ruleset size: "+df.format(mean(numOfRules))+" (stdev "+df.format(Math.sqrt(var(numOfRules,mean(numOfRules))))+")"+" avg. number of attributes in rules: "+ df.format(mean(numOfTerms))+" (stdev "+df.format(Math.sqrt(var(numOfTerms,mean(numOfTerms))))+") avg. number of attributes per rule: "+df.format(mean(numConstructsPerRule)) +" (stdev "+df.format(Math.sqrt(var(numConstructsPerRule,mean(numConstructsPerRule))))+")");
                    System.out.println("-----ACC-----");
                        logFile.println("-----ACC-----"); 
                    for (int i=0;i<clsTab.length;i++){      
                        System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(accOrigModelByFolds[i]))+" (stdev "+ df.format(Math.sqrt(var(accOrigModelByFolds[i],mean(accOrigModelByFolds[i]))))+")");
                            logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(accOrigModelByFolds[i]))+" (stdev "+ df.format(Math.sqrt(var(accOrigModelByFolds[i],mean(accOrigModelByFolds[i]))))+")");
                     }   
                    System.out.println("-----Learning and testing time-----");
                        logFile.println("-----Learning and testing time-----");  
                    for (int i=0;i<clsTab.length;i++){  
                        System.out.println("Avg. learning time "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllTime[i]))+" [ms] (stdev "+df.format(Math.sqrt(var(learnAllTime[i],mean(learnAllTime[i]))))+" [ms])");
                            logFile.println("Avg. learning time "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllTime[i]))+" [ms] (stdev "+df.format(Math.sqrt(var(learnAllTime[i],mean(learnAllTime[i]))))+" [ms])");
                    }  
                }      

                if(!jakulin){
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");  
                    System.out.println("Only Logical features");
                        logFile.println("Only Logical features");
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");   

                    System.out.println("Number of logical feat: "+df.format(mean(numOfFeatByFoldsLF))+" (stdev "+ df.format(Math.sqrt(var(numOfFeatByFoldsLF,mean(numOfFeatByFoldsLF))))+")");
                        logFile.println("Number of logical feat: "+df.format(mean(numOfFeatByFoldsLF))+" (stdev "+ df.format(Math.sqrt(var(numOfFeatByFoldsLF,mean(numOfFeatByFoldsLF))))+")");
                    System.out.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsLF[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[0],mean(numOfTreeByFoldsLF[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsLF[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[1],mean(numOfTreeByFoldsLF[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsLF[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[2],mean(numOfTreeByFoldsLF[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsLF[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[3],mean(numOfTreeByFoldsLF[3]))))+")");           
                        logFile.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsLF[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[0],mean(numOfTreeByFoldsLF[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsLF[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[1],mean(numOfTreeByFoldsLF[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsLF[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[2],mean(numOfTreeByFoldsLF[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsLF[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsLF[3],mean(numOfTreeByFoldsLF[3]))))+")");            
                    System.out.println("Avg. num of logical feat in tree: "+df.format(mean(numOfLogicalInTree[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfLogicalInTree[0],mean(numOfLogicalInTree[0]))))+")"+" avg. sum of (logical) constructs: "+df.format(mean(numOfLogicalInTree[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfLogicalInTree[1],mean(numOfLogicalInTree[1]))))+")");
                        logFile.println("Avg. num of logical feat in tree: "+df.format(mean(numOfLogicalInTree[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfLogicalInTree[0],mean(numOfLogicalInTree[0]))))+")"+" avg. sum of (logical) constructs: "+df.format(mean(numOfLogicalInTree[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfLogicalInTree[1],mean(numOfLogicalInTree[1]))))+")");
                    System.out.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsLF))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsLF,mean(numOfRulesByFoldsLF))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsLF))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsLF,mean(numOfTermsByFoldsLF))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsLF)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsLF,mean(numOfRatioByFoldsLF))))+")");  
                        logFile.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsLF))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsLF,mean(numOfRulesByFoldsLF))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsLF))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsLF,mean(numOfTermsByFoldsLF))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsLF)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsLF,mean(numOfRatioByFoldsLF))))+")");

                    System.out.println("-----ACC-----");
                        logFile.println("-----ACC-----");           
                    for(int c=0;c<clsTab.length;c++){
                        System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsLF[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsLF[c],mean(accByFoldsLF[c]))))+")");
                            logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsLF[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsLF[c],mean(accByFoldsLF[c]))))+")");
                    }
                    System.out.println("-----Learning and testing time-----");
                        logFile.println("-----Learning and testing time-----");  
                    for (int i=0;i<clsTab.length;i++){    
                        System.out.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeLF[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeLF[i],mean(learnAllFCTimeLF[i]))))+")");
                            logFile.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeLF[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeLF[i],mean(learnAllFCTimeLF[i]))))+")");
                    }
                    System.out.println("-----Feature construction time-----");
                        logFile.println("-----Feature construction time-----");  
                    System.out.println("Avg. FC time (all feat): "+df.format(mean(allFCTimeLF))+" [ms] stdev "+ df.format(Math.sqrt(var(allFCTimeLF,mean(allFCTimeLF)))));
                        logFile.println("Avg. FC time (all feat): "+df.format(mean(allFCTimeLF))+" [ms] stdev "+ df.format(Math.sqrt(var(allFCTimeLF,mean(allFCTimeLF)))));    
                }

                if(!jakulin && numerFeat){  
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");  
                    System.out.println("Numerical features");
                        logFile.println("Numerical features");    
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");   

                    System.out.println("Number of numerical feat: "+df.format(mean(numFeatByFoldsNum))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsNum,mean(numFeatByFoldsNum))))+")");
                        logFile.println("Number of numerical feat: "+df.format(mean(numFeatByFoldsNum))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsNum,mean(numFeatByFoldsNum))))+")");        
                    System.out.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsNum[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[0],mean(numOfTreeByFoldsNum[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsNum[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[1],mean(numOfTreeByFoldsNum[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsNum[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[2],mean(numOfTreeByFoldsNum[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsNum[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[3],mean(numOfTreeByFoldsNum[3]))))+")");           
                        logFile.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsNum[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[0],mean(numOfTreeByFoldsNum[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsNum[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[1],mean(numOfTreeByFoldsNum[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsNum[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[2],mean(numOfTreeByFoldsNum[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsNum[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsNum[3],mean(numOfTreeByFoldsNum[3]))))+")");        
                    System.out.println("Avg. num of numerical feat in tree: "+df.format(mean(numOfNumerical))+" (stdev "+ df.format(Math.sqrt(var(numOfNumerical,mean(numOfNumerical))))+")"+" avg. sum of (only numerical) constructs (in tree): "+ df.format(mean(sumOfConstrNum))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrNum,mean(sumOfConstrNum))))+")");
                        logFile.println("Avg. num of numerical feat in tree: "+df.format(mean(numOfNumerical))+" (stdev "+ df.format(Math.sqrt(var(numOfNumerical,mean(numOfNumerical))))+")"+" avg. sum of (only numerical) constructs (in tree): "+ df.format(mean(sumOfConstrNum))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrNum,mean(sumOfConstrNum))))+")");    
                    System.out.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsNum))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsNum,mean(numOfRulesByFoldsNum))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsNum))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsNum,mean(numOfTermsByFoldsNum))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsNum)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsNum,mean(numOfRatioByFoldsNum))))+")");  
                        logFile.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsNum))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsNum,mean(numOfRulesByFoldsNum))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsNum))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsNum,mean(numOfTermsByFoldsNum))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsNum)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsNum,mean(numOfRatioByFoldsNum))))+")"); 

                    System.out.println("-----ACC-----");
                        logFile.println("-----ACC-----");           
                    for(int c=0;c<clsTab.length;c++){            
                        System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsNum[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsNum[c],mean(accByFoldsNum[c]))))+")");
                            logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsNum[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsNum[c],mean(accByFoldsNum[c]))))+")");
                    }
                    System.out.println("-----Learning and testing time-----");
                        logFile.println("-----Learning and testing time-----");  
                    for (int i=0;i<clsTab.length;i++){    
                        System.out.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeNum[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeNum[i],mean(learnAllFCTimeNum[i]))))+")");
                            logFile.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeNum[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeNum[i],mean(learnAllFCTimeNum[i]))))+")");
                    }
                    System.out.println("-----Feature construction time-----");
                        logFile.println("-----Feature construction time-----");  
                    System.out.println("Avg. FC time (all feat): "+df.format(mean(numericalFCTime))+" [ms] stdev "+ df.format(Math.sqrt(var(numericalFCTime,mean(numericalFCTime)))));
                        logFile.println("Avg. FC time (all feat): "+df.format(mean(numericalFCTime))+" [ms] stdev "+ df.format(Math.sqrt(var(numericalFCTime,mean(numericalFCTime)))));   
                }

                if(!jakulin){  
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");  
                    System.out.println("Relational features");
                        logFile.println("Relational features");    
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");   

                    System.out.println("Number of relational feat: "+df.format(mean(numFeatByFoldsRE))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsRE,mean(numFeatByFoldsRE))))+")");
                        logFile.println("Number of relational feat: "+df.format(mean(numFeatByFoldsRE))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsRE,mean(numFeatByFoldsRE))))+")");        
                    System.out.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsRE[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[0],mean(numOfTreeByFoldsRE[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsRE[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[1],mean(numOfTreeByFoldsRE[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsRE[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[2],mean(numOfTreeByFoldsRE[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsRE[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[3],mean(numOfTreeByFoldsRE[3]))))+")");           
                        logFile.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsRE[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[0],mean(numOfTreeByFoldsRE[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsRE[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[1],mean(numOfTreeByFoldsRE[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsRE[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[2],mean(numOfTreeByFoldsRE[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsRE[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsRE[3],mean(numOfTreeByFoldsRE[3]))))+")");        
                    System.out.println("Avg. num of relational feat in tree: "+df.format(mean(numOfRelational))+" (stdev "+ df.format(Math.sqrt(var(numOfRelational,mean(numOfRelational))))+")"+" avg. sum of (only Cartesian) constructs (in tree): "+ df.format(mean(sumOfConstrRel))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrRel,mean(sumOfConstrRel))))+")");
                        logFile.println("Avg. num of relational feat in tree: "+df.format(mean(numOfRelational))+" (stdev "+ df.format(Math.sqrt(var(numOfRelational,mean(numOfRelational))))+")"+" avg. sum of (only Cartesian) constructs (in tree): "+ df.format(mean(sumOfConstrRel))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrRel,mean(sumOfConstrRel))))+")");    
                    System.out.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsRE))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsRE,mean(numOfRulesByFoldsRE))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsRE))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsRE,mean(numOfTermsByFoldsRE))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsRE)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsRE,mean(numOfRatioByFoldsRE))))+")");  
                        logFile.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsRE))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsRE,mean(numOfRulesByFoldsRE))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsRE))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsRE,mean(numOfTermsByFoldsRE))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsRE)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsRE,mean(numOfRatioByFoldsRE))))+")"); 

                    System.out.println("-----ACC-----");
                        logFile.println("-----ACC-----");           
                    for(int c=0;c<clsTab.length;c++){            
                        System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsRE[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsRE[c],mean(accByFoldsRE[c]))))+")");
                            logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsRE[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsRE[c],mean(accByFoldsRE[c]))))+")");
                    }
                    System.out.println("-----Learning and testing time-----");
                        logFile.println("-----Learning and testing time-----");  
                    for (int i=0;i<clsTab.length;i++){    
                        System.out.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeRE[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeRE[i],mean(learnAllFCTimeRE[i]))))+")");
                            logFile.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeRE[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeRE[i],mean(learnAllFCTimeRE[i]))))+")");
                    }
                    System.out.println("-----Feature construction time-----");
                        logFile.println("-----Feature construction time-----");  
                    System.out.println("Avg. FC time (all feat): "+df.format(mean(relationalFCTime))+" [ms] stdev "+ df.format(Math.sqrt(var(relationalFCTime,mean(relationalFCTime)))));
                        logFile.println("Avg. FC time (all feat): "+df.format(mean(relationalFCTime))+" [ms] stdev "+ df.format(Math.sqrt(var(relationalFCTime,mean(relationalFCTime)))));   
                }        

                System.out.println("*********************************************************************************");
                      logFile.println("*********************************************************************************"); 
                
                if(!jakulin){  
                    System.out.println("Cartesian product");
                        logFile.println("Cartesian product");    
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");  
                }
                else{
                    System.out.println("Jakulin's interaction information");
                        logFile.println("Jakulin's interaction information");    
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");      
                }

                System.out.println("Number of \"Cartesian\" feat: "+df.format(mean(numFeatByFoldsCP))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsCP,mean(numFeatByFoldsCP))))+")");
                    logFile.println("Number of \"Cartesian\" feat: "+df.format(mean(numFeatByFoldsCP))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsCP,mean(numFeatByFoldsCP))))+")");        
                System.out.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsCP[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[0],mean(numOfTreeByFoldsCP[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsCP[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[1],mean(numOfTreeByFoldsCP[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsCP[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[2],mean(numOfTreeByFoldsCP[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsCP[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[3],mean(numOfTreeByFoldsCP[3]))))+")");           
                    logFile.println("Avg. tree size (nodes+leaves): "+df.format(mean(numOfTreeByFoldsCP[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[0],mean(numOfTreeByFoldsCP[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numOfTreeByFoldsCP[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[1],mean(numOfTreeByFoldsCP[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numOfTreeByFoldsCP[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[2],mean(numOfTreeByFoldsCP[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numOfTreeByFoldsCP[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfTreeByFoldsCP[3],mean(numOfTreeByFoldsCP[3]))))+")");        
                System.out.println("Avg. num of \"Cartesian\" feat in tree: "+df.format(mean(numOfCartesian))+" (stdev "+ df.format(Math.sqrt(var(numOfCartesian,mean(numOfCartesian))))+")"+" avg. sum of (only Cartesian) constructs (in tree): "+ df.format(mean(sumOfConstrCart))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrCart,mean(sumOfConstrCart))))+")");
                    logFile.println("Avg. num of \"Cartesian\" feat in tree: "+df.format(mean(numOfCartesian))+" (stdev "+ df.format(Math.sqrt(var(numOfCartesian,mean(numOfCartesian))))+")"+" avg. sum of (only Cartesian) constructs (in tree): "+ df.format(mean(sumOfConstrCart))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrCart,mean(sumOfConstrCart))))+")");    
                System.out.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsCP))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsCP,mean(numOfRulesByFoldsCP))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsCP))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsCP,mean(numOfTermsByFoldsCP))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsCP)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsCP,mean(numOfRatioByFoldsCP))))+")");  
                    logFile.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFoldsCP))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFoldsCP,mean(numOfRulesByFoldsCP))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsCP))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsCP,mean(numOfTermsByFoldsCP))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsCP)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsCP,mean(numOfRatioByFoldsCP))))+")"); 

                System.out.println("-----ACC-----");
                    logFile.println("-----ACC-----");           
                for(int c=0;c<clsTab.length;c++){            
                    System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsCP[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsCP[c],mean(accByFoldsCP[c]))))+")");
                        logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accByFoldsCP[c]))+" (stdev "+ df.format(Math.sqrt(var(accByFoldsCP[c],mean(accByFoldsCP[c]))))+")");
                }
                System.out.println("-----Learning and testing time-----");
                    logFile.println("-----Learning and testing time-----");  
                for (int i=0;i<clsTab.length;i++){    
                    System.out.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeCP[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeCP[i],mean(learnAllFCTimeCP[i]))))+")");
                        logFile.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTimeCP[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTimeCP[i],mean(learnAllFCTimeCP[i]))))+")");
                }
                System.out.println("-----Feature construction time-----");
                    logFile.println("-----Feature construction time-----");  
                System.out.println("Avg. FC time (all feat): "+df.format(mean(cartesianFCTime))+" [ms] stdev "+ df.format(Math.sqrt(var(cartesianFCTime,mean(cartesianFCTime)))));
                    logFile.println("Avg. FC time (all feat): "+df.format(mean(cartesianFCTime))+" [ms] stdev "+ df.format(Math.sqrt(var(cartesianFCTime,mean(cartesianFCTime)))));   

                if(!jakulin){      
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");        
                    System.out.println("Only Furia and THR feat");
                        logFile.println("Only Furia and THR feat");
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");   
                    System.out.println("Number of FURIA feat: "+ df.format(mean(numFeatByFoldsFuriaThr[1]))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsFuriaThr[1],mean(numFeatByFoldsFuriaThr[1]))))+")"+" number of thr. feat: "+ df.format(mean(numFeatByFoldsFuriaThr[0]))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsFuriaThr[0],mean(numFeatByFoldsFuriaThr[0]))))+")");
                        logFile.println("Number of FURIA feat: "+ df.format(mean(numFeatByFoldsFuriaThr[1]))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsFuriaThr[1],mean(numFeatByFoldsFuriaThr[1]))))+")"+" number of thr. feat: "+ df.format(mean(numFeatByFoldsFuriaThr[0]))+" (stdev "+ df.format(Math.sqrt(var(numFeatByFoldsFuriaThr[0],mean(numFeatByFoldsFuriaThr[0]))))+")");
                    System.out.println("Avg. tree size (nodes+leaves): "+df.format(mean(numTreeByFoldsFuriaThr[0]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[0],mean(numTreeByFoldsFuriaThr[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numTreeByFoldsFuriaThr[1]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[1],mean(numTreeByFoldsFuriaThr[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numTreeByFoldsFuriaThr[2]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[2],mean(numTreeByFoldsFuriaThr[2]))))+") avg. sum of constructs / num of nodes: "+df.format(mean(numTreeByFoldsFuriaThr[3]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[3],mean(numTreeByFoldsFuriaThr[3]))))+")");
                        logFile.println("Avg. tree size (nodes+leaves): "+df.format(mean(numTreeByFoldsFuriaThr[0]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[0],mean(numTreeByFoldsFuriaThr[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numTreeByFoldsFuriaThr[1]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[1],mean(numTreeByFoldsFuriaThr[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numTreeByFoldsFuriaThr[2]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[2],mean(numTreeByFoldsFuriaThr[2]))))+") avg. sum of constructs / num of nodes: "+df.format(mean(numTreeByFoldsFuriaThr[3]))+" (stdev "+ df.format(Math.sqrt(var(numTreeByFoldsFuriaThr[3],mean(numTreeByFoldsFuriaThr[3]))))+")");
                    System.out.println("Avg. num of FURIA feat. in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsF[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[0],mean(numOfFuriaThrInTreeByFoldsF[0]))))+")"+" avg. sum of terms in constructs (Furia feat) in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsF[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[1],mean(numOfFuriaThrInTreeByFoldsF[1]))))+")"+" avg. num of THR feat. in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsF[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[2],mean(numOfFuriaThrInTreeByFoldsF[2]))))+")"+" avg. sum of terms in constructs (THR feat) in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsF[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[3],mean(numOfFuriaThrInTreeByFoldsF[3]))))+")");
                        logFile.println("Avg. num of FURIA feat. in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsF[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[0],mean(numOfFuriaThrInTreeByFoldsF[0]))))+")"+" avg. sum of terms in constructs (Furia feat) in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsF[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[1],mean(numOfFuriaThrInTreeByFoldsF[1]))))+")"+" avg. num of THR feat. in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsF[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[2],mean(numOfFuriaThrInTreeByFoldsF[2]))))+")"+" avg. sum of terms in constructs (THR feat) in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsF[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsF[3],mean(numOfFuriaThrInTreeByFoldsF[3]))))+")");
                    System.out.println("Avg. ruleset size: "+df.format(mean(complexityOfFuria[0]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuria[0],mean(complexityOfFuria[0]))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(complexityOfFuria[1]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuria[1],mean(complexityOfFuria[1]))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(complexityOfFuria[2])) +" (stdev "+df.format(Math.sqrt(var(complexityOfFuria[2],mean(complexityOfFuria[2]))))+")");
                        logFile.println("Avg. ruleset size: "+df.format(mean(complexityOfFuria[0]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuria[0],mean(complexityOfFuria[0]))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(complexityOfFuria[1]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuria[1],mean(complexityOfFuria[1]))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(complexityOfFuria[2])) +" (stdev "+df.format(Math.sqrt(var(complexityOfFuria[2],mean(complexityOfFuria[2]))))+")");    

                    System.out.println("-----ACC-----");
                        logFile.println("-----ACC-----"); 
                    for(int c=0;c<clsTab.length;c++){     
                        System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accuracyByFoldsFuriaThr[c]))+" (stdev "+ df.format(Math.sqrt(var(accuracyByFoldsFuriaThr[c],mean(accuracyByFoldsFuriaThr[c]))))+")");
                            logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accuracyByFoldsFuriaThr[c]))+" (stdev "+ df.format(Math.sqrt(var(accuracyByFoldsFuriaThr[c],mean(accuracyByFoldsFuriaThr[c]))))+")");
                    }    		    
                    System.out.println("-----Learning and testing time-----");
                        logFile.println("-----Learning and testing time-----"); 
                    for(int c=0;c<clsTab.length;c++){
                        System.out.println("Avg. learning time from FC(only Furia and THR feat) "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" "+df.format(mean(learnFuriaThrTime[c]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnFuriaThrTime[c],mean(learnFuriaThrTime[c]))))+")");
                            logFile.println("Avg. learning time from FC(only Furia and THR feat) "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" "+df.format(mean(learnFuriaThrTime[c]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnFuriaThrTime[c],mean(learnFuriaThrTime[c]))))+")");
                    }
                    System.out.println("-----Feature construction time-----");
                        logFile.println("-----Feature construction time-----");  
                    System.out.println("Avg. FC time (only Furia and THR feat): "+df.format(mean(furiaThrTime))+" [ms] stdev "+ df.format(Math.sqrt(var(furiaThrTime,mean(furiaThrTime)))));
                        logFile.println("Avg. FC time (only Furia and THR feat): "+df.format(mean(furiaThrTime))+" [ms] stdev "+ df.format(Math.sqrt(var(furiaThrTime,mean(furiaThrTime)))));

                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");       
                    System.out.println("All features dataset");
                        logFile.println("All features dataset");
                    System.out.println("*********************************************************************************");
                        logFile.println("*********************************************************************************");                  
                    System.out.println("Avg. number of groups for feature construction: "+df.format(mean(numOfGroupsOfFeatConstr))+" (stdev "+ df.format(Math.sqrt(var(numOfGroupsOfFeatConstr,mean(numOfGroupsOfFeatConstr))))+")");
                        logFile.println("Avg. number of groups for feature construction: "+df.format(mean(numOfGroupsOfFeatConstr))+" (stdev "+ df.format(Math.sqrt(var(numOfGroupsOfFeatConstr,mean(numOfGroupsOfFeatConstr))))+")");

                    System.out.println("Avg. size of groups (num. of candidate attr.) per folds (unm of attr. in all groups): "+df.format(mean(avgTermsPerFold))+" (stdev "+ df.format(Math.sqrt(var(avgTermsPerFold,mean(avgTermsPerFold))))+")");
                        logFile.println("Avg. size of groups (num. of candidate attr.) per folds (unm of attr. in all groups):: "+df.format(mean(avgTermsPerFold))+" (stdev "+ df.format(Math.sqrt(var(avgTermsPerFold,mean(avgTermsPerFold))))+")");      
                    System.out.println("Avg. size of groups (num. of candidate attr.) per groups per folds (avg. of avg.): "+df.format(mean(avgTermsPerGroup))+" (stdev "+ df.format(Math.sqrt(var(avgTermsPerGroup,mean(avgTermsPerGroup))))+")");
                        logFile.println("Avg. size of groups (num. of candidate attr.) per groups per folds (avg. of avg.): "+df.format(mean(avgTermsPerGroup))+" (stdev "+ df.format(Math.sqrt(var(avgTermsPerGroup,mean(avgTermsPerGroup))))+")");    
                    System.out.println("Max avg. group (length) of constructs (num of attr.): "+df.format(mean(maxGroupOfConstructs))+" (stdev "+ df.format(Math.sqrt(var(maxGroupOfConstructs,mean(maxGroupOfConstructs))))+")");
                        logFile.println("Max avg. group (length) of constructs (num of attr.): "+df.format(mean(maxGroupOfConstructs))+" (stdev "+ df.format(Math.sqrt(var(maxGroupOfConstructs,mean(maxGroupOfConstructs))))+")");     

                    System.out.println("Number of logical feat: "+df.format(mean(numberOfFeatByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[0],mean(numberOfFeatByFolds[0]))))+")"+(numerFeat ?(" number of numerical feat: "+df.format(mean(numberOfFeatByFolds[5]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[5],mean(numberOfFeatByFolds[5]))))+")"):"")+" number of relational feat: "+df.format(mean(numberOfFeatByFolds[4]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[4],mean(numberOfFeatByFolds[4]))))+")"+" number of \"Cartesian\" feat: "+df.format(mean(numberOfFeatByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[3],mean(numberOfFeatByFolds[3]))))+")"+" number of FURIA feat: "+ df.format(mean(numberOfFeatByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[2],mean(numberOfFeatByFolds[2]))))+")"+" number of thr. feat: "+ df.format(mean(numberOfFeatByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[1],mean(numberOfFeatByFolds[1]))))+")");            
                        logFile.println("Number of logical feat: "+df.format(mean(numberOfFeatByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[0],mean(numberOfFeatByFolds[0]))))+")"+(numerFeat ?(" number of numerical feat: "+df.format(mean(numberOfFeatByFolds[5]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[5],mean(numberOfFeatByFolds[5]))))+")"):"")+" number of relational feat: "+df.format(mean(numberOfFeatByFolds[4]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[4],mean(numberOfFeatByFolds[4]))))+")"+" number of \"Cartesian\" feat: "+df.format(mean(numberOfFeatByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[3],mean(numberOfFeatByFolds[3]))))+")"+" number of FURIA feat: "+ df.format(mean(numberOfFeatByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[2],mean(numberOfFeatByFolds[2]))))+")"+" number of thr. feat: "+ df.format(mean(numberOfFeatByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfFeatByFolds[1],mean(numberOfFeatByFolds[1]))))+")");

                    System.out.println("Avg. tree size (nodes+leaves): "+df.format(mean(numberOfTreeByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[0],mean(numberOfTreeByFolds[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numberOfTreeByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[1],mean(numberOfTreeByFolds[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numberOfTreeByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[2],mean(numberOfTreeByFolds[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numberOfTreeByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[3],mean(numberOfTreeByFolds[3]))))+")");
                        logFile.println("Avg. tree size (nodes+leaves): "+df.format(mean(numberOfTreeByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[0],mean(numberOfTreeByFolds[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numberOfTreeByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[1],mean(numberOfTreeByFolds[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numberOfTreeByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[2],mean(numberOfTreeByFolds[2]))))+")"+ " avg. sum of constructs / num of nodes: "+df.format(mean(numberOfTreeByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFolds[3],mean(numberOfTreeByFolds[3]))))+")");

                    System.out.println("Avg. num of logical feat in tree: "+df.format(mean(numOfLogInTreeAll))+" (stdev "+ df.format(Math.sqrt(var(numOfLogInTreeAll,mean(numOfLogInTreeAll))))+")"+" avg. sum of (only logical) constructs: "+df.format(mean(sumOfConstrLFAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrLFAll,mean(sumOfConstrLFAll))))+")"+(numerFeat ? " avg. num of numerical feat in tree: "+df.format(mean(numOfNumInTreeAll))+" (stdev "+ df.format(Math.sqrt(var(numOfNumInTreeAll,mean(numOfNumInTreeAll))))+")"+" avg. sum of (only numerical) constructs: "+df.format(mean(sumOfConstrNumAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrNumAll,mean(sumOfConstrNumAll))))+")" :"")+" avg. num of relational feat in tree: "+df.format(mean(numOfRelInTreeAll))+" (stdev "+ df.format(Math.sqrt(var(numOfRelInTreeAll,mean(numOfRelInTreeAll))))+")"+" avg. sum of (only relational) constructs: "+df.format(mean(sumOfConstrRelAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrRelAll,mean(sumOfConstrRelAll))))+")"+" avg. num of \"Cartesian\" feat in tree: "+df.format(mean(numOfCartFAll))+" (stdev "+ df.format(Math.sqrt(var(numOfCartFAll,mean(numOfCartFAll))))+")"+" avg. sum of (Cartesian) constructs (in tree): "+df.format(mean(sumOfConstrCartAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrCartAll,mean(sumOfConstrCartAll))))+")"+" avg. num of FURIA feat. in tree: "+df.format(mean(numOfFuriaThrInTreeByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[0],mean(numOfFuriaThrInTreeByFolds[0]))))+")"+" avg. sum of terms in constructs (Furia feat) in tree: "+ df.format(mean(numOfFuriaThrInTreeByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[1],mean(numOfFuriaThrInTreeByFolds[1]))))+")"+" avg. num of THR feat. in tree: "+ df.format(mean(numOfFuriaThrInTreeByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[2],mean(numOfFuriaThrInTreeByFolds[2]))))+")"+" avg. sum of terms in constructs (THR feat) in tree: "+df.format(mean(numOfFuriaThrInTreeByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[3],mean(numOfFuriaThrInTreeByFolds[3]))))+")");        
                        logFile.println("Avg. num of logical feat in tree: "+df.format(mean(numOfLogInTreeAll))+" (stdev "+ df.format(Math.sqrt(var(numOfLogInTreeAll,mean(numOfLogInTreeAll))))+")"+" avg. sum of (only logical) constructs: "+df.format(mean(sumOfConstrLFAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrLFAll,mean(sumOfConstrLFAll))))+")"+(numerFeat ? " avg. num of numerical feat in tree: "+df.format(mean(numOfNumInTreeAll))+" (stdev "+ df.format(Math.sqrt(var(numOfNumInTreeAll,mean(numOfNumInTreeAll))))+")"+" avg. sum of (only numerical) constructs: "+df.format(mean(sumOfConstrNumAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrNumAll,mean(sumOfConstrNumAll))))+")" :"")+" avg. num of relational feat in tree: "+df.format(mean(numOfRelInTreeAll))+" (stdev "+ df.format(Math.sqrt(var(numOfRelInTreeAll,mean(numOfRelInTreeAll))))+")"+" avg. sum of (only relational) constructs: "+df.format(mean(sumOfConstrRelAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrRelAll,mean(sumOfConstrRelAll))))+")"+" avg. num of \"Cartesian\" feat in tree: "+df.format(mean(numOfCartFAll))+" (stdev "+ df.format(Math.sqrt(var(numOfCartFAll,mean(numOfCartFAll))))+")"+" avg. sum of (Cartesian) constructs (in tree): "+df.format(mean(sumOfConstrCartAll))+" (stdev "+ df.format(Math.sqrt(var(sumOfConstrCartAll,mean(sumOfConstrCartAll))))+")"+" avg. num of FURIA feat. in tree: "+df.format(mean(numOfFuriaThrInTreeByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[0],mean(numOfFuriaThrInTreeByFolds[0]))))+")"+" avg. sum of terms in constructs (Furia feat) in tree: "+ df.format(mean(numOfFuriaThrInTreeByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[1],mean(numOfFuriaThrInTreeByFolds[1]))))+")"+" avg. num of THR feat. in tree: "+ df.format(mean(numOfFuriaThrInTreeByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[2],mean(numOfFuriaThrInTreeByFolds[2]))))+")"+" avg. sum of terms in constructs (THR feat) in tree: "+df.format(mean(numOfFuriaThrInTreeByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFolds[3],mean(numOfFuriaThrInTreeByFolds[3]))))+")");

                    System.out.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFolds))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFolds,mean(numOfRulesByFolds))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsF))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsF,mean(numOfTermsByFoldsF))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsF)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsF,mean(numOfRatioByFoldsF))))+")"); 
                        logFile.println("Avg. ruleset size: "+df.format(mean(numOfRulesByFolds))+" (stdev "+ df.format(Math.sqrt(var(numOfRulesByFolds,mean(numOfRulesByFolds))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(numOfTermsByFoldsF))+" (stdev "+ df.format(Math.sqrt(var(numOfTermsByFoldsF,mean(numOfTermsByFoldsF))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(numOfRatioByFoldsF)) +" (stdev "+df.format(Math.sqrt(var(numOfRatioByFoldsF,mean(numOfRatioByFoldsF))))+")");  

                    System.out.println("Mean unimp. OR feat.: "+df.format(mean(numberOfUnImpFeatByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[0],mean(numberOfUnImpFeatByFolds[0]))))+")"+" mean unimp. EQU feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[1],mean(numberOfUnImpFeatByFolds[1]))))+")"+" mean unimp. XOR feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[2],mean(numberOfUnImpFeatByFolds[2]))))+")"+" mean unimp. IMPL feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[3],mean(numberOfUnImpFeatByFolds[3]))))+")"+" mean unimp. AND feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[4]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[4],mean(numberOfUnImpFeatByFolds[4]))))+")"+" mean unimp. LESSTHAN feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[5]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[5],mean(numberOfUnImpFeatByFolds[5]))))+")"+" mean unimp. DIFF feat.: "+df.format(mean(numberOfUnImpFeatByFolds[6]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[6],mean(numberOfUnImpFeatByFolds[6]))))+")"+" mean unimp. \"Cartesian\" feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[7]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[7],mean(numberOfUnImpFeatByFolds[7]))))+")");
                        logFile.println("Mean unimp. OR feat.: "+df.format(mean(numberOfUnImpFeatByFolds[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[0],mean(numberOfUnImpFeatByFolds[0]))))+")"+" mean unimp. EQU feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[1],mean(numberOfUnImpFeatByFolds[1]))))+")"+" mean unimp. XOR feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[2],mean(numberOfUnImpFeatByFolds[2]))))+")"+" mean unimp. IMPL feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[3],mean(numberOfUnImpFeatByFolds[3]))))+")"+" mean unimp. AND feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[4]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[4],mean(numberOfUnImpFeatByFolds[4]))))+")"+" mean unimp. LESSTHAN feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[5]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[5],mean(numberOfUnImpFeatByFolds[5]))))+")"+" mean unimp. DIFF feat.: "+df.format(mean(numberOfUnImpFeatByFolds[6]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[6],mean(numberOfUnImpFeatByFolds[6]))))+")"+" mean unimp. \"Cartesian\" feat.: "+ df.format(mean(numberOfUnImpFeatByFolds[7]))+" (stdev "+ df.format(Math.sqrt(var(numberOfUnImpFeatByFolds[7],mean(numberOfUnImpFeatByFolds[7]))))+")");

                    System.out.println("-----ACC-----");
                        logFile.println("-----ACC-----");    
                    for(int c=0;c<clsTab.length;c++){        
                        System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accuracyByFolds[c]))+" (stdev "+ df.format(Math.sqrt(var(accuracyByFolds[c],mean(accuracyByFolds[c]))))+")");
                            logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accuracyByFolds[c]))+" (stdev "+ df.format(Math.sqrt(var(accuracyByFolds[c],mean(accuracyByFolds[c]))))+")");
                    }
                    System.out.println("-----Learning and testing time-----");
                        logFile.println("-----Learning and testing time-----");  
                    for (int i=0;i<clsTab.length;i++){    
                        System.out.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTime[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTime[i],mean(learnAllFCTime[i]))))+")");
                            logFile.println("Avg. learning time from FC (all feat), for "+(excludeUppers(clsTab[i].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[i].getClass().getSimpleName()))+" \t"+df.format(mean(learnAllFCTime[i]))+" [ms] (stdev "+ df.format(Math.sqrt(var(learnAllFCTime[i],mean(learnAllFCTime[i]))))+")");
                    }
                    System.out.println("-----Feature construction time-----");
                        logFile.println("-----Feature construction time-----");  
                    System.out.println("Avg. FC time (all feat): "+df.format(mean(allFCTime))+" [ms] (stdev "+ df.format(Math.sqrt(var(allFCTime,mean(allFCTime))))+")");  
                        logFile.println("Avg. FC time (all feat): "+df.format(mean(allFCTime))+" [ms] (stdev "+ df.format(Math.sqrt(var(allFCTime,mean(allFCTime))))+")");     
                }    

                if(!jakulin){    
                    if(!exhaustive){
                        System.out.println("*********************************************************************************");
                            logFile.println("*********************************************************************************");        
                        System.out.println("FS on validation dataset - results");
                            logFile.println("FS on validation dataset - results");
                        System.out.println("*********************************************************************************");
                            logFile.println("*********************************************************************************"); 

                        for(int i=0;i<clsTab.length;i++){
                            System.out.println("When using "+(excludeUppers(clsTab[i].getClass().getSimpleName()))+", number of logical feat: "+df.format(mean(threeDtoTwoD(featByFoldsPS,i)[0]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[0],mean(threeDtoTwoD(featByFoldsPS,i)[0]))))+")"+(numerFeat ? (" number of numerical feat: "+df.format(mean(threeDtoTwoD(featByFoldsPS,i)[5]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[5],mean(threeDtoTwoD(featByFoldsPS,i)[5]))))+")"): "")+" number of relational feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[4]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[4],mean(threeDtoTwoD(featByFoldsPS,i)[4]))))+")"+" number of \"Cartesian\" feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[3]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[3],mean(threeDtoTwoD(featByFoldsPS,i)[3]))))+")"+" number of FURIA feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[2]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[2],mean(threeDtoTwoD(featByFoldsPS,i)[2]))))+")"+" number of thr. feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[1]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[1],mean(threeDtoTwoD(featByFoldsPS,i)[1]))))+")");
                                logFile.println("When using "+(excludeUppers(clsTab[i].getClass().getSimpleName()))+", number of logical feat: "+df.format(mean(threeDtoTwoD(featByFoldsPS,i)[0]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[0],mean(threeDtoTwoD(featByFoldsPS,i)[0]))))+")"+(numerFeat ? (" number of numerical feat: "+df.format(mean(threeDtoTwoD(featByFoldsPS,i)[5]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[5],mean(threeDtoTwoD(featByFoldsPS,i)[5]))))+")"): "")+" number of relational feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[4]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[4],mean(threeDtoTwoD(featByFoldsPS,i)[4]))))+")"+" number of \"Cartesian\" feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[3]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[3],mean(threeDtoTwoD(featByFoldsPS,i)[3]))))+")"+" number of FURIA feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[2]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[2],mean(threeDtoTwoD(featByFoldsPS,i)[2]))))+")"+" number of thr. feat: "+ df.format(mean(threeDtoTwoD(featByFoldsPS,i)[1]))+" (stdev "+ df.format(Math.sqrt(var(threeDtoTwoD(featByFoldsPS,i)[1],mean(threeDtoTwoD(featByFoldsPS,i)[1]))))+")");
                        }

                        System.out.println("Avg. tree size (nodes+leaves): "+df.format(mean(numberOfTreeByFoldsPS[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[0],mean(numberOfTreeByFoldsPS[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numberOfTreeByFoldsPS[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[1],mean(numberOfTreeByFoldsPS[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numberOfTreeByFoldsPS[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[2],mean(numberOfTreeByFoldsPS[2]))))+") avg. sum of constructs / num of nodes: "+df.format(mean(numberOfTreeByFoldsPS[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[3],mean(numberOfTreeByFoldsPS[3]))))+")");
                            logFile.println("Avg. tree size (nodes+leaves): "+df.format(mean(numberOfTreeByFoldsPS[0]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[0],mean(numberOfTreeByFoldsPS[0]))))+")"+" avg. number of leaves: "+ df.format(mean(numberOfTreeByFoldsPS[1]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[1],mean(numberOfTreeByFoldsPS[1]))))+")"+" avg. sum of constructs: "+ df.format(mean(numberOfTreeByFoldsPS[2]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[2],mean(numberOfTreeByFoldsPS[2]))))+") avg. sum of constructs / num of nodes: "+df.format(mean(numberOfTreeByFoldsPS[3]))+" (stdev "+ df.format(Math.sqrt(var(numberOfTreeByFoldsPS[3],mean(numberOfTreeByFoldsPS[3]))))+")");        

                        System.out.println("Avg. num of logical feat. in tree: "+df.format(mean(numLogFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numLogFeatInTreeFS[0],mean(numLogFeatInTreeFS[0]))))+")"+" avg. sum of (only logical) constructs in tree: "+df.format(mean(numLogFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numLogFeatInTreeFS[1],mean(numLogFeatInTreeFS[1]))))+")"+(numerFeat ? (" avg. num of numerical feat. in tree: "+df.format(mean(numNumFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numNumFeatInTreeFS[0],mean(numNumFeatInTreeFS[0]))))+")"+" avg. sum of (only numerical) constructs in tree: "+df.format(mean(numNumFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numNumFeatInTreeFS[1],mean(numNumFeatInTreeFS[1]))))+")") :"")+" avg. num of relational feat. in tree: "+df.format(mean(numRelFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numRelFeatInTreeFS[0],mean(numRelFeatInTreeFS[0]))))+")"+" avg. sum of (only relational) constructs in tree: "+df.format(mean(numRelFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numRelFeatInTreeFS[1],mean(numRelFeatInTreeFS[1]))))+")"+" avg. num of Cartesian feat. in tree: "+df.format(mean(numCartFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numCartFeatInTreeFS[0],mean(numCartFeatInTreeFS[0]))))+")"+" avg. sum of constructs (of Cartesian feat) in tree: "+df.format(mean(numCartFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numCartFeatInTreeFS[1],mean(numCartFeatInTreeFS[1]))))+")"+" avg. num of FURIA feat. in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsP[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[0],mean(numOfFuriaThrInTreeByFoldsP[0]))))+")"+" avg. sum of terms in constructs (Furia feat) in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsP[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[1],mean(numOfFuriaThrInTreeByFoldsP[1]))))+")"+" avg. num of THR feat. in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsP[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[2],mean(numOfFuriaThrInTreeByFoldsP[2]))))+")"+" avg. sum of terms in constructs (THR feat) in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsP[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[3],mean(numOfFuriaThrInTreeByFoldsP[3]))))+")");  
                            logFile.println("Avg. num of logical feat. in tree: "+df.format(mean(numLogFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numLogFeatInTreeFS[0],mean(numLogFeatInTreeFS[0]))))+")"+" avg. sum of (only logical) constructs in tree: "+df.format(mean(numLogFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numLogFeatInTreeFS[1],mean(numLogFeatInTreeFS[1]))))+")"+(numerFeat ? (" avg. num of numerical feat. in tree: "+df.format(mean(numNumFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numNumFeatInTreeFS[0],mean(numNumFeatInTreeFS[0]))))+")"+" avg. sum of (only numerical) constructs in tree: "+df.format(mean(numNumFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numNumFeatInTreeFS[1],mean(numNumFeatInTreeFS[1]))))+")") :"")+" avg. num of relational feat. in tree: "+df.format(mean(numRelFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numRelFeatInTreeFS[0],mean(numRelFeatInTreeFS[0]))))+")"+" avg. sum of (only relational) constructs in tree: "+df.format(mean(numRelFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numRelFeatInTreeFS[1],mean(numRelFeatInTreeFS[1]))))+")"+" avg. num of Cartesian feat. in tree: "+df.format(mean(numCartFeatInTreeFS[0]))+" (stdev "+ df.format(Math.sqrt(var(numCartFeatInTreeFS[0],mean(numCartFeatInTreeFS[0]))))+")"+" avg. sum of constructs (of Cartesian feat) in tree: "+df.format(mean(numCartFeatInTreeFS[1]))+" (stdev "+ df.format(Math.sqrt(var(numCartFeatInTreeFS[1],mean(numCartFeatInTreeFS[1]))))+")"+" avg. num of FURIA feat. in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsP[0]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[0],mean(numOfFuriaThrInTreeByFoldsP[0]))))+")"+" avg. sum of terms in constructs (Furia feat) in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsP[1]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[1],mean(numOfFuriaThrInTreeByFoldsP[1]))))+")"+" avg. num of THR feat. in tree: "+ df.format(mean(numOfFuriaThrInTreeByFoldsP[2]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[2],mean(numOfFuriaThrInTreeByFoldsP[2]))))+")"+" avg. sum of terms in constructs (THR feat) in tree: "+df.format(mean(numOfFuriaThrInTreeByFoldsP[3]))+" (stdev "+ df.format(Math.sqrt(var(numOfFuriaThrInTreeByFoldsP[3],mean(numOfFuriaThrInTreeByFoldsP[3]))))+")");

                        System.out.println("Avg. ruleset size: "+df.format(mean(complexityOfFuriaPS[0]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuriaPS[0],mean(complexityOfFuriaPS[0]))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(complexityOfFuriaPS[1]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuriaPS[1],mean(complexityOfFuriaPS[1]))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(complexityOfFuriaPS[2])) +" (stdev "+df.format(Math.sqrt(var(complexityOfFuriaPS[2],mean(complexityOfFuriaPS[2]))))+")");
                            logFile.println("Avg. ruleset size: "+df.format(mean(complexityOfFuriaPS[0]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuriaPS[0],mean(complexityOfFuriaPS[0]))))+")"+" avg. number of terms of construct in Furia feat.: "+ df.format(mean(complexityOfFuriaPS[1]))+" (stdev "+ df.format(Math.sqrt(var(complexityOfFuriaPS[1],mean(complexityOfFuriaPS[1]))))+") avg. number of terms in constructs per ruleset: "+df.format(mean(complexityOfFuriaPS[2])) +" (stdev "+df.format(Math.sqrt(var(complexityOfFuriaPS[2],mean(complexityOfFuriaPS[2]))))+")");   

                        System.out.println("-----ACC-----");
                            logFile.println("-----ACC-----"); 
                        for(int c=0;c<clsTab.length;c++){
                            System.out.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accuracyByFoldsPS[c]))+" (stdev "+ df.format(Math.sqrt(var(accuracyByFoldsPS[c],mean(accuracyByFoldsPS[c]))))+")");
                                logFile.println("Avg. class. ACC "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(accuracyByFoldsPS[c]))+" (stdev "+ df.format(Math.sqrt(var(accuracyByFoldsPS[c],mean(accuracyByFoldsPS[c]))))+")");
                        }

                        System.out.println("-----Search time-----");
                            logFile.println("-----Search time-----");  
                        for(int c=0;c<clsTab.length;c++){
                            System.out.println("Avg. search time "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(paramSearchTime[c]))+" (stdev "+ df.format(Math.sqrt(var(paramSearchTime[c],mean(paramSearchTime[c]))))+")");
                                logFile.println("Avg. search time "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(paramSearchTime[c]))+" (stdev "+ df.format(Math.sqrt(var(paramSearchTime[c],mean(paramSearchTime[c]))))+")");
                        }

                        System.out.println("-----Learning and testing time-----");
                            logFile.println("-----Learning and testing time-----");    
                        for(int c=0;c<clsTab.length;c++){
                            System.out.println("Avg. learn time "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(paramSLearnT[c]))+" (stdev "+ df.format(Math.sqrt(var(paramSLearnT[c],mean(paramSLearnT[c]))))+")");
                                logFile.println("Avg. learn time "+(excludeUppers(clsTab[c].getClass().getSimpleName()).equals("FURIA")?"FU":excludeUppers(clsTab[c].getClass().getSimpleName()))+" \t"+df.format(mean(paramSLearnT[c]))+" (stdev "+ df.format(Math.sqrt(var(paramSLearnT[c],mean(paramSLearnT[c]))))+")");
                        }
                    }
                }            

                tTotal.stop();
                if(!justExplain){
                    System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                        logFile.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"); 
                    System.out.println("Total processing time ("+fileName+"): "+tTotal.diff());
                        logFile.println("Total processing time ("+fileName+"): "+tTotal.diff());
                    System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                        logFile.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"); 
                }
            }
        }
        
        if(justExplain || (justExplain && visualisation)){
            impGroupsKD.close();
            attrImpListMDL_KD.close();     
            discIntervalsKD.close();              
        }
        else if (!visualisation || (!justExplain && !visualisation)){
            logFile.close();
            if(!visualisation && !exhaustive && !jakulin)
                bestParamPerFold.close();
            if(!treeSHAP)
                samplesStat.close();
            
            impGroups.close();
            if(!jakulin)
                attrImpListMDL.close();     
            discIntervals.close();  
        }
        if(groupsByThrStat && !visualisation && !justExplain &&!exhaustive && !jakulin)
            groupsStat.close();
        
//        rCaller.deleteTempFiles();
        rCaller.stopRCallerOnline();        
    }

    //for constructing TRAIN dataset of depth N
    public static Instances addLogFeatDepth(Instances data, List newTmpComb,OperationLog ol, boolean kononenko, int folds, int N) throws Exception{ //Discretization is by Fayyad & Irani's MDL method (the default).
        String attName="";
        Remove remove;
        Add filter;
        String attr1Val="";
        Enumeration<Object> atrValues=null;
        Instances newData=new Instances(data);
        Instances newBinAttr=null,allDiscrete=null;
        int tmp=0;
        int countUnInf=0;
        Set setB = new HashSet();       //for controlling names of generate combinations
        String tmpArr[], newTmpDisc[];	//indexes for combinations
	    
        for(int i=0;i<newTmpComb.size();i++){
            tmpArr=newTmpComb.get(i).toString().replace("[","").replace("]", "").trim().split(",");
            if(tmpArr.length < N) //this shouldn't never happen if number of attributes is less than the depth then there is no constructive induction
                continue;

            allDiscrete=null; //for each combination e.g., [3,4,7] we dereff. allDiscrete
            for(int j=0;j<tmpArr.length;j++){
                attr1Val="";
                if(newData.attribute(Integer.parseInt(tmpArr[j].trim())).isNominal()){
                    atrValues= newData.attribute(Integer.parseInt(tmpArr[j].trim())).enumerateValues();
                    while (atrValues.hasMoreElements())
                        attr1Val+=(String) atrValues.nextElement();
                    if(!((attr1Val.equals("01") || attr1Val.equals("10")) && newData.attributeStats(newData.attribute(Integer.parseInt(tmpArr[j].trim())).index()).distinctCount<=2)) 
                        newBinAttr=discretizeFI(data, Integer.parseInt(tmpArr[j].trim()),kononenko);
                    else{
                        remove= new Remove();
                        remove.setAttributeIndices((newData.attribute(Integer.parseInt(tmpArr[j].trim())).index()+1)+"");//rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1. e.g., first-3,5,6-last
                        remove.setInvertSelection(true);
                        remove.setInputFormat(newData);
                        newBinAttr = Filter.useFilter(newData, remove); //just one attribute
                    }
                }
                else if(newData.attribute(Integer.parseInt(tmpArr[j].trim())).isNumeric())
                    newBinAttr=discretizeFI(data, Integer.parseInt(tmpArr[j].trim()),kononenko); 
                if(j==0)
                    allDiscrete=new Instances(newBinAttr); //trick to initialize allDiscrete object
                else
                    allDiscrete=Instances.mergeInstances(allDiscrete,newBinAttr);   //we have now all discretized attributes, we don't have class attr.
            }
        
            newBinAttr=null;
            newTmpDisc=new String[allDiscrete.numAttributes()];
            for(int j=0;j<allDiscrete.numAttributes();j++)  //in allDiscrete we don't have class attribute
                newTmpDisc[j]=j+"";
        
            List newDiscComb=Arrays.asList(Generator.combination(newTmpDisc).simple(N).stream().toArray()); 
        
            String newTmpDiscName;
            for(int j=0;j<newDiscComb.size();j++){
                setB.clear();
                tmpArr=newDiscComb.get(j).toString().replace("[","").replace("]", "").trim().split(",");
                for(int k=0;k<tmpArr.length;k++){
                    newTmpDiscName=allDiscrete.attribute(Integer.parseInt(tmpArr[k].trim())).name();
                    if(newTmpDiscName.contains("=="))
                        setB.add(newTmpDiscName.split("==")[0]);    //get just original name of the attribute
                    else
                        setB.add(newTmpDiscName);
                }
                if(setB.size()<N)//this means that we have in combination at least two parts from the same attribute e.g. A1-B1ofB2, A1-B1ofB3
                    continue;
            
                attName=allDiscrete.attribute(Integer.parseInt(tmpArr[0].trim())).name();
                for(int k=1; k<N;k++)
                    attName+=" "+ ol.name().toLowerCase()+" "+allDiscrete.attribute(Integer.parseInt(tmpArr[k].trim())).name();	
            
                filter= new Add();
                filter.setAttributeIndex("" + (newData.numAttributes())); //parameter of the method must be String
                filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n
                filter.setNominalLabels("0,1");
                filter.setInputFormat(newData);
                newData = Filter.useFilter(newData, filter);
            
                double tmp1Attr,tmp2Attr;
                for(int m = 0; m < newData.numInstances(); m++){
                    tmp1Attr=allDiscrete.instance(m).value(Integer.parseInt(tmpArr[0].trim()));
                    tmp2Attr=allDiscrete.instance(m).value(Integer.parseInt(tmpArr[1].trim()));
                    tmp=computeOperationTwoOperand((int)tmp1Attr,ol,(int)tmp2Attr); //we take values from two tmp datasets that are discretized and
                    for(int l=2; l<N;l++){
                        tmp2Attr=allDiscrete.instance(m).value(Integer.parseInt(tmpArr[l].trim()));
                        tmp=computeOperationTwoOperand(tmp,ol,(int)tmp2Attr); //we take values from two tmp datasets that are discretized and binarized
                    }								
                    newData.instance(m).setValue(newData.numAttributes() - 2, tmp); //enriched dataset
                }
                if(newData.attributeStats(newData.attribute(attName).index()).distinctCount==1){
                    unInfFeatures.add(attName);
                    countUnInf++;
                    remove= new Remove();
                    remove.setAttributeIndices((newData.attribute(attName).index()+1)+"");//rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1. e.g., first-3,5,6-last
                    remove.setInputFormat(newData);
                    newData = Filter.useFilter(newData, remove);
                }
            }			
        }
	
        newBinAttr=null;
        allDiscrete=null;
        newTmpDisc=null;
        tmpArr=null;
        setB.clear();
        System.gc();
        
        return newData;
    }

    //for constructing TEST dataset of depth N
    public static Instances addLogFeatDepth(Instances train, Instances test, List newTmpComb,OperationLog ol, boolean kononenko, int folds, int N) throws Exception{ //Discretization is by Fayyad & Irani's MDL method (the default).
        String attName="";
        Remove remove;
        Add filter;
        String attr1Val="";
        Enumeration<Object> atrValues=null;
        Instances newData=new Instances(test);
        Instances newBinAttr=null,allDiscrete=null;
        int tmp=0;
        Set setB = new HashSet(); //for controlling names of generate combinations
        String tmpArr[], newTmpDisc[];	//indexes for combinations
	
        for(int i=0;i<newTmpComb.size();i++){       
            tmpArr=newTmpComb.get(i).toString().replace("[","").replace("]", "").trim().split(",");
            if(tmpArr.length < N) //this shouldn't never happen if number of attributes is less than the depth then there is no constructive induction
                continue;
            allDiscrete=null; //for each combination e.g., [3,4,7] we have to dereff. allDiscrete
            for(int j=0;j<tmpArr.length;j++){
                attr1Val="";
                if(newData.attribute(Integer.parseInt(tmpArr[j].trim())).isNominal()){
                    atrValues= newData.attribute(Integer.parseInt(tmpArr[j].trim())).enumerateValues();
                    while (atrValues.hasMoreElements())
                        attr1Val+=(String) atrValues.nextElement();
                    if(!((attr1Val.equals("01") || attr1Val.equals("10")) && newData.attributeStats(newData.attribute(Integer.parseInt(tmpArr[j].trim())).index()).distinctCount<=2)) 
                        newBinAttr=discretizeFITestBasedOnTrain(train, test,Integer.parseInt(tmpArr[j].trim()),kononenko); 
                    else{
                        remove= new Remove();
                        remove.setAttributeIndices((newData.attribute(Integer.parseInt(tmpArr[j].trim())).index()+1)+"");//rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1. e.g., first-3,5,6-last
                        remove.setInvertSelection(true);
                        remove.setInputFormat(newData);
                        newBinAttr = Filter.useFilter(newData, remove); //just one attribute
                    }
                }
                else if(newData.attribute(Integer.parseInt(tmpArr[j].trim())).isNumeric())
                        newBinAttr=discretizeFITestBasedOnTrain(train, test,Integer.parseInt(tmpArr[j].trim()),kononenko); 
                if(j==0)
                    allDiscrete=new Instances(newBinAttr); //trick to initialize allDiscrete object
                else
                    allDiscrete=Instances.mergeInstances(allDiscrete,newBinAttr);   //we have now all discretized attributes, we don't have class attr.
            }

            newTmpDisc=new String[allDiscrete.numAttributes()];
            for(int j=0;j<allDiscrete.numAttributes();j++)  //in allDiscrete we don't have class attribute
                newTmpDisc[j]=j+"";

            List newDiscComb=Arrays.asList(Generator.combination(newTmpDisc).simple(N).stream().toArray()); 

            String newTmpDiscName;
            for(int j=0;j<newDiscComb.size();j++){
                setB.clear();
                tmpArr=newDiscComb.get(j).toString().replace("[","").replace("]", "").trim().split(",");
                for(int k=0;k<tmpArr.length;k++){
                    newTmpDiscName=allDiscrete.attribute(Integer.parseInt(tmpArr[k].trim())).name();
                    if(newTmpDiscName.contains("=="))
                        setB.add(newTmpDiscName.split("==")[0]);    //get just original name of the attribute
                    else
                        setB.add(newTmpDiscName);
                }
                if(setB.size()<N)//this means that we have in combination at least two parts from the same attribute e.g. A1-B1ofB2, A1-B1ofB3
                    continue;

                attName=allDiscrete.attribute(Integer.parseInt(tmpArr[0].trim())).name();
                for(int k=1; k<N;k++)
                    attName+=" "+ ol.name().toLowerCase()+" "+allDiscrete.attribute(Integer.parseInt(tmpArr[k].trim())).name();	
                
                if(unInfFeatures.contains(attName)) //if feature is not informative and doesn't exist in train dataset
                    continue;

                filter= new Add();
                filter.setAttributeIndex("" + (newData.numAttributes())); //parameter of the method must be String
                filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n
                filter.setNominalLabels("0,1");
                filter.setInputFormat(newData);
                newData = Filter.useFilter(newData, filter);

                double tmp1Attr,tmp2Attr;
                for(int m = 0; m < newData.numInstances(); m++){
                    tmp1Attr=allDiscrete.instance(m).value(Integer.parseInt(tmpArr[0].trim()));
                    tmp2Attr=allDiscrete.instance(m).value(Integer.parseInt(tmpArr[1].trim()));
                    tmp=computeOperationTwoOperand((int)tmp1Attr,ol,(int)tmp2Attr); //we take values from two tmp datasets that are discretized and
                    for(int l=2; l<N;l++){
                        tmp2Attr=allDiscrete.instance(m).value(Integer.parseInt(tmpArr[l].trim()));
                        tmp=computeOperationTwoOperand(tmp,ol,(int)tmp2Attr); //we take values from two tmp datasets that are discretized and binarized
                    }
                    newData.instance(m).setValue(newData.numAttributes() - 2, tmp);    //enriched dataset
                }
           }				
        }				
        
        return newData;
    }  
    
    //add relational feature
    public static Instances addRelFeat(Instances data, List newTmpComb, OperationRel op, boolean train, int folds) throws Exception{ //Discretization is by Fayyad & Irani's MDL method (the default).
        // we need folds for saving info of unimportant features
        String attName="";
        Remove remove;
        Add filter;
        int countUnInf=0;
        String combName;
        Instances newData=new Instances(data);
        int idxAttr1,idxAttr2;
        int tmp;
        Set setA = new HashSet(); //for controlling attribute names
        Set setB = new HashSet(); //for controlling names of generate combinations
    
        for (int i=0;i<data.numAttributes()-1;i++)
            setA.add(data.attribute(i).name());
    
        for(int j=0;j<newTmpComb.size();j++){   //we get combinations in style [1,2]
            idxAttr1=Integer.parseInt(newTmpComb.get(j).toString().replace("[","").replace("]", "").trim().split(",")[0].trim());
            idxAttr2=Integer.parseInt(newTmpComb.get(j).toString().replace("[","").replace("]", "").trim().split(",")[1].trim());
            attName="";
            if(newData.attribute(idxAttr1).isNumeric() && newData.attribute(idxAttr2).isNumeric()){
                combName=newData.attribute(idxAttr1).name()+newData.attribute(idxAttr2).name();
                if(setB.contains(combName)) //if combination exists, we don't generate feature
                    continue;
                else
                    setB.add(combName); 

                attName=newData.attribute(idxAttr1).name()+" "+op.name()+" "+newData.attribute(idxAttr2).name();
                if(setA.contains(attName) || unInfFeatures.contains(attName)) //if attribute exists, we don't add it
                    continue;
                else
                    setA.add(attName); 

                filter= new Add();
                filter.setAttributeIndex("" + (newData.numAttributes()));
                filter.setAttributeName(attName);
                filter.setNominalLabels("0,1");
                filter.setInputFormat(newData);
                newData = Filter.useFilter(newData, filter);

                for(int m = 0; m < newData.numInstances(); m++){
                    tmp=computeRelOpTwoOperand(newData.instance(m).value(idxAttr1), op, newData.instance(m).value(idxAttr2));
                    newData.instance(m).setValue(newData.numAttributes() - 2, tmp);    //enriched dataset
                }
            }
            
            if(train){
            if(!attName.trim().equals(""))
                if(newData.attributeStats(newData.attribute(attName).index()).distinctCount==1){
                    unInfFeatures.add(attName);
                    countUnInf++;
                    remove= new Remove();
                    remove.setAttributeIndices((newData.attribute(attName).index()+1)+"");
                    remove.setInputFormat(newData);
                    newData = Filter.useFilter(newData, remove);
                }
            }         
        }

        if(train){
            if(op.name().equalsIgnoreCase("LESSTHAN"))
                numberOfUnImpFeatByFolds[5][folds]=countUnInf;
            if(op.name().equalsIgnoreCase("DIFF"))
                numberOfUnImpFeatByFolds[6][folds]=countUnInf;          
        }    
        
        return newData;
    }

    //generate Cartesian product
    public static Instances addCartFeat(Instances data, List newTmpComb,boolean kononenko, int folds, int N, boolean train) throws Exception{
        String attName="";
        Remove remove;
        Add filter;
        Instances newData=new Instances(data);
        String tmp;
        int countUnInf=0;
        String tmpArr[];	//indexes for combinations
	    
        for(int i=0;i<newTmpComb.size();i++){
            tmpArr=newTmpComb.get(i).toString().replace("[","").replace("]", "").trim().split(",");
            if(tmpArr.length < N) //this shouldn't never happen if number of attributes is less than the depth then there is no constructive induction
                continue;

            attName= newData.attribute(Integer.parseInt(tmpArr[0].trim())).name()+"_x_"+newData.attribute(Integer.parseInt(tmpArr[1].trim())).name();	
            if(!train){
                if(unInfFeatures.contains(attName)) //if feature is not informative and doesn't exist in train dataset
                    continue;
            }
            filter= new Add();
            filter.setAttributeIndex("" + (newData.numAttributes())); //parameter of the method must be String
            filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n
			
            String allDIscValues=genDiscValues(data,Integer.parseInt(tmpArr[0].trim()),Integer.parseInt(tmpArr[1].trim()) );
            filter.setNominalLabels(allDIscValues);
            filter.setInputFormat(newData);
            newData = Filter.useFilter(newData, filter);
            
            for(int m = 0; m < newData.numInstances(); m++){
                tmp=mergeValues(newData.instance(m).stringValue(Integer.parseInt(tmpArr[0].trim())),newData.instance(m).stringValue(Integer.parseInt(tmpArr[1].trim())));
                newData.instance(m).setValue(newData.numAttributes() - 2, tmp);    //enriched dataset
            }
        
            if(train){ //we don't remove uninfotmative features on test fold
                if(newData.attributeStats(newData.attribute(attName).index()).distinctCount==1){
                    unInfFeatures.add(attName);
                    countUnInf++;
                    remove= new Remove();
                    remove.setAttributeIndices((newData.attribute(attName).index()+1)+"");
                    remove.setInputFormat(newData);
                    newData = Filter.useFilter(newData, remove);
                }
            }
        }
        if(train)
            numberOfUnImpFeatByFolds[7][folds]=countUnInf;
        
        return newData;
    }

    //generate Cartesian product
    public static Instances addCartFeat(Instances origData, Instances discData,List newTmpComb,boolean kononenko, int folds, int N, boolean train) throws Exception{
        String attName="";
        Remove remove;
        Add filter;
        Instances newData=new Instances(origData);
        String tmp;
        int countUnInf=0;
        String tmpArr[];	//indexes for combinations
	    
        for(int i=0;i<newTmpComb.size();i++){
            tmpArr=newTmpComb.get(i).toString().replace("[","").replace("]", "").trim().split(",");
            if(tmpArr.length < N) //this shouldn't never happen if number of attributes is less than the depth then there is no constructive induction
                continue;

            attName= newData.attribute(Integer.parseInt(tmpArr[0].trim())).name()+"_x_"+newData.attribute(Integer.parseInt(tmpArr[1].trim())).name();
            if(!train){
                if(unInfFeatures.contains(attName)) //if feature is not informative and doesn't exist in train dataset
                    continue;
            }
            filter= new Add();
            filter.setAttributeIndex("" + (newData.numAttributes())); //parameter of the method must be String
            filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n
			
            String allDIscValues=genDiscValues(discData,Integer.parseInt(tmpArr[0].trim()),Integer.parseInt(tmpArr[1].trim()) );
            filter.setNominalLabels(allDIscValues);
            filter.setInputFormat(newData);
            newData = Filter.useFilter(newData, filter);
            
            for(int m = 0; m < newData.numInstances(); m++){
                //we take data from discretized dataset
                tmp=mergeValues(discData.instance(m).stringValue(Integer.parseInt(tmpArr[0].trim())),discData.instance(m).stringValue(Integer.parseInt(tmpArr[1].trim())));
                newData.instance(m).setValue(newData.numAttributes() - 2, tmp);    //enriched dataset
            }
            if(train){
                if(newData.attributeStats(newData.attribute(attName).index()).distinctCount==1){
                    unInfFeatures.add(attName);
                    countUnInf++;
                    remove= new Remove();
                    remove.setAttributeIndices((newData.attribute(attName).index()+1)+"");
                    remove.setInputFormat(newData);
                    newData = Filter.useFilter(newData, remove);
                }
            }
        }
        if(train)
            numberOfUnImpFeatByFolds[7][folds]=countUnInf;
        
        return newData;
    }

    //generateFeatFromFuriaNoOR
    public static List<String> genFeatFromFuria(Instances data,ArrayList<String> allComb, int c, double cfF, double cfI, boolean covering, boolean featFromExplClass){ //cfF=0.85 ... Furia, cfI=0.9 number of intances covered, c ... class to explain 
        List<String> list=null;
        ArrayList<String> oneConcept=new ArrayList<>();
        ArrayList<String> oneAttrConcepts=new ArrayList<>();
        HashSet<String> allFeatures = new HashSet<>();
        String mergedOrRule="",attName="";
        Instances newData= new Instances(data);
        int tmp;
        String classValue="";
        String nomValue;
        RemoveRange rr=new RemoveRange();
        String newTmp[];
        boolean ruleTrue=true;
        int distribucija[]=null;
        double pct = 0;
        int count;
        try{
        int nc=data.attributeStats(data.classIndex()).nominalCounts[c]; //number of instances from the explained class
        String className=data.attribute(data.classIndex()).name();  //we need class name for parsing
        String classValueExp=data.attribute(data.classIndex()).value(c);

        for(int j=0;j<allComb.size();j++){
            count=0;
            newTmp=allComb.get(j).split(",");
            String attrList="";

            for(int i=0;i<newTmp.length;i++)
                attrList+=(Integer.parseInt(newTmp[i])+1)+",";

            attrList+=(data.classIndex()+1);
            //use only attributes from concept
            Remove remove1= new Remove();
            remove1.setAttributeIndices(attrList); //we select attributes that we need in combination with setInvertSelection, otherwise we select attributes that we delete
            remove1.setInvertSelection(true);    //we need to remove not selected attributes - invert selection
            remove1.setInputFormat(data);

            Instances instNew = Filter.useFilter(data, remove1); //select only attributes that are in the model (inst - test dataset)
            instNew.setClassIndex(instNew.numAttributes()-1); //set class attribute 

            String ruleName[]=null;
            Classifier model;

            //using Furia
            FURIA fu=new FURIA();
            fu.setOptimizations(5); //number of optimization steps
            model=fu;
            model.buildClassifier(instNew);
            ArrayList<Rule> arrRule=new ArrayList<>();
            arrRule= fu.getRuleset();  
            ArrayList<String> newRules=new ArrayList<>();
            double tmpCF;
            ruleName=new String[arrRule.size()];

            for(int i = 0; i < ruleName.length; i++){
                tmpCF=Math.round(100.0 * ((FURIA.RipperRule) arrRule.get(i)).getConfidence())/ 100.0;
                ruleName[i] = ((FURIA.RipperRule)arrRule.get(i)).toString(instNew.classAttribute())+" (CF = "+ tmpCF + ")\n";

                if(ruleName[i].contains("???"))
                    ruleName[i]=ruleName[i].replaceAll("???", "-");

                if(featFromExplClass)
                    if(!ruleName[i].substring(ruleName[i].indexOf(className)+className.length()+1,ruleName[i].indexOf("(CF")).trim().equals(classValueExp) || tmpCF < cfF)
                        continue;

                if(!ruleName[i].contains(" and ")){ 
                    if(ruleName[i].contains("[-inf")){                  
                            if(ruleName[i].contains(",") && !ruleName[i].contains("inf, inf")){ //e.g., (VvLOX in [-inf, -inf, 0.182135, 0.349099])
                                ruleName[i]=ruleName[i].replace("in [-", "<= ");
                                String tabTmp[]=ruleName[i].split(",");
                                ruleName[i]=ruleName[i].replace("]", "").replace(")","").substring(0,ruleName[i].indexOf("<=")+3)+tabTmp[tabTmp.length-2].trim()+")";
                            }
                            if(ruleName[i].contains(",") && ruleName[i].contains("inf, inf")){  //e.g (VvAGPL in [-0.25942, -0.249411, inf, inf]) (VvGLC2 in [-0.343839, 3.317059, inf, inf])
                                ruleName[i]=ruleName[i].replace("in [", ">= ");
                                String tabTmp[]=ruleName[i].split(",");
                                ruleName[i]=ruleName[i].replace("]", "").replace(")","").substring(0,ruleName[i].indexOf(">=")+2)+tabTmp[0]+")";
                            }   
                    }

                    if(ruleName[i].contains("inf, inf")){           //e.g., (Cycle time in [64.7, 113.2, inf, inf]); Cycle time is attr.
                        ruleName[i]=ruleName[i].replace("in [", ">= ");
                            if(ruleName[i].contains(",")){
                                String tabTmp[]=ruleName[i].split(",");
                                    ruleName[i]=ruleName[i].replace("]", "").replace(")","").substring(0,ruleName[i].indexOf(">=")+2)+tabTmp[1]+")";
                            }
                    }      
                    oneConcept.add(ruleName[i].trim());
                }
                else
                    newRules.add(ruleName[i]);                       
            }

            for(int i=0;i<oneConcept.size();i++){
                if(oneConcept.get(i).contains(" => "))
                    mergedOrRule=oneConcept.get(i).split(" => ")[0].trim();
                else
                    mergedOrRule=oneConcept.get(i);
            oneAttrConcepts.add(mergedOrRule);
            }

            newRules.addAll(oneAttrConcepts);
            oneAttrConcepts.clear();
            oneConcept.clear();

            String attrNameRule;
            double attrValue;

            for (int i = 0; i < newRules.size(); i++){
                if(newRules.get(i).contains(" and ")){
                    attName=newRules.get(i).split(" => ")[0].trim(); //left side ... attributes
                    int idxStart=newRules.get(i).split(" => ")[1].trim().indexOf("=");
                    int idxEnd=newRules.get(i).split(" => ")[1].trim().indexOf("(");
                    classValue=newRules.get(i).split(" => ")[1].trim().substring(idxStart+1, idxEnd).trim(); //right side ... class value
                    pct=Double.parseDouble(newRules.get(i).split(" => ")[1].trim().substring(newRules.get(i).split(" => ")[1].trim().indexOf("CF")).split(" = ")[1].replace(")","").trim());
                    newRules.set(i,newRules.get(i).split(" => ")[0].trim()); //left side ... attributes

                    String allRules1[];
                    allRules1=newRules.get(i).split(" and ");
                    newRules.set(i,"");
                    for(int p=0;p<allRules1.length;p++){   
                        if(allRules1[p].contains("-inf, -inf")){
                            allRules1[p]=allRules1[p].replace("in [-inf", "<= ");
                            String tabTmp[]=allRules1[p].split(",");
                            allRules1[p]=allRules1[p].replace("]", "").replace(")","").substring(0,allRules1[p].indexOf("<=")+3)+tabTmp[tabTmp.length-2].replace("]", "").replace(")","").trim()+")";
                            }
                        else if(allRules1[p].contains("inf, inf")){                        
                            allRules1[p]=allRules1[p].replace("in [", ">= ");
                            String tabTmp[]=allRules1[p].split(",");
                            allRules1[p]=allRules1[p].replace("]", "").replace(")","").substring(0,allRules1[p].indexOf(">=")+2)+tabTmp[1]+")";
                        }                                                        
                            if(p==allRules1.length-1)
                                newRules.set(i,newRules.get(i)+allRules1[p].trim());
                            else
                                newRules.set(i,newRules.get(i)+allRules1[p].trim()+" and ");                                                
                     }                      
                    attName=newRules.get(i);

                }
                else
                    attName=newRules.get(i); //attribute name with just on "condition"

                if(!attName.equals(""))         
                    allFeatures.add(attName);

            if(covering){   //controling covering instances by features                    
                String attrTmpRule[] = null;
                    if(attName.contains(" and "))
                        attrTmpRule= attName.trim().split(" and "); //left side ... attributes e.g., (A3 = 1) and (A2 = 1) and (A1 = 0)
                    else{
                        attName=attName+"@";    
                        attrTmpRule=attName.trim().split("@");
                    }

                attName="";
                String remove="";

                for(int l = 0; l < newData.numInstances(); l++){
                    ruleTrue=true;
                    tmp=0;

                    for(int k=0;k<attrTmpRule.length;k++){  //parsing e.g., (A3 = 1) and (A2 = 1) and (A1 = 0) we parse attribute name and value
                            //AND
                        if(attrTmpRule[k].contains(" = ")){
                            attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" = ")[0].trim();
                            if(newData.attribute(attrNameRule).isNominal()){
                                nomValue=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" = ")[1].trim();  //nominal value as a string
                                if(!newData.instance(l).stringValue(newData.attribute(attrNameRule).index()).equals(nomValue)){
                                    ruleTrue=false;                               
                                    break;
                                }
                            }
                            else{
                                attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" = ")[1].trim());
                                if(newData.instance(l).value(newData.attribute(attrNameRule)) != attrValue){
                                    ruleTrue=false;                               
                                    break;
                                } 
                            }
                        }
                        else if (attrTmpRule[k].contains(" >= ")){    
                            attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" >= ")[0].trim();
                            attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" >= ")[1].trim()); //Exception in thread "main" java.lang.NumberFormatException: For input string: "st_sales_growth"
                            if(!(newData.instance(l).value(newData.attribute(attrNameRule)) >= attrValue)){
                                ruleTrue=false;                               
                                break;
                            }  
                        } 
                        else if (attrTmpRule[k].contains(" <= ")){    
                            attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" <= ")[0].trim();
                            attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" <= ")[1].trim());
                            if(!(newData.instance(l).value(newData.attribute(attrNameRule)) <= attrValue)){
                                ruleTrue=false;                               
                                break;
                            } 
                        }
                    }
                    if(ruleTrue)
                        remove+=(l+1)+",";
                }

                //remove covered instances
                rr.setInstancesIndices(remove);
                distribucija=newData.attributeStats(newData.classIndex()).nominalCounts;
                rr.setInputFormat(newData);
                newData = Filter.useFilter(newData, rr);
                distribucija=newData.attributeStats(newData.classIndex()).nominalCounts;
                remove="";
            }
            }

            if(covering){
                if(distribucija!=null){
                    if(distribucija[c]<=Math.ceil(nc-(cfI*nc))) // all covered    //if(distribucija[classToExplain]==0)
                        break;
                }
            }
        }
         list = new ArrayList<>(allFeatures);
        }
        catch(Exception e){
            System.out.println("Instability in FURIA algorithm");
                logFile.println("Instability in FURIA algorithm");
            e.printStackTrace(System.err);
        }
        
        return list;
    }    

    public static Instances addFeatNumOfN(Instances data,ArrayList<String> allFeat) throws Exception{
        String attName, attrNameRule,nomValue;
        Add filter;
        String splitChar;
        String nominalLabels;
        double attrValue;
        Instances newData=new Instances(data);
        int tmp;
    
        for(int i=0;i<allFeat.size();i++){
            nominalLabels="";
            attName=allFeat.get(i);
            if(!attName.contains(" and "))    //num-of-N features that have only one condition are not generated because they are the same as those generated by FURIA -> e.g., the result of num-of-N ((A8> = 1)) and (A8> = 1) is the same
                continue;

            String attrTmpRule [] = attName.trim().split(" and "); //left side ... attributes e.g., (A3 = 1) and (A2 = 1) and (A1 = 0)
        
            for(int a=0;a<=attrTmpRule.length;a++)  //max is all conditions (attrTmpRule.length), min is 0 -> 0...attrTmpRule.length
                if(a==attrTmpRule.length)
                    nominalLabels+=a;
                else
                    nominalLabels+=a+",";

            filter= new Add();
            filter.setAttributeIndex("" + (newData.numAttributes())); //parameter of the method must be String

            attName="num-of-N("+attName+")";
        
            filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n
            filter.setNominalLabels(nominalLabels); //filter.setNominalLabels("0,1,2,3,4");
            filter.setInputFormat(newData);
            newData = Filter.useFilter(newData, filter);
        
            for(int j = 0; j < newData.numInstances(); j++){
                tmp=0;
                String tmpOrRule []=null;
                for(int k=0;k<attrTmpRule.length;k++){  //parsing e.g., (A3 = 1) and (A2 = 1) and (A1 = 0) we parse attribute name and value
                    //OR
                    if(attrTmpRule[k].contains(" or ")){
                        tmpOrRule=attrTmpRule[k].split(" or ");                        
                        for(int n=0;n<tmpOrRule.length;n++){
                            splitChar=tmpOrRule[n].contains(" = ")?" = ":tmpOrRule[n].contains(" >= ")?" >= ":" <= ";  //we consider only one of these three options =, >= in <= 
                            attrNameRule=tmpOrRule[n].trim().replace("(", "").replace(")", "").split(splitChar)[0].trim();
                            attrValue=Double.parseDouble(tmpOrRule[n].trim().replace("(", "").replace(")", "").split(splitChar)[1].trim());

                            if(newData.attribute(attrNameRule).isNominal()){
                                nomValue=tmpOrRule[n].trim().replace("(", "").replace(")", "").split("=")[1].trim();  //nominal value as a string    
                                if(newData.instance(j).stringValue(newData.attribute(attrNameRule).index()).equals(nomValue)){
                                    tmp++;                               
                                }
                            }
                            else{
                                if(splitChar.equals(" = "))
                                    if(newData.instance(j).value(newData.attribute(attrNameRule)) == attrValue)
                                        tmp++;                                
                                if(splitChar.equals(" >= "))
                                    if(newData.instance(j).value(newData.attribute(attrNameRule)) >= attrValue)
                                        tmp++;                                  
                                if(splitChar.equals(" <= "))
                                    if(newData.instance(j).value(newData.attribute(attrNameRule)) <= attrValue)
                                        tmp++;                              
                            }
                        }
                        continue; //if the rule contains or then we need to move on reviewing the next rule
                    }
                    //AND
                    if(attrTmpRule[k].contains(" = ")){
                        attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split("=")[0].trim();             
                        if(newData.attribute(attrNameRule).isNominal()){
                            nomValue=attrTmpRule[k].trim().replace("(", "").replace(")", "").split("=")[1].trim();  //nominal value as a string                    
                            if(newData.instance(j).stringValue(newData.attribute(attrNameRule).index()).equals(nomValue))
                                tmp++;                                                      
                        }
                        else{
                            attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split("=")[1].trim());
                            if(newData.instance(j).value(newData.attribute(attrNameRule)) == attrValue)
                                tmp++;                          
                        }
                    }
                    else if (attrTmpRule[k].contains(" >= ")){
                        attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" >= ")[0].trim();
                        attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split(">=")[1].trim());
                        if(newData.instance(j).value(newData.attribute(attrNameRule)) >= attrValue)
                            tmp++;
                    } 
                    else if (attrTmpRule[k].contains(" <= ")){
                        attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split("<=")[0].trim();
                        attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split("<=")[1].trim());
                        if(newData.instance(j).value(newData.attribute(attrNameRule)) <= attrValue)
                            tmp++;                         
                    }
                }
            newData.instance(j).setValue(newData.numAttributes() - 2, tmp);    
            }
        
        }
        
        return newData;
    }

    public static Instances addFeatures(Instances data,ArrayList<String> allFeat) throws Exception{ //add features from FURIA to dataset
        String attName, attrNameRule, nomValue;
        Add filter;
        double attrValue;
        Instances newData=new Instances(data);
        int tmp=0;
        
        for(int i=0;i<allFeat.size();i++){
            attName=allFeat.get(i);
            String attrTmpRule [] = attName.trim().split(" and "); //left side ... attributes e.g., (A3 = 1) and (A2 = 1) and (A1 = 0)
 
            filter= new Add();
            filter.setAttributeIndex("" + (newData.numAttributes())); //parameter of the method must be String
            filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n
            filter.setNominalLabels("0,1");
            filter.setInputFormat(newData);
            newData = Filter.useFilter(newData, filter);
        
            for(int j = 0; j < newData.numInstances(); j++){
                tmp=1;
                for(int k=0;k<attrTmpRule.length;k++){  //parsing e.g., (A3 = 1) and (A2 = 1) and (A1 = 0) we parse attribute name and value
                    if(tmp==0) //if the conditions within the OR are false then return false 
                        break;
                    //AND
                    if(attrTmpRule[k].contains(" = ")){
                        attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" = ")[0].trim();  
                        if(newData.attribute(attrNameRule).isNominal()){
                            nomValue=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" = ")[1].trim();  //nominal value as a string
                            if(!newData.instance(j).stringValue(newData.attribute(attrNameRule).index()).equals(nomValue)){
                                tmp=0;                                
                                break;
                            }
                        }
                        else{
                            attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split("=")[1].trim());
                            if(newData.instance(j).value(newData.attribute(attrNameRule)) != attrValue){
                                tmp=0;                   
                                break;
                            } 
                        }
                    }
                    else if (attrTmpRule[k].contains(" >= ")){
                        attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" >= ")[0].trim();
                        attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" >= ")[1].trim());
                        if(!(newData.instance(j).value(newData.attribute(attrNameRule)) >= attrValue)){
                            tmp=0;                
                            break;
                        }  
                    } 
                    else if (attrTmpRule[k].contains(" <= ")){
                        attrNameRule=attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" <= ")[0].trim();
                        attrValue=Double.parseDouble(attrTmpRule[k].trim().replace("(", "").replace(")", "").split(" <= ")[1].trim());
                        if(!(newData.instance(j).value(newData.attribute(attrNameRule)) <= attrValue)){
                            tmp=0;                              
                            break;
                        } 
                    }
                }
                newData.instance(j).setValue(newData.numAttributes() - 2, tmp);    
            }
        }
        
        return newData;
    }
    
    public static Instances addNumFeat(Instances data, OperationNum op, List newTmpComb) throws Exception{ //Discretization is by Fayyad & Irani's MDL method (the default).
    // we need folds for saving info of unimportant features
    String attName;
    Add filter;
    String combName;
    Instances newData=new Instances(data);
    int idxAttr1,idxAttr2, tmpIdx;
    double tmp;
    Set setA = new HashSet(); //to control attribute names
    Set setB = new HashSet(); //to control the names of generated combinations
    
    for (int i=0;i<data.numAttributes()-1;i++)
        setA.add(data.attribute(i).name());

    for(int j=0;j<newTmpComb.size();j++){   //we get combinations in style [1,2]
        idxAttr1=Integer.parseInt(newTmpComb.get(j).toString().replace("[","").replace("]", "").trim().split(",")[0].trim());
        idxAttr2=Integer.parseInt(newTmpComb.get(j).toString().replace("[","").replace("]", "").trim().split(",")[1].trim());

        if(!(op==OperationNum.ADD) && !(op==OperationNum.ABSDIFF)){
            for(int k=0;k<2; k++){  //we try all combinations A1/A2 and A2/A1
                if(k==1){
                    tmpIdx=idxAttr1;
                    idxAttr1=idxAttr2;
                    idxAttr2=tmpIdx;
                }
                if(newData.attribute(idxAttr1).isNumeric() && newData.attribute(idxAttr2).isNumeric()){
                    combName=newData.attribute(idxAttr1).name()+newData.attribute(idxAttr2).name();
                    if(setB.contains(combName)) //if a combination already exists, we do not generate it
                        continue;
                    else
                        setB.add(combName); 

                    attName=newData.attribute(idxAttr1).name()+" "+op.name()+" "+newData.attribute(idxAttr2).name();
                    if(setA.contains(attName)) //if the attribute already exists, do not add it
                        continue;
                    else
                        setA.add(attName); 

                    filter= new Add();
                    filter.setAttributeIndex("" + (newData.numAttributes())); //parameter of the method must be String
                    filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n

                    filter.setInputFormat(newData);
                    newData = Filter.useFilter(newData, filter);

                    for(int m = 0; m < newData.numInstances(); m++){
                        if(newData.instance(m).value(idxAttr2)==0.0 && op==OperationNum.DIVIDE){ //division by zero NaN in weka NaN is marked the same as missing value -> ?
                            if(newData.instance(m).value(idxAttr1)<0 || newData.instance(m).value(idxAttr2)<0)
                                newData.instance(m).setValue(newData.numAttributes() - 2, -Float.MAX_VALUE); 
                            else
                                newData.instance(m).setValue(newData.numAttributes() - 2, Float.MAX_VALUE);
                        }
                        else{
                            tmp=computeNumOperation(newData.instance(m).value(idxAttr1), op, newData.instance(m).value(idxAttr2));
                            newData.instance(m).setValue(newData.numAttributes() - 2, tmp);    //enriched dataset
                        }
                    }
                }
            }
        }
        else{
            if(newData.attribute(idxAttr1).isNumeric() && newData.attribute(idxAttr2).isNumeric()){
                combName=newData.attribute(idxAttr1).name()+newData.attribute(idxAttr2).name();
                if(setB.contains(combName)) //if combination exists, then we don't generate it
                    continue;
                else
                    setB.add(combName); 

                attName=newData.attribute(idxAttr1).name()+" "+op.name()+" "+newData.attribute(idxAttr2).name();
                if(setA.contains(attName)) //if feature exists then we don't add it
                    continue;
                else
                    setA.add(attName); 

                filter= new Add();
                filter.setAttributeIndex("" + (newData.numAttributes())); //the method parameter must be String
                filter.setAttributeName(attName);  //indexes are from 0 ... n-1, attribute names are from 1 to n
                filter.setInputFormat(newData);
                newData = Filter.useFilter(newData, filter);

                for(int m = 0; m < newData.numInstances(); m++){
                    if(newData.instance(m).value(idxAttr2)==0.0 && op==OperationNum.DIVIDE){ //division by zero NaN in weka NaN is marked the same as missing value -> ?
                        if(newData.instance(m).value(idxAttr1)<0 || newData.instance(m).value(idxAttr2)<0) //enriched dataset, (-)Float.MAX_VALUE signals division by zero
                            newData.instance(m).setValue(newData.numAttributes() - 2, -Float.MAX_VALUE);    
                        else        
                            newData.instance(m).setValue(newData.numAttributes() - 2, Float.MAX_VALUE);
                    }
                    else{
                        tmp=computeNumOperation(newData.instance(m).value(idxAttr1), op, newData.instance(m).value(idxAttr2));
                        newData.instance(m).setValue(newData.numAttributes() - 2, tmp);    //enriched dataset
                    }
                }
            }           
        } 
    }
    
        return newData;
    }
    
    public static int[] numOfFeat(Instances dataset,int numOfAttr) { //counting features 
        int numLogical = 0, numThr=0, numFuria=0,numCartesian=0, numRelational=0, numNumerical=0;
        int features[]=new int[6];
        for(int i=numOfAttr;i<dataset.numAttributes()-1;i++){
            if(dataset.attribute(i).name().contains(" xor ")||(dataset.attribute(i).name().contains(" or ") && !dataset.attribute(i).name().contains("(") )||dataset.attribute(i).name().contains(" equ ")||dataset.attribute(i).name().contains(" impl ") || (dataset.attribute(i).name().contains(" and ") && !dataset.attribute(i).name().contains("(") ))
                numLogical++;
            if(dataset.attribute(i).name().contains("num-of-N(("))
                numThr++;
            if(dataset.attribute(i).name().contains("(") && !dataset.attribute(i).name().contains("num-of-N(("))
                numFuria++;
            if(dataset.attribute(i).name().contains("_x_"))
                numCartesian++;
            if(dataset.attribute(i).name().contains(" LESSTHAN ") || dataset.attribute(i).name().contains(" EQUAL ") || dataset.attribute(i).name().contains(" DIFF "))
                numRelational++;
            if(dataset.attribute(i).name().contains(" DIVIDE ") || dataset.attribute(i).name().contains(" SUBTRACT ") || dataset.attribute(i).name().contains(" ADD "))
                numNumerical++;
        }
        features[0]=numLogical;
        features[1]=numThr;
        features[2]=numFuria;
        features[3]=numCartesian;
        features[4]=numRelational;
        features[5]=numNumerical;
    
        return features;
    }

    public static int numOfLogFeatInTree(Instances data, int numOfOrigAttr, J48 dt) throws Exception{
        int numLogical=0;
        String notEscaped=dt.graph();
        String tmpParse[]=notEscaped.split("\\r?\\n"); //split by new line
        String attName;
        for(int o=1;o<tmpParse.length-1;o++){    //first and last field are digraph J48Tree { and }
            if(tmpParse[o].contains("->") || tmpParse[o].contains("shape=box")) //skip leaves or binary marks (left(0) an right(1))
                continue;
            attName=tmpParse[o].substring(tmpParse[o].indexOf("\"")+1,tmpParse[o].lastIndexOf("\""));   // we don't need " "        
            if(attName.contains(" xor ")||(attName.contains(" or ") && !attName.contains("("))||attName.contains(" equ ")||attName.contains(" impl ")||(attName.contains(" and ") && !attName.contains("(")))
                numLogical++;
        }
        
        return numLogical;
    }

    public static int[] numOfRelFeatInTree(Instances data, int numOfOrigAttr, J48 dt) throws Exception{  //number of relational features and sum of terms in constructs
        int numRelational[]=new int[2];   //0-num of features, 1-sum of terms in constructs
        String notEscaped=dt.graph();
        String tmpParse[]=notEscaped.split("\\r?\\n"); //split by new line
        String attName;
        for(int o=1;o<tmpParse.length-1;o++){    //first and last field are digraph J48Tree { and }
            if(tmpParse[o].contains("->") || tmpParse[o].contains("shape=box")) //skip leaves or binary marks (left(0) an right(1))
                continue;
            attName=tmpParse[o].substring(tmpParse[o].indexOf("\"")+1,tmpParse[o].lastIndexOf("\""));   // we don't need " "        
            if(attName.contains(" LESSTHAN ") || attName.contains(" DIFF ") || attName.contains(" EQUAL ")){
                numRelational[0]++;
                numRelational[1]+=2; //lessthan, diff and equal are features of order 2 e.g., A1 DIFF A2
            }
        }
        
        return numRelational;
    }
    
    public static int[] numOfCartFeatInTree(Instances data, int numOfOrigAttr, J48 dt) throws Exception{  //number of Cartesian features and sum of terms in constructs
        int num[]=new int[2];   //0-num of features, 1-sum of terms in constructs
        String notEscaped=dt.graph();
        String tmpParse[]=notEscaped.split("\\r?\\n"); //split by new line
        String attName;
        for(int o=1;o<tmpParse.length-1;o++){    //first and last field are digraph J48Tree { and }
            if(tmpParse[o].contains("->") || tmpParse[o].contains("shape=box")) //skip leaves or binary marks (left(0) an right(1))
                continue;
            attName=tmpParse[o].substring(tmpParse[o].indexOf("\"")+1,tmpParse[o].lastIndexOf("\""));   // we don't need " "
            if(attName.contains("_x_")){
                num[0]++;
                num[1]+=attName.split("_x_").length;    //sum of terms
            }
        }
        
        return num;
    }

    public static int[] numOfDrThrFeatInTree(Instances data, int numOfOrigAttr, J48 dt) throws Exception{  //number of decision rule and threshold features and sum of terms in constructs in tree
        int num[]=new int[4]; //0-num of feat, 1-sum of terms in constructs of decision rule (FURIA) feat, 2-num of thr feat, 3-sum of construct of thr feat
        String notEscaped=dt.graph();
        String tmpParse[]=notEscaped.split("\\r?\\n"); //split by new line
        String attName;
        for(int o=1;o<tmpParse.length-1;o++){    //first and last field are digraph J48Tree { and }
            if(tmpParse[o].contains("->") || tmpParse[o].contains("shape=box")) //skip leaves or binary marks (left(0) an right(1))
                continue;
            attName=tmpParse[o].substring(tmpParse[o].indexOf("\"")+1,tmpParse[o].lastIndexOf("\""));   // we don't need " "
            if(attName.contains("(") && !attName.contains("num-of-N((")){   //counting decision rule (FURIA) feat
                num[0]++;
                if(attName.contains(" and "))
                    num[1]+=attName.split(" and ").length;
                else
                    num[1]++; // we need to count also features with one attribute e.g., (V16 = success)            
                //NOT IN USE ANYMORE
                if(attName.contains(" or "))   //this is our combination of or rule; not default Furia rule
                    num[1]+=attName.split(" or ").length;
            }
            if(attName.contains("num-of-N((")){ //counting threshold feat
                num[2]++;
                if(attName.contains(" and "))
                    num[3]+=attName.split(" and ").length;
            }
        }
        
        return num;
    }

    public static int[] numOfNumFeatInTree(Instances data, int numOfOrigAttr, J48 dt) throws Exception{  //number of numerical features and sum of terms in constructs in tree
        int numNumerical[]=new int[2];   //0-num of feat, 1-sum of terms in constructs of numerical feat
        String notEscaped=dt.graph();
        String tmpParse[]=notEscaped.split("\\r?\\n"); //split by new line
        String attName;
        for(int o=1;o<tmpParse.length-1;o++){   //first and last field are digraph J48Tree { and }
            if(tmpParse[o].contains("->") || tmpParse[o].contains("shape=box")) //skip leaves or binary marks (left(0) an right(1))
                continue;
            attName=tmpParse[o].substring(tmpParse[o].indexOf("\"")+1,tmpParse[o].lastIndexOf("\""));   // we don't need " "        
            if(attName.contains(" DIVIDE ") || attName.contains(" ADD ") || attName.contains(" SUBTRACT ")){
                numNumerical[0]++;
                numNumerical[1]+=2; //DIVIDE, ADD and SUBTRACT are features of order 2 e.g., A1 DIVIDE A2
            }
        }
        
        return numNumerical;
    }    
        
    //for original model origDataset and enrichedDataset are the same    
    public static int sumOfTermsInConstrInTree(Instances data, int numOfOrigAttr, J48 dt) throws Exception{  //sum of terms in constructs in tree
        String notEscaped=dt.graph();
        String tmpParse[]=notEscaped.split("\\r?\\n"); //split by new line
        String tmpSize[];
        int sumOfConstructs=0;
        String attName;
        for(int o=1;o<tmpParse.length-1;o++){    //first and last field are digraph J48Tree { and }
            if(tmpParse[o].contains("->") || tmpParse[o].contains("shape=box")) //skip leaves or binary marks (left(0) an right(1))
                continue;
            attName=tmpParse[o].substring(tmpParse[o].indexOf("\"")+1,tmpParse[o].lastIndexOf("\""));   //we don't need " "
            if(attName.contains(" or ")){   //binary features (or,xor,equ,impl)
                tmpSize=attName.split(" or ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" impl ")){
                tmpSize=attName.split(" impl ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" xor ")){
                tmpSize=attName.split(" xor ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" equ ")){
                tmpSize=attName.split(" equ ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" and ")){//decision rules and thr features (... and ...) and logical feat A1 and A2
                tmpSize=attName.split(" and ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" LESSTHAN ")){    //A1 LESSTHAN A2
                tmpSize=attName.split(" LESSTHAN ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" EQUAL ")){
                tmpSize=attName.split(" EQUAL ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" DIFF ")){
                tmpSize=attName.split(" DIFF ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains("_x_")){   //A1_X_A2 ... Cartesian product
                tmpSize=attName.split("_x_");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" DIVIDE ")){
                tmpSize=attName.split(" DIVIDE ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" ADD ")){
                tmpSize=attName.split(" ADD ");
                sumOfConstructs+=tmpSize.length;
            }
            else if(attName.contains(" SUBTRACT ")){
                tmpSize=attName.split(" SUBTRACT ");
                sumOfConstructs+=tmpSize.length;
            }
            else
                sumOfConstructs++; //if there is no construct in node there is only one attribute
        }
        
        return sumOfConstructs;
    }

    public static int sumOfLFTermsInConstrInTree(Instances data, int numOfOrigAttr, J48 dt) throws Exception{  //sum of terms in constructs of logical operator features in tree
        String notEscaped=dt.graph();
        String tmpParse[]=notEscaped.split("\\r?\\n");  //split by new line
        String tmpSize[];
        int sumOfConstructs=0;
        String attName;
        for(int o=1;o<tmpParse.length-1;o++){   //first and last field are digraph J48Tree { and }
            if(tmpParse[o].contains("->") || tmpParse[o].contains("shape=box")) //skip leaves or binary marks (left(0) an right(1))
                continue;
            attName=tmpParse[o].substring(tmpParse[o].indexOf("\"")+1,tmpParse[o].lastIndexOf("\""));   // we don't need " "
            if(attName.contains(" or ")){   //binary features (or,xor,equ,impl)
                tmpSize=attName.split(" or ");
                sumOfConstructs+=tmpSize.length;
            }
            if(attName.contains(" impl ")){
                tmpSize=attName.split(" impl ");
                sumOfConstructs+=tmpSize.length;
            }
            if(attName.contains(" xor ")){
                tmpSize=attName.split(" xor ");
                sumOfConstructs+=tmpSize.length;
            }
            if(attName.contains(" equ ")){
                tmpSize=attName.split(" equ ");
                sumOfConstructs+=tmpSize.length;
            }
            if(attName.contains(" and ") && !attName.contains("(")){    //decision rules and thr features (... and ...) and logical feat A1 and A2
                tmpSize=attName.split(" and ");
                sumOfConstructs+=tmpSize.length;
            }
        }
        
        return sumOfConstructs;
    }
    
    public static int sumOfTermsInConstrInRule(ArrayList<Rule> ar1, Instances data){
        int count=0;
        String rule;
        String tmpOuter[];
        for(Rule el: ar1){
            rule=((FURIA.RipperRule)el).toString(data.classAttribute());
            if(rule.contains(" and ")){
                tmpOuter=rule.split(" and ");
                count+=tmpOuter.length;
                for (int i=0;i<tmpOuter.length; i++){ 
                    if(tmpOuter[i].contains(" or "))    //if we have e.g., (A1 or A2) and (A4 or A7) then we get from and "split" 2 and then from each or "split" 1-> 2+1+1=4
                        count+=tmpOuter[i].split(" or ").length-1;    //we have constructive induction of depth 3 for conjunction and disjunction ... -1 because we count one attribute in and split                 
                    else if(tmpOuter[i].contains(" xor "))
                        count++;                    
                    else if(tmpOuter[i].contains(" impl "))
                        count++;                    
                    else if(tmpOuter[i].contains(" equ "))
                        count++;                    
                    else if(tmpOuter[i].contains(" LESSTHAN "))
                        count++;
                    else if(tmpOuter[i].contains(" EQUAL "))
                        count++;                    
                    else if(tmpOuter[i].contains(" DIFF "))
                        count++;                    
                    else if(tmpOuter[i].contains("_x_"))
                        count++;                    
                    else if(tmpOuter[i].contains(" DIVIDE "))
                        count++;                    
                    else if(tmpOuter[i].contains(" ADD "))
                        count++;                   
                    else if(tmpOuter[i].contains(" SUBTRACT "))
                        count++;                    
                }
            }
            else if(rule.contains(" or ")){ //our construction of or in decision rules features
                tmpOuter=rule.split(" or ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" xor ")){ 
                tmpOuter=rule.split(" xor ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" impl ")){ 
                tmpOuter=rule.split(" impl ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" equ ")){
                tmpOuter=rule.split(" equ ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" LESSTHAN ")){ 
                tmpOuter=rule.split(" LESSTHAN ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" EQUAL ")){ 
                tmpOuter=rule.split(" EQUAL ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" DIFF ")){ 
                tmpOuter=rule.split(" DIFF ");
                count+=tmpOuter.length;
            }
            else if(rule.contains("_x_")){  //Cartesian product
                tmpOuter=rule.split("_x_");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" DIVIDE ")){ 
                tmpOuter=rule.split(" DIVIDE ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" ADD ")){ 
                tmpOuter=rule.split(" ADD ");
                count+=tmpOuter.length;
            }
            else if(rule.contains(" SUBTRACT ")){ 
                tmpOuter=rule.split(" SUBTRACT ");
                count+=tmpOuter.length;
            }
            else
                count++;
	}
        
        return count;    
    }
    
    //Fayyad & Irani's MDL
    public static Instances discretizeFI(Instances newData, int attrIdx, boolean kononenko) throws Exception{ //we have to prepare data without class variable
        NominalToBinary nominalToBinary=null;
        Remove remove= new Remove();
        remove.setAttributeIndices((newData.attribute(attrIdx).index()+1)+",last");//rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1. e.g., first-3,5,6-last
        remove.setInvertSelection(true);
        remove.setInputFormat(newData);
        newData = Filter.useFilter(newData, remove); //just one attribute
    
        if(newData.attribute(0).isNominal()){ //we have only one attribute in dataset ... index starts with 0
            nominalToBinary = new NominalToBinary();
            nominalToBinary.setAttributeIndices("first");
            nominalToBinary.setInputFormat(newData);
        }
    
        if(newData.attribute(0).isNumeric()){ 
            //discretization    
            weka.filters.supervised.attribute.Discretize filter;    //because of the same class name in different packages
            //setup filter
            filter = new weka.filters.supervised.attribute.Discretize();
            //Discretization is by Fayyad & Irani's MDL method (the default). Continuous attributes are discretized and binarized.
            //filter.
            newData.setClassIndex(newData.numAttributes()-1); //we need class index for Fayyad & Irani's MDL
            filter.setUseBinNumbers(true); //e.g. BXofY ... B1of1
            filter.setUseKononenko(kononenko);
            filter.setInputFormat(newData);
            //apply filter
            newData = Filter.useFilter(newData, filter);
            //nominal to binary  
            nominalToBinary = new NominalToBinary();
            nominalToBinary.setAttributeIndices("first");
            nominalToBinary.setInputFormat(newData);        
        }
        newData = Filter.useFilter(newData, nominalToBinary);
    
        remove= new Remove();
        remove.setAttributeIndices("last"); //we remove class index
        remove.setInputFormat(newData);
        newData = Filter.useFilter(newData, remove);    
        
        //rename attributes e.g., from 'A7=\'B1of3\'' to A7-B1of3
        String tmp;
        for (int i=0;i<newData.numAttributes();i++){
            tmp=newData.attribute(i).name();
            if(tmp.contains("=") || tmp.contains("'")){
                tmp=tmp.replaceFirst("=","==").replaceAll("'", "");
                newData.renameAttribute(i, tmp);
            }
        }
    
        return newData;
    } 

    //Fayyad & Irani's MDL
    public static Instances discretizeFITestBasedOnTrain(Instances trainData, Instances testData, int attrIdx, boolean kononenko) throws Exception{ //we have to prepare data without class variable
        Instances train=new Instances(trainData);
        Instances test=new Instances(testData);
        NominalToBinary nominalToBinary=null;
    
        Remove remove= new Remove();
        remove.setAttributeIndices((train.attribute(attrIdx).index()+1)+",last");//rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1. e.g.: first-3,5,6-last
        remove.setInvertSelection(true);
        remove.setInputFormat(train);
        train = Filter.useFilter(train, remove); //just one attribute
    
        remove.setAttributeIndices((test.attribute(attrIdx).index()+1)+",last");//rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1. e.g.: first-3,5,6-last
        remove.setInvertSelection(true);
        remove.setInputFormat(test);
        test = Filter.useFilter(test, remove); //just one attribute
    
        if(test.attribute(0).isNominal()){ //we have only one attribute in dataset ... index starts with 0
            nominalToBinary = new NominalToBinary();
            nominalToBinary.setAttributeIndices("first");
            nominalToBinary.setInputFormat(test);
        }
    
        if(test.attribute(0).isNumeric()){ 
            //discretization    
            weka.filters.supervised.attribute.Discretize filter;    //because of the same class name in different packages
            //setup filter
            filter = new weka.filters.supervised.attribute.Discretize();
            //Discretization is by Fayyad & Irani's MDL method (the default). Continuous attributes are discretized and binarized.
            //filter.
            train.setClassIndex(train.numAttributes()-1); //we need class index for Fayyad & Irani's MDL
            test.setClassIndex(test.numAttributes()-1);
        
            filter.setUseBinNumbers(true); //e.g. BXofY ... B1of1
            filter.setUseKononenko(kononenko);
            filter.setInputFormat(train);   
            //apply filter
            train = Filter.useFilter(train, filter); 
            test = Filter.useFilter(test, filter); //we have to apply discretization on test dataset based on info from train dataset
            //nominal to binary  
            nominalToBinary = new NominalToBinary();
            nominalToBinary.setAttributeIndices("first");
            nominalToBinary.setInputFormat(test);        
        }
        test = Filter.useFilter(test, nominalToBinary);
    
        remove= new Remove();
        remove.setAttributeIndices("last"); //we remove class index
        remove.setInputFormat(test);
        test = Filter.useFilter(test, remove);    
    
        //rename attributes e.g., from 'A7=\'B1of3\'' to A7-B1of3
        String tmp;
        for (int i=0;i<test.numAttributes();i++){
            tmp=test.attribute(i).name();
            if(tmp.contains("=") || tmp.contains("'")){
                tmp=tmp.replaceFirst("=", "==").replaceAll("'", "");
                test.renameAttribute(i, tmp);
            }
        }
        
        return test;
    }
    
    //only for discrete attributes
    public static void mdlAllAttrFeat(Instances data) throws Exception {
        Map<String, Double> mapMDL=new TreeMap<>(Collections.reverseOrder());
        for(int i=0;i<data.numAttributes()-1;i++){
            KononenkosMDL kMDL=new KononenkosMDL(data);
            mapMDL.put(data.attribute(i).name(),kMDL.kononenkosMDL(data,i));   //kononenkosMDL ... computing MDL for each attribute
        }
        System.out.println("Kononenko's MDL"); 
        LinkedList<Map.Entry<String, Double>> listMDL = new LinkedList<>(mapMDL.entrySet());
        Comparator<Map.Entry<String, Double>> comparator2 = Comparator.comparing(Map.Entry::getValue);
        Collections.sort(listMDL, comparator2.reversed());  //if we want reversed order ... descending order
        for(Map.Entry<String, Double> me : listMDL)
            System.out.printf(" %4.4f %s\n",me.getValue(), me.getKey()); 
    }

    public static void mdlCORElearn(Instances data, RCaller rCaller, RCode code){  //evaluation of the whole dataset
        try{
        File output = new File("Rdata/dataForR.arff");  // <--- This is the result file 
        OutputStream out = new FileOutputStream(output);        
        DataSink.write(out, data);
        out.close();

        code.clear();
        code.addRCode("library(CORElearn)");
        code.addRCode("library(RWeka)");
        code.addRCode("dataset <- read.arff(\"Rdata/dataForR.arff\")");
        code.addRCode("estMDL <- attrEval(which(names(dataset) == names(dataset)[length(names(dataset))]), dataset, estimator=\"MDL\",outputNumericSplits=TRUE)");   //last attribute is class attribute

        rCaller.setRCode(code);
        rCaller.runAndReturnResultOnline("estMDL");
        String tmpRcall[]=rCaller.getParser().getAsStringArray("attrEval");   //name in R "attrEval", get data from R, evaluated attributes

        Map<String, Double> mapMDL=new TreeMap<>(Collections.reverseOrder());
        for(int i=0;i<data.numAttributes()-1;i++)
            mapMDL.put(data.attribute(i).name(),Double.parseDouble(tmpRcall[i]));   //we get attribute names from Java (Instances data) and evaluation from R
        
        LinkedList<Map.Entry<String, Double>> listMDL = new LinkedList<>(mapMDL.entrySet());
        Comparator<Map.Entry<String, Double>> comparator2 = Comparator.comparing(Map.Entry::getValue);
        Collections.sort(listMDL, comparator2.reversed()); //if we want reversed order ... descending order
        for(Map.Entry<String, Double> me : listMDL)
            if(justExplain)
                attrImpListMDL_KD.printf(" %4.4f %s\n",me.getValue(), me.getKey());
            else
                attrImpListMDL.printf(" %4.4f %s\n",me.getValue(), me.getKey());

        deleteTempRFiles(); //better than rCaller.deleteTempFiles(); deleteTempFiles() sometimes does not delete all tmp files
        output.delete();
        }
        catch (Exception ex){
            System.out.println("Error in the method mdlCORElearn");
                Logger.getLogger(FeatConstr.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void lowLevelReliefF(Instances data) throws Exception {
        AttributeSelection attSel = new AttributeSelection();
        Ranker search = new Ranker();
        ReliefFAttributeEval evals = new ReliefFAttributeEval();
        attSel.setRanking(true);
        attSel.setEvaluator(evals);
        attSel.setSearch(search);
        attSel.SelectAttributes(data);
        String out=attSel.toResultsString().substring(attSel.toResultsString().indexOf("Ranked attributes"),attSel.toResultsString().indexOf("Selected attributes"));
        out=out.replaceAll("[\r\n]+", "\n");
        String[] lines = out.split("\\r?\\n");  //for win view, just to have nice report ;)
        for (String line : lines)
            System.out.printf("%s \r\n",line);
    }

    public static void lowLevelInfoGain(Instances data) throws Exception {
        AttributeSelection attSel = new AttributeSelection();
        Ranker search = new Ranker();
        InfoGainAttributeEval evals = new InfoGainAttributeEval();
        attSel.setRanking(true);
        attSel.setEvaluator(evals);
        attSel.setSearch(search);
        attSel.SelectAttributes(data);
        String out=attSel.toResultsString().substring(attSel.toResultsString().indexOf("Ranked attributes"),attSel.toResultsString().indexOf("Selected attributes"));
        out=out.replaceAll("[\r\n]+", "\n");
        String[] lines = out.split("\\r?\\n");  //for win view, just to have nice report ;)
        for (String line : lines){
            System.out.printf("%s \r\n",line);
            logFile.printf("%s \r\n",line);
        }
    }

    public static void lowLevelGainRatio(Instances data) throws Exception{
        AttributeSelection attSel = new AttributeSelection();
        Ranker search = new Ranker();
        GainRatioAttributeEval evals = new GainRatioAttributeEval();
        attSel.setRanking(true);
        attSel.setEvaluator(evals);
        attSel.setSearch(search);
        attSel.SelectAttributes(data);
        String out=attSel.toResultsString().substring(attSel.toResultsString().indexOf("Ranked attributes"),attSel.toResultsString().indexOf("Selected attributes"));
        out=out.replaceAll("[\r\n]+", "\n");
        String[] lines = out.split("\\r?\\n");  //for win view, just to have nice report ;)
        for (String line : lines) {
            System.out.printf("%s \r\n",line);
            logFile.printf("%s \r\n",line);
        }
    }

    public static double [][] lowLevelReliefFAttrSel(Instances data) throws Exception {
        AttributeSelection attSel = new AttributeSelection();
        Ranker search = new Ranker();
        ReliefFAttributeEval evals = new ReliefFAttributeEval();
        attSel.setRanking(true);
        attSel.setEvaluator(evals);
        attSel.setSearch(search);
        attSel.SelectAttributes(data);
        double out[][]=attSel.rankedAttributes();
    
        return out;
    }
    
    public static double [][] lowLevelInfoGainAttrSel(Instances data) throws Exception {
        AttributeSelection attSel = new AttributeSelection();
        Ranker search = new Ranker();
        InfoGainAttributeEval evals = new InfoGainAttributeEval();
        attSel.setRanking(true);
        attSel.setEvaluator(evals);
        attSel.setSearch(search);
        attSel.SelectAttributes(data);
        double out[][]=attSel.rankedAttributes();
        
        return out;
    }
    
    public static double [][] lowLevelGainRatioAttrSel(Instances data) throws Exception {
        AttributeSelection attSel = new AttributeSelection();
        Ranker search = new Ranker();
        GainRatioAttributeEval evals = new GainRatioAttributeEval();
        attSel.setRanking(true);
        attSel.setEvaluator(evals);
        attSel.setSearch(search);
        attSel.SelectAttributes(data);
        double out[][]=attSel.rankedAttributes();
        
        return out;
    }

    //we take whole original dataset    
    public static Instances justExplainAndConstructFeat(Instances dataset, Classifier predictionModel, boolean isClassification, RCaller rCaller, RCode code) throws Exception{
        System.out.println("Explaining dataset, making constructs ...");
        Instances trainFold = new Instances(dataset);   //we use all instances for train
        trainFold.setClassIndex(trainFold.numAttributes()-1);
        Random rnd = new Random(1);
        RandomForest rf=new RandomForest(); //rf.setSeed(1);
            rf.setNumExecutionSlots(processors);
            rf.setCalcOutOfBag(true);
        int minN=minNoise;
        
        if(trainFold.classAttribute().isNumeric())
            isClassification=false;
        
        trainFold.setClassIndex(trainFold.numAttributes()-1);
         
        namesOfDiscAttr(trainFold);     //save discretization intervals
        //heuristics for class selection for explanations
        //table with frequencies for each class - how many instances occur in a particular class
        double [] classDistr=Arrays.stream(trainFold.attributeStats(trainFold.classIndex()).nominalCounts).asDoubleStream().toArray(); //we convert because we need in log2Multinomial as parameter double array

        for(int i=0;i<minIndexClassifiers(classDistr).length;i++){
            if(minIndexClassifiers(classDistr)[i].v>=Math.ceil(trainFold.numInstances()*instThr/100.00)){   //we choose class to explain - class has to have at least instThr pct of whole instances
                classToExplain=minIndexClassifiers(classDistr)[i].i;
                break;
            }
        }
                     
        double allExplanations[][]=null, allWeights[][]=null;
        float allExplanationsSHAP[][], allWeightsSHAP[][]=null;
            
        List<String>impInter=null;
        Set<String> attrGroups= new LinkedHashSet<>();  //we want to keep the order of insertion and we don't want duplicates so LinkedHashSet
        
        int numClasses=1; //1 - just one iteration, we explain minority class, otherwise numClasses=classDistr.length;          
            
        if(explAllClasses)
            numClasses=classDistr.length;
  
        /*SHAP*/  
        if(treeSHAP){
        /*XGBOOST*/
            for(int c=0;c<numClasses;c++){  //we explain all classes    
                if(explAllClasses)
                    classToExplain=c;
            
                Instances explainData=new Instances(trainFold);
                RemoveWithValues filter = new RemoveWithValues();
                filter.setAttributeIndex("last");  //class
                filter.setNominalIndices((classToExplain+1)+"");    //what we remove ... if we invert selection than we keep ... +1 indexes go from 0, we need indexes from 1 for method setNominalIndices

                filter.setInvertSelection(true);

                filter.setInputFormat(explainData);
                explainData = Filter.useFilter(explainData, filter);
                    
                numInst=trainFold.attributeStats(trainFold.classIndex()).nominalCounts[classToExplain]; //number of instances (from the specified class) to explain //classToExplain instead of i if we explain just one class
                if(numInst==0)
                    continue;   //even this is possible, class has no instances e.g., class autos
                
                System.out.println("Explaining class: "+trainFold.classAttribute().value(classToExplain)+" explaining whole dataset: "+(explAllData?"YES":"NO"));   //classToExplain instead of i if we explain just one class
                    impGroupsKD.println("Explaining class: "+trainFold.classAttribute().value(classToExplain)+" explaining whole dataset: "+(explAllData?"YES":"NO"));
                
                DMatrix trainMat = wekaInstancesToDMatrix(trainFold);
                DMatrix explainMat = wekaInstancesToDMatrix(explainData);
                float tmpContrib[][];
                int numOfClasses=trainFold.numClasses();
                HashMap<String, Object> params = new HashMap<>();
                    params.put("eta", eta); //"eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ]  It is advised to have small values of eta in the range of 0.1 to 0.3 because of overfitting
                    params.put("max_depth", maxDepth);
                    params.put("silent", 1);
                    params.put("nthread", processors);
                    params.put("gamma", gamma);   //"gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ], gamma works by regularising using "across trees" information

                if(numOfClasses==2){    //for binary examples
                    params.put("objective", "binary:logistic");  //binary:logistic ??? logistic regression for binary classification, returns predicted probability (not class)
                    params.put("eval_metric", "error");
                }
                else{   //multi class problems
                    params.put("objective", "multi:softmax");  //multi:softprob multi:softmax
                    params.put("eval_metric", "merror");
                    params.put("num_class", (numOfClasses));    
                }
                
                Map<String, DMatrix> watches = new HashMap<>();
                    watches.put("train", trainMat);
                Booster booster = XGBoost.train(trainMat, params, numOfRounds, watches, null, null);
                String evalNameTrain[]={"train"};
                DMatrix [] trainMatArr={trainMat};
                //last param in evalSet has no sense, just index always returns evaluation of the last iteration ... because we put in booster (booster = XGBoost.train) last iteration?
                String accTrain=booster.evalSet(trainMatArr, evalNameTrain,0); 
                System.out.println("Internal (during building) accuracy of explanation model: "+(1-Double.parseDouble(accTrain.split(":")[1]))*100); //internal evaluation of the model
                    impGroupsKD.println("Internal (during building) accuracy of explanation model: "+(1-Double.parseDouble(accTrain.split(":")[1]))*100);
                    impGroupsKD.println("*********************************************************************************"); 

                if(explAllData)
                    tmpContrib=booster.predictContrib(trainMat, 0);   //Tree SHAP ... for each feature, and last for bias matrix of size (?nsample, nfeats + 1) ... feature contributions (SHAP?  xgboost predict)
                else
                    tmpContrib=booster.predictContrib(explainMat, 0);
                
                trainMatArr=null;
                booster.dispose();
                trainMat.dispose();
                explainMat.dispose();
                if(numOfClasses==2){
                    allExplanationsSHAP=removeCol(tmpContrib, tmpContrib[0].length-1);  //we remove last column, because we do not need column with bias
                }
                else{
                    int [] idxQArr=new int[trainFold.numAttributes()-1];
                    if(classToExplain==0)
                        for(int i=0;i<idxQArr.length;i++)
                            idxQArr[i]=i;
                    else{
                        int start=(classToExplain*trainFold.numAttributes()-1)+1;
                        int j=0;
                        for(int i=start;i<=idxQArr.length*(classToExplain+1);i++){
                            idxQArr[j]=i;
                            j++;
                        }
                    }                
                    allExplanationsSHAP= someColumns(tmpContrib, idxQArr);  //we take just columns of attributes from the class that we explain
                }

                if(numInst<minExplInst)
                    minN=minMinNoise;

                double noiseThr=(explainData.numInstances()*NOISE)/100.0; //we take number of noise threshold from the number of explained instances
                int usedNoise=Math.max((int)Math.ceil(noiseThr),minN);  //makes sense only if NOISE=0

                if(!fileName.contains("justForAccurateTime")){  //because of benchmarking ... java optimization etc.
                    System.out.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Tree SHAP num of expl. inst. "+(explAllData ? trainFold.numInstances() : numInst));
                        impGroupsKD.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Tree SHAP num of expl. inst. "+(explAllData ? trainFold.numInstances() : numInst));
                        impGroupsKD.println("Lower threshold thrL: "+thrL+" upper threshold thrU: "+thrU+" with step: "+step);
            }    
                
                for(double q=thrL;q<=thrU;q=q+step){
                    if(!fileName.contains("justForAccurateTime")){  //because of benchmarking ... java optimization etc.
                        impGroupsKD.println("--------------"); 
                        impGroupsKD.printf("Threshold: %2.2f\n",round(q,1));
                        impGroupsKD.println("--------------"); 
                    }

                    allWeightsSHAP=setWeights(trainFold,allExplanationsSHAP,round(q,1));
                    impInter=(getMostFqSubsets(allWeightsSHAP,trainFold,usedNoise));
                    attrGroups.addAll(impInter);
                }
            }//loop explain all classes SHAP
        }
        else{
            predictionModel.buildClassifier(trainFold);  
            if(excludeUppers(predictionModel.getClass().getSimpleName()).equals("RF")){ //OOB ia also calculated
                rf=(RandomForest)predictionModel;
                System.out.print("Internal evaluation of the model (OOB): "+(1-rf.measureOutOfBagError())*100);
                    impGroupsKD.print("Internal evaluation of the model (OOB): "+(1-rf.measureOutOfBagError())*100);
            }
            /*IME*/            
            for(int i=0;i<numClasses;i++){//we explain all classes    
                if(explAllClasses)
                    classToExplain=i;
                
                Instances explainData=new Instances(trainFold);
                RemoveWithValues filter = new RemoveWithValues();
                filter.setAttributeIndex("last") ;  //class
                filter.setNominalIndices((classToExplain+1)+""); //what we remove ... if we invert selection than we keep ... +1 indexes go from 0, we need indexes from 1 for method setNominalIndices

                filter.setInvertSelection(true);

                filter.setInputFormat(explainData);
                explainData = Filter.useFilter(explainData, filter);
                
                System.out.println("IME (explanation), "+method.name()+", "+(method.name().equals("adaptiveSampling") ? "min samples: "+minS+", sum of samples: "+sumOfSmp : method.name().equals("diffSampling")?" min samples: "+minS:" N_SAMPLES: "+N_SAMPLES)+" - alg. for searching concepts: "+predictionModel.getClass().getSimpleName());
                     impGroupsKD.println("IME (explanation), "+method.name()+", "+(method.name().equals("adaptiveSampling") ? "min samples: "+minS+", sum of samples: "+sumOfSmp : method.name().equals("diffSampling")?" min samples: "+minS:" N_SAMPLES: "+N_SAMPLES)+" - alg. for searching concepts: "+predictionModel.getClass().getSimpleName()); 
                System.out.println("Explaining class: "+trainFold.classAttribute().value(classToExplain)+", explaining whole dataset: "+(explAllData?"YES":"NO"));    //classToExplain instead of i if we explain just one class
                    impGroupsKD.println("Explaining class: "+trainFold.classAttribute().value(classToExplain)+", explaining all dataset: "+(explAllData?"YES":"NO"));
                System.out.println("---------------------------------------------------------------------------------");
                    impGroupsKD.println("---------------------------------------------------------------------------------");
                switch (method){
                    case aproxErrSampling:
                                    System.out.println("Sampling based on mi=(<1-alpha, e>), pctErr = "+pctErr+" error = "+error+".");
                                        impGroupsKD.println("Sampling based on mi=(<1-alpha, e>), pctErr = "+pctErr+" error = "+error+".");
                                    System.out.println("---------------------------------------------------------------------------------");
                                        impGroupsKD.println("---------------------------------------------------------------------------------");
                    break;
                }                
                               
                numInst=trainFold.attributeStats(trainFold.classIndex()).nominalCounts[classToExplain]; //number of explained instances; instances from the explained class //classToExplain instead of i if we explain just one class
                if(numInst==0)
                    continue;   //even this is possible, class has no instances e.g., class autos

                if(!explAllData){
                    if(numInst>maxToExplain){
                        System.out.println("We take only "+maxToExplain+" instances out of "+numInst+".");
                            impGroupsKD.println("We take only "+maxToExplain+" instances out of "+numInst+".");

                        explainData.randomize(rnd);
                        explainData = new Instances(explainData, 0, maxToExplain);

                        numInst=explainData.attributeStats(explainData.classIndex()).nominalCounts[classToExplain]; //for correct print on output
                    }
                    switch (method){
                        case equalSampling: 
                            allExplanations=IME.explainAllDatasetES(trainFold, explainData, predictionModel, N_SAMPLES, classToExplain);//equal sampling
                            break;
                        case adaptiveSamplingSS:  
                            allExplanations=IME.explainAllDatasetAS(trainFold, explainData, predictionModel, minS, sumOfSmp, classToExplain);//we need sumOfSmp (sum of samples) for additive sampling
                            break;
                        case adaptiveSamplingAE:    
                            allExplanations=IME.explainAllDatasetAS(trainFold, explainData, predictionModel, minS, classToExplain, error, pctErr);
                            break;                                              
                        case aproxErrSampling:  
                            allExplanations=IME.explainAllDatasetAES(predictionModel, trainFold, explainData, true, classToExplain, minS, error, pctErr);
                            break;                                           
                        }
                }
                else{
                    switch (method){
                        case equalSampling: 
                            allExplanations=IME.explainAllDatasetES(trainFold, trainFold, predictionModel, N_SAMPLES, classToExplain);//equal sampling
                            break;
                        case adaptiveSamplingSS:  
                            allExplanations=IME.explainAllDatasetAS(trainFold, trainFold, predictionModel, minS, sumOfSmp, classToExplain);//we need sumOfSmp (sum of samples) for additive sampling
                            break;
                        case adaptiveSamplingAE:    
                            allExplanations=IME.explainAllDatasetAS(trainFold, trainFold, predictionModel, minS, classToExplain, error, pctErr);
                            break;                         
                        case aproxErrSampling:  
                            allExplanations=IME.explainAllDatasetAES(predictionModel, trainFold, trainFold, true, classToExplain, minS, error, pctErr);
                            break;
                        }
                }   
                
                if(numInst<minExplInst)
                    minN=minMinNoise;

                double noiseThr=(numInst*NOISE)/100.0;//we take number of noise threshold from the number of explained instances
                int usedNoise=Math.max((int)Math.ceil(noiseThr),minN);  //makes sense only if NOISE=0 or num of explained instances is very low

                System.out.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Number of instances from class ("+explainData.classAttribute().value(classToExplain)+") is "+numInst);
                    impGroupsKD.println("We remove max(NOISE,minNoise) groups, NOISE="+NOISE+"% -> "+(int)Math.ceil(noiseThr)+ ", minNoise="+minN+" we remove groups of size "+usedNoise+". Number of instances from class ("+explainData.classAttribute().value(classToExplain)+") is "+numInst);
                    impGroupsKD.println("Lower threshold thrL: "+thrL+" upper threshold thrU: "+thrU+" with step: "+step);
                
                for(double q=thrL;q<=thrU;q=q+step){
                    impGroupsKD.println("--------------"); 
                    impGroupsKD.printf("Threshold: %2.2f\n",round(q,1));
                    impGroupsKD.println("--------------"); 

                    allWeights=setWeights(trainFold,allExplanations,round(q,1));
                    impInter=(getMostFqSubsets(allWeights,trainFold,usedNoise));
                    attrGroups.addAll(impInter);
                }     
            } //explain both (all) classes IME
        }
        listOfConcepts = new ArrayList<>(attrGroups);
            impGroupsKD.println("*********************************************************************************"); 
            impGroupsKD.println("All potential concepts, based on thresholds.");
            impGroupsKD.print("\t"); printFqAttrOneRow(listOfConcepts,trainFold);
            impGroupsKD.println("\n*********************************************************************************");
        
        //logical features
        //depth 2
        int N2=2;
        List allCombSecOrd=allCombOfOrderN(listOfConcepts,N2); //create groups for second ordered features
            trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.AND, false, 0, N2); //0 - we don't count unInf features
            trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.OR, false, 0, N2); //0 - we don't count unInf features
            trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.EQU, false, 0, N2); //0 - we don't count unInf features
            trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.XOR, false, 0, N2); //0 - we don't count unInf features
            trainFold= addLogFeatDepth(trainFold, allCombSecOrd,OperationLog.IMPL, false, 0, N2); //0 - we don't count unInf features
        //depth 3
        int N3=3;
        List allCombThirdOrd=allCombOfOrderN(listOfConcepts,N3);
            trainFold= addLogFeatDepth(trainFold, allCombThirdOrd,OperationLog.AND, false, 0, N3); //0 - we don't count unInf features
            trainFold= addLogFeatDepth(trainFold, allCombThirdOrd,OperationLog.OR, false, 0, N3); //0 - we don't count unInf features
        
        //decision rule and threshold features
        List<String> listOfFeat;
        listOfFeat=genFeatFromFuria(dataset, (ArrayList<String>) listOfConcepts, classToExplain, cf, pci,covering, featFromExplClass);   //generate features from Furia, parameter of FURIA cfF=0.7, cfI=0.9 stopping criteria 
            trainFold=addFeatures(trainFold, (ArrayList<String>) listOfFeat); //add features from Furia
                
        //num-of-N features ... we are counting true conditions from rules
            trainFold=addFeatNumOfN(trainFold, (ArrayList<String>) listOfFeat); //add num-Of-N features for evaluation
        
        //numerical features division 
        if(numerFeat){
            trainFold=addNumFeat(trainFold, OperationNum.DIVIDE, allCombSecOrd);              
            trainFold=addNumFeat(trainFold, OperationNum.SUBTRACT, allCombSecOrd);
            trainFold=addNumFeat(trainFold, OperationNum.ADD, allCombSecOrd);
        }	
        
        //relational features
            trainFold=addRelFeat(trainFold,allCombSecOrd,OperationRel.LESSTHAN,true,0); //true ... we remove uninformative features, last parameter is here irrelevant, we just put one value
            trainFold=addRelFeat(trainFold,allCombSecOrd,OperationRel.DIFF,true,0); //true ... we remove uninformative features                
        
        //Cartesian features
        boolean  allDiscrete=true;
        for(int i=0;i<dataset.numAttributes();i++)
            if(dataset.attribute(i).isNumeric()){    //check if problem is numeric
                allDiscrete=false; 
                System.out.println("We found continuous attribute!");
                break;
            }
        
        if(!allDiscrete){
            //discretization    
            weka.filters.supervised.attribute.Discretize filterDis;    //because of same class name in different packages
            // setup filter
            filterDis = new weka.filters.supervised.attribute.Discretize();
            //filter.
            dataset.setClassIndex(dataset.numAttributes()-1); //we need class index for Fayyad & Irani's MDL
            filterDis.setInputFormat(dataset);
            // apply filter
            dataset = Filter.useFilter(dataset, filterDis);
        }
        
        trainFold=addCartFeat(trainFold, dataset,allCombSecOrd,false,0,N2,true);
        attrImpListMDL_KD.println("MDL - after CI");
        mdlCORElearn(trainFold, rCaller, code);

        System.out.println("Constructs have been done!");
        
        if(visualisation){        
            System.out.println("Drawing ...");
            rf=new RandomForest();
            rf.setNumExecutionSlots(processors);
            rf.setCalcOutOfBag(true);
            visualizeModelInstances(rf, trainFold, true, RESOLUTION, numOfImpt, visFrom, visTo);  //visualise explanations from e.g., 50th to 60 instance
            System.out.println("Drawing is finished!");
        }  
    
        return trainFold;
    }      
    
    //for original model origDataset and enrichedDataset are the same    
    public static ModelAndAcc evaluateModel(Instances train, Instances test, Classifier model) throws Exception{  //the sum of all terms in constructs
        ModelAndAcc ma=new ModelAndAcc();
        Instances trainFold=new Instances(train);        
        
        model.buildClassifier(trainFold);
        Evaluation eval = new Evaluation(trainFold);
    
        eval.evaluateModel(model, test);
        ma.setClassifier(model);
        ma.setAcc((eval.correct())/(eval.incorrect()+eval.correct())*100.00); // the same ma.setAcc((1.0-eval.errorRate())*100.0);
   
        return ma;
    }    
    
    public static ParamSearchEval paramSearch(Instances train, Instances test, Classifier predictionModel, int numOfAttr, int split, RCaller rCaller, RCode code) throws Exception{
        Instances validation, subTrain, tmpValidation, tmpSubTrain;
        ParamSearchEval pse=new ParamSearchEval();
        StratifiedRemoveFolds fold;
        double attrImp;
        double intClassAcc=0;   //internal class accuracy  - class acc of paramSearch loop
        double maxIntAcc;   //maksimal internal acc
        ArrayList <Parameters> bestRndParam;
        String listOfUnInFeat;
        Remove remove;
        String bestParam, attName;
        int feat[]=new int[6];  //for counting logical, thr, FURIA, Cartesian, relational and numerical features
        int tree[]=new int[3];  //for counting tree size, number of leaves and number of constructs
        int numLogInTree[]=new int[2];  //number of logical features and constructs in tree
        int nC[]=new int[2];    //for counting Cartesian features in tree
        int nR[]=new int[2];    //for counting relational features in tree
        int nN[]=new int[2];    //for counting numerical features in tree
        int complexityF[]=new int[2];   //for counting complexity parameters of Furia
        int furiaThrC[]=new int[4]; //for Furia and Thr feat complexity
        int tmp[];
        long time[]=new long[2];    //0-feature construction time, 1-learning time
        Timer t1=new Timer();    
    
        bestRndParam =new ArrayList<>();
        maxIntAcc=0;

        fold = new StratifiedRemoveFolds();
        fold.setInputFormat(train);
        fold.setSeed(1);
        fold.setNumFolds(split);
        fold.setFold(split);
        fold.setInvertSelection(true);  //because we invert selection we take all folds except the "split" one
        subTrain = Filter.useFilter(train,fold); 

        fold = new StratifiedRemoveFolds();
        fold.setInputFormat(train);
        fold.setSeed(1);
        fold.setNumFolds(split);
        fold.setFold(split);
        fold.setInvertSelection(false);
        validation = Filter.useFilter(train,fold); 
     
        /*FS on validation dataset loop!!!*/  
        t1.start();
        for(int g=0;g<attrImpThrs.length;g++){
            listOfUnInFeat="";
            tmpSubTrain=new Instances(subTrain);
            tmpValidation=new Instances(validation);
            for(int j=numOfAttr;j<tmpSubTrain.numAttributes()-1;j++){
                if(!tmpSubTrain.attribute(j).isNumeric())
                    attrImp=calculateAttrImportance(tmpSubTrain, tmpSubTrain.attribute(j).name(), "MDL");   //faster implementation of MDL only for discrete data
                else
                    attrImp=calcFeatImpMDL(tmpSubTrain, j, rCaller, code);
                
                if(attrImp<=attrImpThrs[g])
                    listOfUnInFeat+=(j+1)+",";                           
            }    
            
            if(!listOfUnInFeat.equals("")){        
                remove= new Remove();
                remove.setAttributeIndices(listOfUnInFeat); //rangeList - a string representing the list of attributes. Since the string will typically come from a user, attributes are indexed from 1. e.g.: first-3,5,6-last
                remove.setInputFormat(tmpSubTrain);
                tmpSubTrain = Filter.useFilter(tmpSubTrain, remove);
                        
                remove.setAttributeIndices(listOfUnInFeat);
                remove.setInputFormat(tmpValidation);
                tmpValidation = Filter.useFilter(tmpValidation, remove);
            }
           
            predictionModel.buildClassifier(tmpSubTrain);
            Evaluation eval = new Evaluation(tmpSubTrain);
            eval.evaluateModel(predictionModel, tmpValidation);
            
            intClassAcc=(eval.correct())/(eval.incorrect()+eval.correct())*100.00;
            bestRndParam.add(new Parameters(intClassAcc,"MDL"+"@"+attrImpThrs[g], tmpSubTrain.numAttributes()-1 )); //new version only MDL method
            if(intClassAcc>maxIntAcc)
                maxIntAcc=intClassAcc;                        
        }
        
        deleteTempRFiles();  
        t1.stop();
        time[0]=t1.diff();
        bestParamPerFold.print("Num. of all parameters "+bestRndParam.size()+". ");
                
        for(int j=0;j<bestRndParam.size();){
            if(bestRndParam.get(j).getAcc()<maxIntAcc)
                bestRndParam.remove(j);
            else
                j++;
        }
                
        bestParamPerFold.println("Num. of max ACC "+bestRndParam.size()+".");                
        if(bestRndParam.size()>1)
            bestParam=bestRndParam.get((int)(Math.random()*bestRndParam.size())).getEvalMeth(); //we take random parameter out of the parameters that have same ACC
        else
            bestParam=bestRndParam.get(0).getEvalMeth();
                
        listOfUnInFeat="";
               
        for(int j=numOfAttr;j<train.numAttributes()-1;j++){
            if(!train.attribute(j).isNumeric()) //we skip evaluation of numerical features
                attrImp=calculateAttrImportance(train, train.attribute(j).name(), bestParam.split("@")[0]); //old version attribute(j) ... indexes start from 0!!!
            else
                attrImp=calcFeatImpMDL(train, j, rCaller, code);
            
            if(attrImp<=Double.parseDouble(bestParam.split("@")[1]))
                listOfUnInFeat+=(j+1)+",";                        
        }
        
        deleteTempRFiles();                
        remove= new Remove();
        remove.setAttributeIndices(listOfUnInFeat);
        remove.setInputFormat(train);
        train = Filter.useFilter(train, remove);                          

        tmp=numOfFeat(train,numOfAttr);
        feat[0]+=tmp[0];    //logical feat
        feat[1]+=tmp[1];    //threshold feat
        feat[2]+=tmp[2];    //decision rule feat (FURIA)
        feat[3]+=tmp[3];    //Cartesian feat
        feat[4]+=tmp[4];    //relational feat
        
        if(numerFeat)
            feat[5]+=tmp[5];    //numerical feat
                
        remove.setInputFormat(test);
        test = Filter.useFilter(test, remove);
        
        t1.start();
        predictionModel.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(predictionModel, test);
        t1.stop();
        time[1]=t1.diff();
                
        if(excludeUppers(predictionModel.getClass().getSimpleName()).equals("J48")){
            J48 j48=new J48(); 
            j48=(J48)(predictionModel);
            tree[0]=(int)j48.measureTreeSize();     //treeSize
            tree[1]=(int)j48.measureNumLeaves();    //numOfLeaves
            tree[2]=sumOfTermsInConstrInTree(train, numOfAttr, j48);    //sumOfTerms
                    
            numLogInTree[0]=numOfLogFeatInTree(train, numOfAttr, j48);
            numLogInTree[1]=sumOfLFTermsInConstrInTree(train, numOfAttr, j48);

            if(numerFeat)
                nN=numOfNumFeatInTree(train,numOfAttr, j48);
            
            nR=numOfRelFeatInTree(train,numOfAttr, j48);
            nC=numOfCartFeatInTree(train,numOfAttr, j48);
            furiaThrC=numOfDrThrFeatInTree(train, numOfAttr, j48);                    
        }
                
        if(excludeUppers(predictionModel.getClass().getSimpleName()).equals("FURIA")){
            FURIA fu=new FURIA();
            fu=(FURIA)(predictionModel);
            complexityF[0]=fu.getRuleset().size();
            complexityF[1]=sumOfTermsInConstrInRule(fu.getRuleset(),train);
        }
                
        attrImpListMDL.println("MDL (param. search)");
        attrImpListMDL.println("The best parameter when using "+predictionModel.getClass().getSimpleName()+": "+bestParam); //use only features above the bestParam MDL score and all attributes
        //mdlCORElearn(train, rCaller, code);
                                 
        pse.setAcc((eval.correct())/(eval.incorrect()+eval.correct())*100.00);
        pse.setFeat(feat);
        pse.setTree(tree);
        pse.setComplexityFuria(complexityF);
        pse.setNumLogFeatInTree(numLogInTree);
        if(numerFeat)
            pse.setNumFeatInTree(nN);
        pse.setRelFeatInTree(nR);
        pse.setCartFeatInTree(nC);
        pse.setFuriaThrComplx(furiaThrC);
        pse.setTime(time);
        
        return pse;
    }  
    
    public static double[][] minMaxNumAttr(Instances data) throws Exception{    //classIndex - 0,1 ... e.g.: {no-recurrence-events,recurrence-events}, we have two indexes no-recurrence-events ... 0, recurrence-events ... 1
        //explain attributes' values
        //find min and max for numeric attributes
        double[][] minMaxForNumericAttributes = new double[data.numAttributes()][2];
        for(int i = 0; i < data.numAttributes(); i++){
            minMaxForNumericAttributes[i][0] = Double.MAX_VALUE;
            minMaxForNumericAttributes[i][1] = -Double.MAX_VALUE;
        }

        //get range for numeric attributes
        Instance tempInst;
        for(int i = 0; i < data.numInstances(); i++){
            tempInst = data.instance(i);
            for(int j = 0; j < data.numAttributes(); j++){
                Attribute tempAttr = data.attribute(j);
                if(tempAttr.isNumeric()){
                    if(tempInst.value(j) < minMaxForNumericAttributes[j][0]) minMaxForNumericAttributes[j][0] = tempInst.value(j);
                    if(tempInst.value(j) > minMaxForNumericAttributes[j][1]) minMaxForNumericAttributes[j][1] = tempInst.value(j);
                }
            }
        }
        
        return minMaxForNumericAttributes;
    }
               
    public static float[][] setWeights(Instances data, float allExplanations[][], double THR){ //returns weights based on explanations ... SHAP returns float values
        float allWeights[][]=new float[allExplanations.length][allExplanations[0].length];
        float absExplanations[][]=clone2DArray(allExplanations);
        //make all explanations positive
        for(int i=0;i<absExplanations.length;i++){
            for(int j=0;j<absExplanations[i].length;j++){
                absExplanations[i][j]=Math.abs(absExplanations[i][j]);
            }
        }

        float tmpArr1D[];
        double thrTmp;
        float rowSum;
        
        for(int i=0;i<absExplanations.length;i++){
            thrTmp=0;
            tmpArr1D=absExplanations[i].clone();
            rowSum=sumArr(tmpArr1D);
            
            for(int a = 0; a < tmpArr1D.length; a++)
                tmpArr1D[a] = tmpArr1D[a]/rowSum;            
            
            Map<String, Float> mapAttrWeights=new TreeMap<>();
            for(int j=0;j<tmpArr1D.length;j++)
                mapAttrWeights.put(data.attribute(j).name(),tmpArr1D[j]);   
            
            LinkedList<Map.Entry<String, Float>> listAttrWeights= new LinkedList<>(mapAttrWeights.entrySet());
            Comparator<Map.Entry<String, Float>> comparator = Comparator.comparing(Map.Entry::getValue);
            Collections.sort(listAttrWeights, comparator.reversed()); //if we want reversed order ... descending order
            int attrIdx;

            for(Map.Entry<String, Float> me : listAttrWeights){
                if(thrTmp<THR){
                    if(me.getValue()!=0){   //if we add 0 to the sum nothing will happen
                        thrTmp+=me.getValue();
                        attrIdx=data.attribute(me.getKey()).index();
                        allWeights[i][attrIdx]=1;
                    }
                } //we are adding to the selection from largest to smallest
                else
                    break;
           }            
        }
    
        return allWeights;
    }

    public static double[][] setWeights(Instances data, double allExplanations[][], double THR){ //returns weights based on explanations ... IME returns double values
        double allWeights[][]=new double[allExplanations.length][allExplanations[0].length];
        double absExplanations[][]=clone2DArray(allExplanations);
        //make all explanations positive
        for(int i=0;i<absExplanations.length;i++){
            for(int j=0;j<absExplanations[i].length;j++){
                absExplanations[i][j]=Math.abs(absExplanations[i][j]);
            }
        }

        double tmpArr1D[];
        double thrTmp;
        double rowSum;
        
        for(int i=0;i<absExplanations.length;i++){
            thrTmp=0;
            tmpArr1D=absExplanations[i].clone();
            rowSum=sumArr(tmpArr1D);
            
            for(int a = 0; a < tmpArr1D.length; a++)
                tmpArr1D[a] = tmpArr1D[a]/rowSum;
            
            Map<String, Double> mapAttrWeights=new TreeMap<>();
            for(int j=0;j<tmpArr1D.length;j++)
                mapAttrWeights.put(data.attribute(j).name(),tmpArr1D[j]);   
            
            LinkedList<Map.Entry<String, Double>> listAttrWeights= new LinkedList<>(mapAttrWeights.entrySet());
            Comparator<Map.Entry<String, Double>> comparator = Comparator.comparing(Map.Entry::getValue);
            Collections.sort(listAttrWeights, comparator.reversed()); //if we want reversed order ... descending order
            int attrIdx;

            for(Map.Entry<String, Double> me : listAttrWeights){
                if(thrTmp<THR){
                    if(me.getValue()!=0){   //if we add 0 to the sum nothing will happen
                        thrTmp+=me.getValue();
                        attrIdx=data.attribute(me.getKey()).index();
                        allWeights[i][attrIdx]=1;
                    }
                } //we are adding to the selection from largest to smallest
                else
                    break;
           } 
        }
        
        return allWeights;
    }
    
    public static double computeNumOperation(double leftOperand, OperationNum op, double rightOperand) {
        switch(op){
            case ADD:
                return leftOperand + rightOperand;
            case SUBTRACT:
                return leftOperand - rightOperand;
            case DIVIDE:
                if(rightOperand!=0)
                    return leftOperand / rightOperand;
                else
                    return 0;
            case ABSDIFF:
                return Math.abs(leftOperand - rightOperand);
        }
        
        return 0;   
    }    
        
    public static int computeOperationTwoOperand(int leftOperand, OperationLog op, int rightOperand) {
        switch(op){
            case AND:
                return ((leftOperand==1) && (rightOperand==1)) ? 1 :0;
            case OR:
                return ((leftOperand==0) && (rightOperand==0)) ? 0 : 1;
            case EQU:
                return (leftOperand == rightOperand) ? 1 : 0;
            case XOR:
                return ((leftOperand==0) && (rightOperand==0) || (leftOperand==1) && (rightOperand==1)) ? 0 : 1;
            case IMPL:
                return ((leftOperand==1) && (rightOperand==0)) ? 0 : 1;
        }
        
        return 0;  
    }
    
    //compute relation operation
    public static int computeRelOpTwoOperand(double leftOperand, OperationRel op, double rightOperand) {
        switch(op){
            case LESSTHAN:
                return (leftOperand < rightOperand) ? 1 : 0;
            case DIFF:
                return (leftOperand != rightOperand) ? 1 : 0;
        }
        
        return 0;  
    }
    
    //for Cartesian product
    public static String mergeValues(String leftOperand, String rightOperand) {
        return leftOperand+"_x_"+rightOperand;  
    }
    
    //Cartesian product - merge all values for combination of attr1 and attr2
    public static String genDiscValues(Instances data, int idx1, int idx2){
        String allValues="";
        String tmpAttr1,tmpAttr2;
        Enumeration<Object> attr1Val=null;
        Enumeration<Object> attr2Val=null;
        
        attr1Val= data.attribute(idx1).enumerateValues();
        attr2Val= data.attribute(idx2).enumerateValues();
        
            while (attr1Val.hasMoreElements()){
                tmpAttr1=(String)attr1Val.nextElement();
                while (attr2Val.hasMoreElements()){
                    tmpAttr2=(String)attr2Val.nextElement();
                    allValues+=tmpAttr1+"_x_"+tmpAttr2+",";
                }
                attr2Val= data.attribute(idx2).enumerateValues();
            }
            allValues=allValues.substring(0, allValues.lastIndexOf(','));
            
        return allValues;
    }    
           
    public static String excludeUppers(String str){
        String strNew="";
        for(int i=0; i<str.length(); i++){
            if(Character.isUpperCase(str.charAt(i)) || Character.isDigit(str.charAt(i)))
                strNew+=str.charAt(i);                
        }
        
        return strNew;
    }
        
    //returns min index [0] of min element in the array
    public static IndexValue[] minIndexClassifiers(double [] arr){
        IndexValue[] array = new IndexValue[arr.length];
        //Fill the array 
        for( int i = 0 ; i < arr.length; i++ )
            array[i] = new IndexValue(i, arr[i]);       
        
        //Sort it 
        Arrays.sort(array, new Comparator<IndexValue>(){
            @Override
            public int compare(IndexValue a, IndexValue b){
                return Double.compare(a.v , b.v);
            }
        });
        
        return array; 
    }
        
    public static double[][] threeDtoTwoD(double [][][] arr, int idx){
        double tmp[][]=new double[arr.length][arr[0].length];
        for(int i=0;i<arr.length;i++)
            for(int j=0;j<arr[i].length;j++)
                tmp[i][j]=arr[i][j][idx];
        
        return tmp;
    }
        
    public static Map<String, Integer> sortByValue(Map<String, Integer> unsortMap) {
        List<Map.Entry<String, Integer>> list =
                new LinkedList<>(unsortMap.entrySet());

        Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1,
                               Map.Entry<String, Integer> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        Map<String, Integer> sortedMap = new LinkedHashMap<>();
        for(Map.Entry<String, Integer> entry : list)
            sortedMap.put(entry.getKey(), entry.getValue());
        
        return sortedMap;
    }
    
    public static ArrayList<String> getMostFqSubsets(double allWeights[][], Instances data,int pctVal) { //delete all those subsets that are represented less than n%
        String attrSets []=new String[allWeights.length];
        String tmpSet="";
        int x=0,count=0;
            for(int i=0;i<allWeights.length;i++){
                for(int j=0;j<allWeights[i].length;j++){
                if(allWeights[i][j]==1){
                    if(count==0)
                        tmpSet=""+j;
                    else
                        tmpSet+=","+j;  //indexes
                    count++;
                }
            }
            attrSets[x]=tmpSet;
            x++;
            tmpSet="";
            count=0;
        }

        if (ArrayUtils.contains( attrSets, "" )){
            List<String> list = new ArrayList<>(Arrays.asList(attrSets));
            list.removeAll(Collections.singleton(""));
            attrSets=list.toArray(new String[list.size()]); 
        }

        Map<String, Integer> sortedMap = sortByValue(computeWordFrequencyMap(attrSets));   //computeWordFrequencyMap(words) returns frequencyMap
        sortedMap.entrySet().removeIf(entry -> entry.getValue() <= pctVal);   //delete all those subsets that are less or equaly represented in n% of the explained cases
        printMap2(sortedMap, data); //skupine atributov in njihove frekvence

        ArrayList<String> result2 = new ArrayList(sortedMap.keySet());  //sortedMap.keySet() ... attribute indexes (frequent attributes), sortedMap.values() ... occurrence frequencies
        List removedList = new ArrayList();
        for(String temp : result2){
            if(temp.split(",").length<2) 
                removedList.add(temp);
        }
        result2.removeAll(removedList);
        
        return result2;
    }
    
    public static ArrayList<String> getMostFqSubsets(float allWeights[][], Instances data, int pctVal){ //delete all subsets that are representented less than n%... for the SHAP version
        String attrSets []=new String[allWeights.length];
        String tmpSet="";
        int x=0,count=0;
        for(int i=0;i<allWeights.length;i++){
            for(int j=0;j<allWeights[i].length;j++){
                if(allWeights[i][j]==1){
                    if(count==0)
                        tmpSet=""+j;
                    else
                        tmpSet+=","+j;  //indexes
                    count++;
                }
            }
            attrSets[x]=tmpSet;
            x++;
            tmpSet="";
            count=0;
        }  
     
        if (ArrayUtils.contains( attrSets, "" )){
            List<String> list = new ArrayList<String>(Arrays.asList(attrSets));
            list.removeAll(Collections.singleton(""));
            attrSets=list.toArray(new String[list.size()]); 
        }
        
        Map<String, Integer> sortedMap = sortByValue(computeWordFrequencyMap(attrSets));   //computeWordFrequencyMap(words) returns frequencyMap
        sortedMap.entrySet().removeIf(entry -> entry.getValue() <= pctVal);   //delete all subsets that are represented in less than n% explained instances  
        printMap2(sortedMap, data); //groups of attributes and their frequencies
             
        ArrayList<String> result2 = new ArrayList(sortedMap.keySet());  //sortedMap.keySet() ... attribute indexes (frequent attributes), sortedMap.values() ... occurrence frequencies
 
        List removedList = new ArrayList(); 
        for(String temp : result2){
            if(temp.split(",").length<2) 
                removedList.add(temp);
        }
        result2.removeAll(removedList);

        return result2;
    }
    
    public static Map<String, Integer> computeWordFrequencyMap(String[] words) {
        Map<String, Integer> result = new HashMap<>(words.length);

        for(String word : words)
            result.put(word, result.getOrDefault(word, 0) + 1);

        return result;
    }
    
    public static double[][] absArray(double arr [][]){
        for(int i=0; i<arr.length;i++)
            for(int j=0;j<arr[i].length;j++)
                arr[i][j]=Math.abs(arr[i][j]);
        
        return arr;
    }
    
    public static boolean isEmpty(String arr []){
        for(String x:arr)
            if(!x.isEmpty())
                return false;
        
        return true;
    }

    public static double[][] clone2DArray(double[][] a) {
        double[][] b = new double[a.length][];
        for(int i = 0; i < a.length; i++) {
            b[i] = new double[a[i].length];
            for(int j = 0; j < a[i].length; j++)
                b[i][j] = a[i][j];
        }
        
        return b;
    }
    
    public static float[][] clone2DArray(float[][] a) {
        float[][] b = new float[a.length][];
        for(int i = 0; i < a.length; i++){
            b[i] = new float[a[i].length];
            for(int j = 0; j < a[i].length; j++)
                b[i][j] = a[i][j];
        }
        
       return b;
    }

    public static double calculateAttrImportance(Instances data, String attName, String evaluationAlg) throws Exception{
        double idxTab[][]=null;
        Instances newData=new Instances(data);
        Remove remove= new Remove();
        remove.setAttributeIndices((newData.attribute(attName).index()+1)+",last");
        remove.setInvertSelection(true);
        remove.setInputFormat(newData);
        newData = Filter.useFilter(newData, remove); //just one attribute and class
        KononenkosMDL kMDL=new KononenkosMDL();
        //RobniksMSE rMSE=new RobniksMSE();

        switch(evaluationAlg){
            case "ReliefF": 
                idxTab=lowLevelReliefFAttrSel(newData); //evaluation of added feature - dataset consists of only added feature and class
                break;         
            case "GainRatio":
                idxTab=lowLevelGainRatioAttrSel(newData);
                break;
            case "InfoGain":
                idxTab=lowLevelInfoGainAttrSel(newData);
                break;
            case "MDL": 
                return kMDL.kononenkosMDL(newData,newData.attribute(attName).index());
            //case "MSE": return rMSE.mseNumericAttr(newData.attribute(attName).index(), newData);
            case "NoEvaluation": 
                return 999;    //just one "high" value ... more than 1 - this means we take all attributes
            default:
                System.out.println("Wrong evaluation method!");
        }
        
        return idxTab[0][1];    //importance of one feature
    }

    public static double calcFeatImpMDL(Instances data, int j, RCaller rCaller, RCode code) throws Exception{ //evaluation of one feature
        Instances newData=new Instances(data);
        Remove remove= new Remove();                
        remove.setAttributeIndices((newData.attribute(data.attribute(j).name()).index()+1)+",last");
        remove.setInvertSelection(true);
        remove.setInputFormat(newData);
        newData = Filter.useFilter(newData, remove); //just one attribute and class

        File output;  
        OutputStream out;

        double featEval=-999;
        try{            
            output = new File("Rdata/dataForROneAttr.arff");// <--- This is the result file
            out = new FileOutputStream(output);        
            DataSink.write(out, newData);
            out.close();
                        
            code.clear();
            /**********************************R code************************************************************************************/
            code.addRCode("library(CORElearn)");
            code.addRCode("library(RWeka)");
            code.addRCode("dataset <- read.arff(\"Rdata/dataForROneAttr.arff\")");
            code.addRCode("estMDL <- attrEval(which(names(dataset) == names(dataset)[length(names(dataset))]), dataset, estimator=\"MDL\",outputNumericSplits=TRUE)");   //last attribute is class attribute
            rCaller.setRCode(code);
            rCaller.runAndReturnResultOnline("estMDL"); //When you are done with this process, you must explicitly stop it!
            String tmpRcall[]=rCaller.getParser().getAsStringArray("attrEval"); //name in R "attrEval", get data from R, evaluated attributes
            featEval=Double.parseDouble(tmpRcall[0]);

            //deleteTempRFiles(); is performed after FS on validation set and after taking features for training set
            output.delete();    //delete temp file
        }
        catch(Exception ex){
            System.out.println("Error in the method mdlCORElearn");
                Logger.getLogger(FeatConstr.class.getName()).log(Level.SEVERE, null, ex);
        }
                
        return featEval;    //importance of one feature
    }

    public static DMatrix wekaInstancesToDMatrix(Instances insts) throws XGBoostError {
        int numRows = insts.numInstances();
        int numCols = insts.numAttributes()-1;

        float[] data = new float[numRows*numCols];
        float[] labels = new float[numRows];

        int ind = 0;
        for (int i = 0; i < numRows; i++){
            for (int j = 0; j < numCols; j++)
                data[ind++] = (float) insts.instance(i).value(j);
            labels[i] = (float) insts.instance(i).classValue();
        }

        DMatrix dmat = new DMatrix(data, numRows, numCols);
        dmat.setLabel(labels);
        
        return dmat;
    }

    public static float[][] removeCol(float [][] array, int colRemove){   //remove column
        int row = array.length;
        int col = array[0].length;
        float [][] newArray = new float[row][col-1];

        for(int i = 0; i < row; i++){
            for(int j = 0; j < colRemove; j++){              
                newArray[i][j] = array[i][j];                
            }
            for(int j = colRemove; j < col-1; j++){                          
               newArray[i][j] = array[i][j+1];
            }
        }
        
        return newArray;
    }

    public static float[][] someColumns(float origTab[][], int [] selectedColumns){
        float newArray[][]=new float[origTab.length][selectedColumns.length];
        for(int i=0;i<origTab.length;i++){
            for(int j=0;j<selectedColumns.length;j++)
                newArray[i][j] = origTab[i][selectedColumns[j]];            
        }
        
        return newArray;
    }

    public static void namesOfDiscAttr(Instances trainData){      
        Instances newData;
        NominalToBinary nominalToBinary = new NominalToBinary();

        weka.filters.supervised.attribute.Discretize discFilter;    //because of the same class name in different packages
        discFilter = new weka.filters.supervised.attribute.Discretize();
        Remove remove= new Remove();
        String indices="";
        boolean allDiscrete=true;

        try{ 
            for(int i=0;i<trainData.numAttributes()-1;i++)
                if(trainData.attribute(i).isNumeric()){
                    allDiscrete=false;
                    break;
                }
            
            if(!allDiscrete){
            //get indices of numeric attributes, we will discretize only numeric attributes
            for(int i=0;i<trainData.numAttributes()-1;i++)
                if(trainData.attribute(i).isNumeric())
                    indices+=(i+1)+","; 

            remove.setAttributeIndices(indices+",last");
            remove.setInvertSelection(true);
            remove.setInputFormat(trainData);
            trainData = Filter.useFilter(trainData, remove); 

            trainData.setClassIndex(trainData.numAttributes()-1); //we need class index for Fayyad & Irani's MDL
            discFilter.setInputFormat(trainData);
            newData=Filter.useFilter(trainData, discFilter);

            nominalToBinary.setInputFormat(newData); 
            newData = Filter.useFilter(newData, nominalToBinary);
            for(int i=0;i<newData.numAttributes()-1;i++)
                if(justExplain)
                    discIntervalsKD.println(newData.attribute(i).name());
                else
                    discIntervals.println(newData.attribute(i).name());
            }
            else
                if(justExplain)
                    discIntervalsKD.println("No numeric attributes.");
                else
                    discIntervals.println("No numeric attributes.");
        } 
        catch(Exception e) {
            System.out.println("ERROR in method namesOfDiscAttr"+e.toString());
        }
    }

    public static List interInfoJakulin(Instances data, List allCombSecOrd, int N) throws Exception{
        String tmpArr[];
        int idx1, idx2;
        int idx3=data.numAttributes()-1;    //class attribute
        double a,b,c,ab,ac,bc,abc;
        double intInf;
        Map<String, Double> mapIG=new TreeMap<>(Collections.reverseOrder());

        double arr1Doub[];
        double arr2Doub[];
        double arr3Doub[];

        Enumeration<Object> attr1Val=null;
        Enumeration<Object> attr2Val=null;
        Enumeration<Object> attr3Val=null;

        //get all values from the class
        attr3Val= data.attribute(idx3).enumerateValues();
        //get frequencies for class
        arr3Doub=getFrequencies(data, attr3Val, idx3);  //class

        for(int i=0;i<allCombSecOrd.size();i++){
            tmpArr=allCombSecOrd.get(i).toString().replace("[","").replace("]", "").trim().split(",");
            idx1=Integer.parseInt(tmpArr[0].trim());
            idx2=Integer.parseInt(tmpArr[1].trim());

            //get all possible values of the attribute - enumerate
            attr1Val= data.attribute(idx1).enumerateValues();
            attr2Val= data.attribute(idx2).enumerateValues();
            //get frequencies for attribute pairs
            arr1Doub=getFrequencies(data, attr1Val, idx1);
            arr2Doub=getFrequencies(data, attr2Val, idx2);

            a=ContingencyTables.entropy(arr1Doub);
            b=ContingencyTables.entropy(arr2Doub);
            c=ContingencyTables.entropy(arr3Doub);
            ab=entropy2Attr(data,idx1,idx2);
            bc=entropy2Attr(data,idx2,idx3);
            ac=entropy2Attr(data,idx1,idx3);
            abc=entropy3Attr(data,idx1,idx2,idx3);

            //interaction information by Jakulin
            intInf=ab+bc+ac-a-b-c-abc;
            mapIG.put("["+idx1+","+idx2+"]",intInf);         
        }
        mapIG=orderMapByDescValue(mapIG);

        ArrayList<String> result2 = new ArrayList(mapIG.keySet()); 
        List<String> firstNElementsList = result2.stream().limit(result2.size()< N ? result2.size() : N).collect(Collectors.toList());
  
        return firstNElementsList;
    }

    public static double entropy2Attr(Instances data, int idx1, int idx2){    
        String arr1[]=null, arr2[]=null;
        String tmpAttr1="",tmpAttr2="";
        Enumeration<Object> attr1Val=null;
        Enumeration<Object> attr2Val=null;

        attr1Val= data.attribute(idx1).enumerateValues();
        attr2Val= data.attribute(idx2).enumerateValues();

        //get frequencies for attr1
        while (attr1Val.hasMoreElements())
            tmpAttr1+=(String)attr1Val.nextElement()+",";
   
        arr1=tmpAttr1.split(",");
        double arr1Doub[]=new double[arr1.length];

        for(int i=0;i<data.numInstances();i++){
            String tmp=data.instance(i).stringValue(idx1);
            for(int j=0;j<arr1.length;j++)
                if(tmp.equals(arr1[j]))
                    arr1Doub[j]++;
        }

        //get frequencies for attr2
        while (attr2Val.hasMoreElements())
            tmpAttr2+=(String)attr2Val.nextElement()+",";
        
        arr2=tmpAttr2.split(",");
        double arr2Doub[]=new double[arr2.length];
        for(int i=0;i<data.numInstances();i++){
            String tmp=data.instance(i).stringValue(idx2);
            for(int j=0;j<arr2.length;j++)
                if(tmp.equals(arr2[j]))
                    arr2Doub[j]++;
        }
    
        double contingTab2[][]=new double[arr1.length][arr2.length];
        int idxI=0,idxJ=0;
        for(int i=0;i<data.numInstances();i++){
        String tmp1=data.instance(i).stringValue(idx1);
        String tmp2=data.instance(i).stringValue(idx2);
            for(int j=0;j<arr1.length;j++){
                if(arr1[j].equals(tmp1)){
                idxI=j;
                break;
                }
            }
            for(int k=0;k<arr2.length;k++){
                if(arr2[k].equals(tmp2)){
                idxJ=k;
                break;
                }
            }
        contingTab2[idxI][idxJ]++;
        }

        return ContingencyTables.entropy(flatten2dTo1d(contingTab2));
    }

    public static double entropy3Attr(Instances data, int idx1, int idx2, int idx3){    
        String arr1[]=null, arr2[]=null, arr3[]=null;
        String tmpAttr1="",tmpAttr2="",tmpAttr3="";
        Enumeration<Object> attr1Val=null;
        Enumeration<Object> attr2Val=null;
        Enumeration<Object> attr3Val=null;

        attr1Val= data.attribute(idx1).enumerateValues();
        attr2Val= data.attribute(idx2).enumerateValues();
        attr3Val= data.attribute(idx3).enumerateValues();

        //get frequencies for attr1
        while (attr1Val.hasMoreElements())
            tmpAttr1+=(String)attr1Val.nextElement()+",";

        arr1=tmpAttr1.split(",");
        double arr1Doub[]=new double[arr1.length];

        for(int i=0;i<data.numInstances();i++){
            String tmp=data.instance(i).stringValue(idx1);
            for(int j=0;j<arr1.length;j++)
                if(tmp.equals(arr1[j]))
                    arr1Doub[j]++;
        }

        //get frequencies for attr2
        while (attr2Val.hasMoreElements())
            tmpAttr2+=(String)attr2Val.nextElement()+",";

        arr2=tmpAttr2.split(",");
        double arr2Doub[]=new double[arr2.length];
        for(int i=0;i<data.numInstances();i++){
            String tmp=data.instance(i).stringValue(idx2);
            for(int j=0;j<arr2.length;j++)
                if(tmp.equals(arr2[j]))
                    arr2Doub[j]++;
        }

        //get frequencies for attr3
        while (attr3Val.hasMoreElements())
            tmpAttr3+=(String)attr3Val.nextElement()+",";

        arr3=tmpAttr3.split(",");
        double arr3Doub[]=new double[arr3.length];
        for(int i=0;i<data.numInstances();i++){
            String tmp=data.instance(i).stringValue(idx3);
            for(int j=0;j<arr3.length;j++)
                if(tmp.equals(arr3[j]))
                    arr3Doub[j]++;
        }

        double contingTab3[][][]=new double[arr1.length][arr2.length][arr3.length];
        int idxI=0, idxJ=0, idxK=0;
        String tmp1, tmp2, tmp3;
        for(int i=0;i<data.numInstances();i++){
            tmp1=data.instance(i).stringValue(idx1);
            tmp2=data.instance(i).stringValue(idx2);
            tmp3=data.instance(i).stringValue(idx3);
            for(int j=0;j<arr1.length;j++){
                if(arr1[j].equals(tmp1)){
                    idxI=j;
                    break;
                }
            }
            for(int k=0;k<arr2.length;k++){
                if(arr2[k].equals(tmp2)){
                    idxJ=k;
                    break;
                }
            }
            for(int l=0;l<arr3.length;l++){
                if(arr3[l].equals(tmp3)){
                    idxK=l;
                    break;
                }
            }
            contingTab3[idxI][idxJ][idxK]++;
        }

        return ContingencyTables.entropy(flatten3dTo1d(contingTab3));
    }

    //for entropy - Jakulin's method
    public static double[] getFrequencies(Instances data, Enumeration<Object> attrVal, int idx){
        String tmpAttr="";
        String arr[]=null;
        double arrDoub[];
        while (attrVal.hasMoreElements())
            tmpAttr+=(String)attrVal.nextElement()+",";
        
        arr=tmpAttr.split(",");
        arrDoub=new double[arr.length];
        for(int i=0;i<data.numInstances();i++){
            String tmp=data.instance(i).stringValue(idx);
            for(int j=0;j<arr.length;j++)
                if(tmp.equals(arr[j]))
                    arrDoub[j]++;
        }
        
        return arrDoub;
    }
    
    //for entropy - Jakulin's method
    public static double[] flatten2dTo1d(double tab[][]){
        double arr[]=new double[tab.length*tab[0].length];
        int x=0;
        for(int i=0;i<tab.length;i++){
            for(int j=0;j<tab[i].length;j++){
                arr[x]=tab[i][j];
                x++;
            }
        }
        
        return arr;
    }

    //for entropy - Jakulin's method
    public static double[] flatten3dTo1d(double tab[][][]){
        double arr[]=new double[tab.length*tab[0].length*tab[0][0].length];
        int x=0;
        for(int i=0;i<tab.length;i++){
            for(int j=0;j<tab[i].length;j++){
                for(int k=0;k<tab[i][j].length;k++){
                    arr[x]=tab[i][j][k];
                    x++;
                }
            }
        }
        
        return arr;
    }
    
    //for Jakulin's method
    public static Map<String, Double> orderMapByDescValue(Map<String, Double> unorderedMap) {
        return unorderedMap.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(Collectors.toMap(
                    Map.Entry::getKey,
                    Map.Entry::getValue,
                    (x, y) -> { throw new IllegalStateException("Unexpected merge request"); },
                    LinkedHashMap::new));
    }
    
    //set dots for psi, calculate attr. importance and get indexes of n important features, for all features set numOfImpt=Integer.MAX_VALUE
    public static void setDotsAndLine(Classifier predictionModel, Instances data,int m, boolean isClassification, int RESOLUTION, double minMaxForNumericAttributes[][], int numOfImpt) throws Exception{    //classIndex - 0,1 ... e.g.: {no-recurrence-events,recurrence-events}, we have two indexes no-recurrence-events ... 0, recurrence-events ... 1
        dotsA = new ArrayList[data.numAttributes() -1];
        dotsB = new ArrayList[data.numAttributes() -1];	

        ArrayList<Double[]> attrImp = new ArrayList<>();
        Map<String, Double> sortedMap=new TreeMap<>();

            //explain attributes' values
            for (int i = 0; i < data.numAttributes() - 1; i++){
                dotsA[i] = new ArrayList();
                dotsB[i] = new ArrayList();

                Attribute tempAttr = data.attribute(i);
                
                //for nominal attribute we calculate contributions of all its values
                if (tempAttr.isNominal()){
                    for (int j = 0; j < tempAttr.numValues(); j++){
                        double tempValue = j;              
                        double[] psi = IME.explainValueAttrImp(predictionModel, data, i, j, m, true, minMaxForNumericAttributes,isClassification);
                        dotsA[i].add(tempValue);
                        dotsA[i].add(mean(psi));    //see Formula 14 in ??trumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and information systems, 41(3), 647-665.
                        attrImp.add(ArrayUtils.toObject(psi));
                    }	
                }

                //for numerical attribute, we calculate contributions of the value between the min and max value 
                //distributions are evenly distributed, the number depends on the resolution we want for visualisation
                if (tempAttr.isNumeric()){ 
                    for (int j = 0; j < RESOLUTION; j++){		
                        double tempValue = minMaxForNumericAttributes[i][0] + j * ((minMaxForNumericAttributes[i][1] - minMaxForNumericAttributes[i][0]) / (double)RESOLUTION);                
                        double[] psi = IME.explainValueAttrImp(predictionModel, data, i, tempValue, m, false, minMaxForNumericAttributes, isClassification);
                        dotsA[i].add(tempValue);
                        dotsA[i].add(mean(psi));    //see Formula 14 in ??trumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and information systems, 41(3), 647-665.
                        attrImp.add(ArrayUtils.toObject(psi));
                    }	
                }

                dotsB[i].add(Math.sqrt(varArrList(attrImp,meanArrList(attrImp)))); 
                sortedMap.put(tempAttr.name(),Math.sqrt(varArrList(attrImp,meanArrList(attrImp))));    

                attrImp.clear();
            }

            //print sorted map
            LinkedList<Map.Entry<String, Double>> list= new LinkedList<>(sortedMap.entrySet());
            Comparator<Map.Entry<String, Double>> comparator = Comparator.comparing(Map.Entry::getValue);
            Collections.sort(list, comparator.reversed()); //if we want reversed order ... descending order
            for(Map.Entry<String, Double> me : list){ 
                System.out.printf(" %4.4f %s\n",me.getValue(), me.getKey()); 
//                    logFile.printf(" %4.4f %s\n",me.getValue(), me.getKey());
            }

            //get attribute importance
            double attrImpVal[]=new double[data.numAttributes()-1];
            for(int i=0; i<data.numAttributes()-1;i++){
                for(int j=0;j<dotsB.length;j++){
                    attrImpVal[i]=dotsB[i].get(0);
                }
            }
            double attrImpCp[]=attrImpVal.clone();
            Arrays.sort(attrImpVal);

            //get id of attributes that are above nth attr
            if(attrImpVal.length>numOfImpt){
                nThHigh="";
                for(int i=0; i<attrImpCp.length;i++){
                    if(attrImpCp[i]>=attrImpVal[attrImpVal.length-numOfImpt] && attrImpCp[i]>0){  //if among the n-th most important is also the one that is not informative (equal to 0) it is not drawn
                        nThHigh+=(i+1)+",";
                    }
                }
                if(nThHigh.contains(","))
                    nThHigh=nThHigh.substring(0, nThHigh.lastIndexOf(','));
            }
    } 
    
    public static void visualizeModelInstances(Classifier predictionModel, Instances data, boolean isClassification, int RESOLUTION, int numOfImpt, int fromInst, int toInst) throws Exception{
        predictionModel.buildClassifier(data);
        System.out.println("Attribute importance using explanation method IME, prediction alg. is "+predictionModel.getClass().getSimpleName());
        boolean visExplFC;  //visualisation before or after FC
        String modelName=predictionModel.getClass().getSimpleName();        
        String classValueName=(new Instances(data,0,1)).instance(0).classAttribute().value(classToExplain);  // get class name of the explained class
        
        //visualise model        
        String outputDir, fName, format;  
        
        datasetName=fileName; 
        if(datasetName.contains(".arff") && datasetName.contains("LFeat-"))
            datasetName=datasetName.substring(0, datasetName.indexOf("LFeat-")+"LFeat-".length()-1);
        if(datasetName.contains(".arff") && !datasetName.contains("LFeat-"))                   
            datasetName=datasetName.substring(0, datasetName.indexOf(".arff"));
        
        format=".eps";  
        if(isClassification)
            fName=modelName + "_" + datasetName + "_model-class_"+classValueName;
        else
            fName=modelName + "_" + datasetName + "_model-regr";
     
        /***************************************calculate attribute importance of thr changed dataset - added features***************************************/       
        setDotsAndLine(predictionModel, data,N_SAMPLES,isClassification,RESOLUTION, minMaxNumAttr(data),numOfImpt); //sets also parameter nThHigh ... indexes for attributes with high importance
        
        visExplFC = justExplain && visualisation;
        
        outputDir = visExplFC ? "visualisation/afterFC/eps/" : "visualisation/beforeFC/eps/";
        //draw only numOfImpt or less most informative attributes in the model
        /*Model visualisation*/
        if(data.numAttributes()-1 > numOfImpt){
            Remove remove= new Remove();
            Instances dataCp=new Instances(data);
            remove= new Remove();
            String attr=nThHigh;
            attr=attr+","+dataCp.numAttributes(); //+1 ... class
            remove.setAttributeIndices(attr);
            remove.setInvertSelection(true);    //we need to remove unselected attributes - invert selection
            remove.setInputFormat(dataCp);
            dataCp = Filter.useFilter(dataCp, remove);      //select only attributes that are in the model
            dataCp.setClassIndex(dataCp.numAttributes()-1); //set class attribute 
            String tmp[]=attr.split(",");
            int[] integers = new int[tmp.length]; 
            for (int i = 0; i < integers.length; i++)
                integers[i] = Integer.parseInt(tmp[i]); 
        
            ArrayList<Double>[] dotsACp = new ArrayList[dataCp.numAttributes()-1];
            ArrayList<Double>[] dotsBCp = new ArrayList[dataCp.numAttributes()-1];

            int x=0;
            for(int i=0;i<dotsA.length;i++){
                int m=i+1;
                if(ArrayUtils.contains(integers,m)){
                    dotsACp[x]=dotsA[i];
                    dotsBCp[x]=dotsB[i];
                    x++;
                }
            }
            Visualize.modelVisualToFileAttrImptLine(outputDir +fName+format, modelName, datasetName, dataCp, dotsACp, dotsBCp,isClassification,RESOLUTION,classToExplain,"A4", visExplFC);         
        }
        else if(data.numAttributes()-1 <= 6)   //for format A4, for pdf and png format
            Visualize.modelVisualToFileAttrImptLine(outputDir +fName+format, modelName, datasetName, data, dotsA, dotsB,isClassification,RESOLUTION,classToExplain,"A4", visExplFC);     
        else
            Visualize.modelVisualToFileAttrImptLine(outputDir +fName+format, modelName, datasetName, data, dotsA, dotsB,isClassification,RESOLUTION,classToExplain,"AA", visExplFC); //AA just smth. different of A4
        

        Visualize.attrImportanceVisualizationSorted(outputDir +fName+"-attrImp"+format, modelName, datasetName, data, drawLimit, dotsB,isClassification,RESOLUTION,"AA",visExplFC);
        /*Instance visualisation*/
        //pdf, png -> model
        if(pdfPng){
            if(visExplFC){
                covertToPdfAndPng(fName,format, "visualisation/afterFC/eps/","visualisation/afterFC/pdf/","visualisation/afterFC/png/");
                covertToPdfAndPng(fName+"-attrImp",format, "visualisation/afterFC/eps/","visualisation/afterFC/pdf/","visualisation/afterFC/png/"); //plot attribute importance
            }
            else{
                covertToPdfAndPng(fName,format, "visualisation/beforeFC/eps/","visualisation/beforeFC/pdf/","visualisation/beforeFC/png/");
                covertToPdfAndPng(fName+"-attrImp",format, "visualisation/beforeFC/eps/","visualisation/beforeFC/pdf/","visualisation/beforeFC/png/"); //plot attribute importance
            }
        }  

        for(int i = fromInst; i <= toInst; i++){
            outputDir = visExplFC ? "visualisation/afterFC/eps/" : "visualisation/beforeFC/eps/";
            double[] instanceExplanation = IME.explainInstance(predictionModel, data, new Instances(data,(i-1),1), N_SAMPLES, isClassification, classToExplain);	
            double pred = -1;   

            if (isClassification)
                pred = predictionModel.distributionForInstance((new Instances(data,(i-1),1)).instance(0))[classToExplain];
            else
                pred = predictionModel.classifyInstance((new Instances(data,(i-1),1)).instance(0));					
            format=".eps";
            if(isClassification)
                fName=modelName + "_" + datasetName + "_instance_" + (i)+ "-class_"+classValueName;
            else
                fName=modelName + "_" + datasetName + "_instance_" + (i)+ "-regr";
            Visualize.instanceVisualizationToFile(outputDir +fName+format, modelName, datasetName, new Instances(data,(i-1),1), i, topHigh, instanceExplanation, pred, classToExplain, isClassification, visExplFC);
        
            //pdf, png -> instance(s)
            if(pdfPng)
                if(visExplFC)
                    covertToPdfAndPng(fName, format,"visualisation/afterFC/eps/","visualisation/afterFC/pdf/","visualisation/afterFC/png/");
                else
                    covertToPdfAndPng(fName, format,"visualisation/beforeFC/eps/","visualisation/beforeFC/pdf/","visualisation/beforeFC/png/");
        } 
    }
    
    public static void covertToPdfAndPng(String fName, String inFormat,String inDirEps, String outDirPdf, String outDirPng) throws Exception { 
        FileOutputStream fos;
        PSDocument document;
        File f;
        PDFConverter converter;
        PDFDocument documentNew;
        SimpleRenderer renderer;
        List<Image> images;
        String format;
        
        //load PostScript document
        document = new PSDocument();

        f=new File(inDirEps+fName+inFormat);
        document.load(f);

        //create OutputStream
        outDirPdf=(justExplain && visualisation) ? "visualisation/afterFC/pdf/" : "visualisation/beforeFC/pdf/";
        format=".pdf";
        fos = new FileOutputStream(new File(outDirPdf+fName+format));
        //create converter
        converter = new PDFConverter();
        converter.setPaperSize(PaperSize.A4);  

        //convert eps to pdf
        converter.convert(document, fos);

        //convert pdf to png    
        documentNew = new PDFDocument();
        documentNew.load(new File(outDirPdf+fName+format));

        renderer = new SimpleRenderer();            
        renderer.setResolution(300);// set resolution (in DPI)

        //render
        outDirPng=(justExplain && visualisation) ? "visualisation/afterFC/png/" : "visualisation/beforeFC/png/"; //does not override existing image ... delete image with the same name before run
        format=".png";
        images = renderer.render(documentNew);
        for (int j = 0; j < images.size(); j++) {
            ImageIO.write((RenderedImage) images.get(j), "png", new File(outDirPng+fName+format));  //save images to png format
        }
    }
    
    //combinations of order N
    public static List allCombOfOrderN(List<String> allComb, int N){ 
        Set<String> hSet = new HashSet<>();   
        List newTmpComb;
        String tmpArrCG[];
        for(int i=0;i<allComb.size();i++){
            tmpArrCG=allComb.get(i).split(",");		
            newTmpComb=Arrays.asList(Generator.combination(tmpArrCG).simple(N).stream().toArray());
            hSet.addAll(newTmpComb);
        }
        ArrayList<String> finalComb = new ArrayList<>(hSet);

        return finalComb;
    }

    public static double sumArr(double tab []){
        return Arrays.stream(tab).sum();
    }

    public static float sumArr(float tab []){
        float sum=0;
        for(int i=0;i<tab.length;i++)
            sum+=tab[i];
        
        return sum;
    }
    
    public static double mean(double[] d){
	double sum = 0;
	for (int i = 0; i < d.length; i++) 
            sum += d[i];
	
        return sum / d.length;
    }
    
    public static double mean(int[] d){
	double sum = 0;
	for (int i = 0; i < d.length; i++) 
            sum += d[i];
	
        return sum / d.length;
    }
    
    public static double mean(long[] d){
	double sum = 0;
	for (int i = 0; i < d.length; i++) 
            sum += d[i];
	
        return sum / d.length;
    }
        
    public static double meanArrList(ArrayList<Double[]> arrList){
	double sum = 0;
        int count =0;
	for (int i = 0; i < arrList.size(); i++){
            for(int j = 0; j < arrList.get(i).length; j++){
                sum += arrList.get(i)[j];
                count++;
            }
        }	
        
        return sum / count;
    }
    
    public static double var(double[] d, double m){     //variance
	double sum = 0;
	for (int i = 0; i < d.length; i++) 
            sum += (d[i] - m) * (d[i] - m);
        
        return sum / d.length;
    }
    
    public static double var(int[] d, double m){     //variance
	double sum = 0;
	for (int i = 0; i < d.length; i++) 
            sum += (d[i] - m) * (d[i] - m);
        
        return sum / d.length;
    }
    
    public static double var(long[] d, double m){     //variance
	double sum = 0;
	for (int i = 0; i < d.length; i++) 
            sum += (d[i] - m) * (d[i] - m);
        
        return sum / d.length;
    }
    
    public static double varArrList(ArrayList<Double[]> arrList, double m){     //variance
	double sum = 0;
        int count =0;
	for (int i = 0; i < arrList.size(); i++){
            for(int j=0; j < arrList.get(i).length;j++){
                sum += (arrList.get(i)[j]-m) *(arrList.get(i)[j]-m);
                count++;
            }
        }	
        
        return sum / (count);
    }
    
    public static double varArrList2(ArrayList<Double> arrList, double m){     //variance
	double sum = 0;
        int count =0;
	for (int i = 0; i < arrList.size(); i++){
                sum += (arrList.get(i)-m) *(arrList.get(i)-m);
                count++;
        }
        
        return sum / (count);
    }
    
    public static double var(Vector d){
        double m1= 0;
        for (int i = 0; i < d.size(); i++)
            m1 += (Double)d.elementAt(i);
        
        m1 /= d.size();

        double sum = 0;
        for (int i = 0; i < d.size(); i++) 
            sum += ((Double)d.elementAt(i) - m1) * ((Double)d.elementAt(i)- m1);
    
        return sum / d.size();
    }
    
    public static String rnd3(double d) {
    	DecimalFormat twoDForm = new DecimalFormat("0.000");
        
        return twoDForm.format(d).replace(",",".");
    }
    
    public static double round(double value, int precision){
        int scale=(int) Math.pow(10, precision);
        
        return (double) Math.round(value * scale) / scale;
    }
    
    public static void print1d(double tab []){
        for(int i=0;i<tab.length;i++){
            System.out.print(rnd3(tab[i])+"\t");
                logFile.print(rnd3(tab[i])+"\t");
	}
    }

    public static void print2d(double arr [][]) {
        for(int i=0; i<arr.length; i++){
            for(int j=0; j<arr[i].length;j++){
                System.out.printf("%9.3f",arr[i][j]);
            }
            System.out.println();
        }
    }
    
    public static void print2d(float arr [][]) {
        for(int i=0; i<arr.length; i++){
            for(int j=0; j<arr[i].length;j++){
                System.out.printf("%9.3f",arr[i][j]);
            }
            System.out.println();
        }
    }
    
    public static void printList(List ar1){
        for(int i=0;i<ar1.size();i++)
            System.out.println(ar1.get(i)+" ");            	
    }    
    
    public static void printFqAttrOneRow(List<String> ar1, Instances data){
        String [] tmp;
        for(String el: ar1){
            tmp=el.split(",");
            for(int i=0;i<tmp.length;i++)
                if(justExplain)
                    impGroupsKD.print(data.attribute(Integer.parseInt(tmp[i])).name()+" ");            
                else
                    impGroups.print(data.attribute(Integer.parseInt(tmp[i])).name()+" ");            
            if(justExplain)
                impGroupsKD.print("~#~ ");
            else
                impGroups.print("~#~ ");
	}
    } 
    
    public static int[] printMaxConstructLength(List<String> ar1){
        String [] tmp;
        int max=0;
        int sum=0;
        int sumMax[]=new int[2];
        for(String el: ar1){
            tmp=el.split(",");
            sum+=tmp.length;
            if(tmp.length>max)
                max=tmp.length;
        }
        sumMax[0]=sum;
        sumMax[1]=max;
        
        return sumMax;
    } 
    
    //for Jakulin's method - to get attribute names; just for info
    public static void printAttrNamesIntInf(Instances data, List list){
        String tmpArr[];
        for(int i=0;i<list.size();i++){    
            tmpArr=list.get(i).toString().replace("[","").replace("]", "").trim().split(",");
            System.out.println(data.attribute(Integer.parseInt(tmpArr[0].trim())).name()+" - "+data.attribute(Integer.parseInt(tmpArr[1].trim())).name());
                impGroups.println(data.attribute(Integer.parseInt(tmpArr[0].trim())).name()+" - "+data.attribute(Integer.parseInt(tmpArr[1].trim())).name());
        }
    }   
            
    public static void printMap1(Map<String, Integer> frequencyMap){
        Iterator<String> tmpIterator1 = frequencyMap.keySet().iterator();
            while (tmpIterator1.hasNext()){
                String str = tmpIterator1.next();
                System.out.println(str + ": " + frequencyMap.get(str));
                    logFile.println(str + ": " + frequencyMap.get(str));
            }
    }
    
    public static void printMap2(Map<String, Integer> frequencyMap, Instances data){
        Iterator<String> tmpIterator1 = frequencyMap.keySet().iterator();
        String [] tmp1; String str;
            while(tmpIterator1.hasNext()){
                str = tmpIterator1.next();
                tmp1=str.split(",");
                if(justExplain)
                    impGroupsKD.print("\t");
                else
                    impGroups.print("\t");
                for(int i=0;i<tmp1.length;i++){
                    if(i==tmp1.length-1){
                        if(justExplain)
                            impGroupsKD.print(data.attribute(Integer.parseInt(tmp1[i])).name());
                        else
                            impGroups.print(data.attribute(Integer.parseInt(tmp1[i])).name());
                    }
                    else{
                        if(justExplain)
                            impGroupsKD.print(data.attribute(Integer.parseInt(tmp1[i])).name()+" | ");
                        else
                            impGroups.print(data.attribute(Integer.parseInt(tmp1[i])).name()+" | ");
                    }
                }
                if(justExplain)
                    impGroupsKD.println(": " + frequencyMap.get(str));
                else
                    impGroups.println(": " + frequencyMap.get(str));
            }
    }
    
    public static void deleteXGBdll(){
        File tempF=new File(tmpDir);
            for(File fileTmp : tempF.listFiles()){
                if(fileTmp.getName().contains("xgboost4j")){
                    fileTmp.delete();                
                }
            }
    }
    
    public static void deleteTempRFiles(){
        File tempF=new File(tmpDir);
            for(File fileTmp : tempF.listFiles()){
                if(fileTmp.getName().contains("RControl") || fileTmp.getName().contains("ROutput") || fileTmp.getName().contains("getTmpDir")){
                    fileTmp.delete();                
                }
            }
    }
}
