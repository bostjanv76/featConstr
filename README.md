# Explainable Feature Construction (EFC)
We present a novel method for efficient constructive induction. The method significantly speeds up the process of learning new, powerful features for predictive modelling and improves prediction performance by reducing the search space, coins a novel approach to using instance explanations for feature construction (FC). The proposed feature construction method contributes to more successful and more comprehensible prediction models, which are becoming an important part of scientific, industrial, and societal processes.

The proposed EFC[^1] method consist of the following four steps:
1.	Explanation of a model predictions for individual instances.
2.	Identification of groups of attributes that commonly appear together in explanations.
3.	Efficient creation of constructs from the identified groups.
4.	Evaluation of constructs and selection of the best as new features.

![methodology_v35New-scenario2](https://user-images.githubusercontent.com/88408507/155245515-200af9d9-940b-49fb-a508-3011fca4bc3c.svg)

# Using EFC
## Basic use
Put the desired dataset(s) in demo folder (datasets/demo) and run the program. If you want to test the method on one of the experimental datasets (toy, artificial, UCI, real), then just uncomment the line where the desired experimental dataset(s) is/are and comment the line with the folder where the demonstration dataset(s) is/are.
```java
    /*****demo datasets*****/
    folder = new File("datasets/demo");
    /*****toy datasets*****/       
    //folder = new File("datasets/toy");
    /*****artificial datasets*****/ 
    //folder = new File("datasets/artificial");
    /*****UCI datasets*****/
    //folder = new File("datasets/uci");
    /*****real dataset - credit score*****/       
    //folder = new File("datasets/real");
```

**The default settings are following:**
- EFC is enabled when flag variables justExplain, visualisation, exhaustive, and jakulin are set to **false**. 
- Black-box prediction algorithm: **XGBoost**
-	XGBoost parameters: 
    -	number of decision trees is 100 (**numOfRounds=100**) 
    -	size of decision trees is 3 (**maxDepth=3**) 
    -	shrinkage is 0.3 (**eta=0.3**) 
    -	pseudo-regularization hyperparameter is 1 (**gamma=1**)
-	Explanation method: **Tree SHAP**
-	Explain just instances from the minority class (**explAllClasses=false**).
-	Types of features: **logical operator features, decision rules features, threshold features**
    -   logical operators: **EQU, XOR, IMPL**
-	Testing classifiers: decision trees (**j48**), Naïve Bayes (**NB**), support vector machines (**SVM**), k-nearest neighbours (**kNN**), decision rules (**FURIA**), and random forest (**RF**)
-	Prediction model evaluation: **10-fold CV**

Results are printed and saved in the logs/efc folder.
-	_impGroups-"time-date"_ – the file that stores groups of attributes that co-occur in explanations
-	_report-"time-date"_ – the file that stores ACC and learning time of all classifiers for all method settings
-	_attrImpListMDL-"time-date"_ – the file that stores MDL[^2] scores of attributes/features; attribute/feature evaluation step
-	_attrImpListReliefF-"time-date"_ – the file that stores ReliefF[^3] scores of attributes/features; attribute/feature evaluation step
-	_discretizationIntervals-"time-date"_ – the file that stores discretization intervals of numerical attributes for calculating logical features
-	_params-"time-date"_ – the file that stores the best parameters used in the FS setting of the method

## Advanced use
**Choosing explanation method**

We can choose between two explanation methods (SHAP[^4] and IME[^5]). If the explanation method Tree SHAP is chosen then flag variable treeSHAP must be set to **true** (treeSHAP=true), otherwise the explanation method IME is automatically chosen.

**Additional settings**

When the IME explanation method is chosen, we can choose between different sampling methods (equalSampling, adaptiveSamplingSS<sup><code>[**\***](#sumOfSamples)</code></sup>, adaptiveSamplingAE<sup><code>[**\*\***](#aproxError)</code></sup>, aproxErrSampling). The list of included classifiers for the prediction and visualisation model (if IME is chosen) is: rf, mp, svmLin, svmPoly, svmRBF, nb, j48, and furia.

```java
    method=IMEver.adaptiveSamplingSS;   //sampling method
    predictionModel=rf;                 //model (based on the chosen classifier) for explanations; use of the IME method
    visualModel=rf;                     //model (based on the chosen classifier) for visualisations; use of the IME method
```

<sup><code><a id='sumOfSamples'>**\***</a></code></sup><sub><sup>_Stopping criteria is sum of samples._</sub></sup>

<sup><code><a id='aproxError'>**\*\***</a></code></sup><sub><sup>_Stopping criteria is approximation error for all attributes._</sub></sup>

When selecting feature types, at least one feature type must be activated (**true**). Different operators can be used for different types of features. The full set of implemented logical operators is {AND, OR, EQU, XOR, IMPL}, relational operators {LESSTHAN, DIFF}, and numerical operators {ADD, SUBTRACT, DIVIDE, ABSDIFF}. The depth of the feature construction is only controlled for conjunctions and disjunctions; for example, depth (**featDepth**) 3 means depth 2 and 3.

```java
    logFeat=true;                           //enable/disable generation of logical operators features
    decRuleFeat=true;                       //enable/disable generation of decision rules features
    thrFeat=true;                           //enable/disable generation of threshold features
    relatFeat=true;                         //enable/disable generation of relational features
    cartFeat=true;                          //enable/disable generation of Cartesian product features
    numerFeat=true;                         //enable/disable generation of numerical features
    operationLogUse={"AND","OR"};           //choose logical operators   
    operationRelUse={"LESSTHAN","DIFF"};    //choose relational operators                
    operationNumUse={"ADD","SUBTRACT"};     //choose numerical operators
```

**Other EFC parameters**

Groups of attributes that commonly appear together in explanations can be regulated with a noise threshold (**noiseThr**). The noise threshold determines the minimal required empirical support for candidate groups (for FC), i.e. the minimal required frequency to accept the attribute group as important.

```java
    explAllClasses=true;    //explain all (true) classes or just minority class (false)
    thrL=0.1;               //lower weight threshold 
    thrU=0.8;               //upper weight threshold
    step=0.1;               //step for traversing all thresholds from thrL to thrU
    NOISE=1;                //noiseThr=(numInst*NOISE)/100.0; NOISE=0 (we take all groups of attributes)
    evalFeatDuringFC=false; //enable/disable feature evaluation during FC process
    featThr=0.05;           //evaluation threshold (use of MDL); useful only when evalFeatDuringFC is enabled
    folds=10;               //evaluation of models, folds=1 means no CV and using split in ratio listed below
    splitTrain=5;           //5 ... 80%:20%, 4 ... 75%:25%, 3 ... 66%:33%; useful only when folds=1
```
Flag variables that allow additional options (_knowledge discovery_, _visualisation_, _FC based on exhaustive search_, _FC based on interaction information_) must be set to **false**. Additional flag variables are **groupsByThrStat** and **writeAccByFoldsInFile**. The first enables statistics to count groups of attributes identified by the EFC for each fold for a given threshold. The results are stored in the _groupsStat-"time-date"_ file in the logs/efc folder. The second enables the storage of ACC for each fold for each prediction algorithm; the results are stored in _algorithmName-byFolds-"time-date"_.

```java
    justExplain=false;
    visualisation=false;
    exhaustive=false;
    jakulin=false;
```     

----- Exhaustive search (generate all possible combinations between attributes)

Flag variable exhaustive must be set to **true** and jakulin, justExplain and visualisation to **false**. Results are printed and saved in the logs/exhaustive folder.

```java
    justExplain=false;
    visualisation=false;
    exhaustive=true;
    jakulin=false;
```     
----- FC based on interaction information 

Flag variables jakulin and exhaustive must be set to **true** and flags justExplain and visualisation to **false**. Calculate interaction information between all combinations of attributes[^6] and construct features. Results are printed and saved in the logs/jakulin folder.

```java
    justExplain=false;
    visualisation=false;
    exhaustive=true;
    jakulin=true;
```
----- Knowledge Discovery (construct features from the whole dataset and evaluate them)

To activate Knowledge Discovery (KD), the flag variable justExplain must be set to **true** and visualisation to **false**. New constructs of FC are evaluated by MDL scores. The results are printed and saved in the subfolder kd (logs/kd). To save new constructs with original attributes, the flag variable **saveConstructs** must be activated (saveConstructs=true) - the file _"dataset name"-origPlusXLFeat-"time-date".arff_ is created; X in the file name indicates the feature level {1,2}<sup><code>[**&#8224;**](#featLevel)</code></sup>. If the flag variable **renameGenFeat** is activated, the constructed features are renamed<sup><code>[**&#8224;&#8224;**](#renamedFeat)</code></sup> and saved in another file _"dataset name"-origPlusRenXLFeat-"time-date".arff_ - this dataset serves for the next<sup><code>[**&#8224;&#8224;&#8224;**](#datNextLevel)</code></sup> level construction. 
-	_impGroups-"time-date"_ – the file that stores groups of attributes that co-occur in explanations
-	_attrImpListMDL-"time-date"_ – the file that stores MDL scores of attributes/features; attribute/feature evaluation step
-	_attrImpListReliefF-"time-date"_ – the file that stores ReliefF scores of attributes/features; attribute/feature evaluation step
-	_discretizationIntervals-"time-date"_ – the file that stores discretization intervals of numerical attributes for calculating logical features

```java          
    justExplain=true;
    visualisation=false;
```    
<sup><code><a id='featLevel'>**&#8224;**</a></code></sup><sub><sup>_First level features are generated from attributes, second level features are generated from attributes and first level features._</sub></sup>

<sup><code><a id='renamedFeat'>**&#8224;&#8224;**</a></code></sup><sub><sup>_Features are renamed in the form FSLX, where S is the serial number of the feature and X the level of feature construction; the renamed features are explained in the _names-X-level-feat-"time-date".dat_ file._</sub></sup>

<sup><code><a id='datNextLevel'>**&#8224;&#8224;&#8224;**</a></code></sup><sub><sup>_For the construction of second level features, the dataset "dataset name"-origPlusRen1LFeat-"time-date".arff must be used._</sub></sup>

----- Visualisation

To visualise explanations of instances from visFrom to visTo, the flag variable visualisation must be set to **true**. Besides that, attribute importance is also visualised. The default prediction algorithm is Random Forest and for the explanations the IME method is used. For each explained instance, only the topHigh (default: 6) attributes with the highest absolute contributions are shown. All images are saved in the visualisation folder<sup><code>[**&#8225;**](#folderStruct)</code></sup>. For attribute importance[^7], which is based on the instance explanations, we draw (max.) 20 of the most important attributes.

<sup><code><a id='folderStruct'>**&#8225;**</a></code></sup><sub><sup>_The folder consists of two subfolders (beforeFC and afterFC); visualisation can be performed before (justExplain=false) or after (justExplain=true) FC._</sub></sup>

```java
    justExplain=false;      //false - visualisation of original dataset, true - visualisation after FC
    visualisation=true;     //visualisation of explanations using IME method
    visFrom=1, visTo=10;    //visualise instances from visFrom to visTo
    drawLimit=20;           //we draw (max.) 20 the most important attributes (attribute importance visualisation)
    topHigh=10;             //visualise features with highest contributions (instance explanation visualisation)
    RESOLUTION=100;         //density for model visualisation
    N_SAMPLES=100;          //if we use equalSampling ... number of samples  
    pdfPng=true;            //besides eps, print also pdf and png format
```

## Requirements
**Java**
* JDK
* NetBeans

**R**
* R with the packages CORElearn and RWeka must be installed on the system; to use the MDL evaluation measure in the attrEval function.

:memo: **Note:** Visual C++ Redistributable must also be installed (because of xgboost4j.jar) and the library gsdll64.dll must be placed in the system32 folder (for converting eps to pdf).

## Authors
EFC was created by Boštjan Vouk, Marko Robnik-Šikonja and Matej Guid.

[^1]: Vouk, B., Guid, M., & Robnik-Šikonja, M. (2023). Feature construction using explanations of individual predictions. Engineering Applications of Artificial Intelligence, 120, 105823. https://doi.org/jtnn
[^2]: Kononenko, I. (1995). On biases in estimating multi-valued attributes. In Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence (I), 1034–1040. https://bit.ly/3HkEGhT
[^3]: Robnik-Šikonja, M., & Kononenko, I. (2003). Theoretical and empirical analysis of Relieff and RRelieff. Machine Learning. https://doi.org/d63s9s
[^4]: Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in neural information processing systems 30 (NIPS 2017) (pp. 4765–4774). Curran Associates, Inc. https://bit.ly/3zhk5Is
[^5]: Štrumbelj, E., & Kononenko, I. (2010). An efficient explanation of individual classifications using game theory. Journal of Machine Learning Research, 11, 1–18.
[^6]: Jakulin, A. (2005). Machine learning based on attribute interactions [Doctoral dissertation, University of Ljubljana]. ePrints.FRI. https://bit.ly/3eiJ18x
[^7]: Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and Information Systems, 41(3), 647–665. https://doi.org/f6pnsr
