# Explainable Feature Construction (EFC)
We present a novel method for efficient constructive induction. The method significantly speeds up the process of learning new, powerful features for predictive modelling and improves prediction performance by reducing the search space, coins a novel approach to using instance explanations for feature construction (FC). The proposed feature construction method contributes to more successful and more comprehensible prediction models, which are becoming an important part of scientific, industrial, and societal processes. 

The proposed EFC method consist of the following four steps:
1.	Explanation of a model predictions for individual instances.
2.	Identification of groups of attributes that commonly appear together in explanations.
3.	Efficient creation of constructs from the identified groups.
4.	Evaluation of constructs and selection of the best as new features.

![methodology_v35New-scenario2](https://user-images.githubusercontent.com/88408507/155245515-200af9d9-940b-49fb-a508-3011fca4bc3c.svg)

# Using EFC
## Basic use
Put the desired dataset(s) in demo folder (datasets/demo) and run the program. If you want to test the method on one of the experimental datasets (artificial, UCI, real), then just uncomment the line where the desired experimental dataset(s) is/are and comment the line with the folder where the demonstration dataset(s) is/are.
```java
    /*****demo datasets*****/
    folder = new File("datasets/demo");
    /*****artificial datasets*****/ 
    //folder = new File("datasets/artificial");
    /*****UCI datasets*****/
    //folder = new File("datasets/uci");
    /*****real dataset - credit score*****/       
    //folder = new File("datasets/real");
```

**The default settings are following:**
- EFC is enabled when flag variables jakulin and exhaustive are set to **false**. 
- Black-box prediction algorithm: **XGBoost**
-	XGBoost parameters: 
    -	number of decision trees is 100 (**numOfRounds=100**) 
    -	size of decision trees is 3 (**maxDepth=3**) 
    -	shrinkage is 0.3 (**eta=0.3**) 
    -	pseudo-regularization hyperparameter is 1 (**gamma=1**)
-	Explanation method: **Tree SHAP**
-	Explain just instances from the minority class (**explAllClasses=false**).
-	Do not calculate numerical features (**numerFeat=false**).
-	Testing classifiers= decision trees (**j48**), Naïve Bayes (**NB**), decision rules (**FURIA**), and random forest (**RF**)
-	Prediction model evaluation: **10-fold CV**.

Results are printed and saved in the logs folder.
-	_impGroups-"time-date"_ – the file that stores groups of attributes that co-occur in explanations
-	_report-"time-date"_ – the file that stores ACC and learning time of all classifiers for all method settings
-	_attrImpListMDL-"time-date"_ – the file that stores MDL scores of attributes after FC; attribute evaluation step
-	_discretizationIntervals-"time-date"_ – the file that stores discretization intervals of numerical attributes for calculating logical features
-	_params-"time-date"_ – the file that stores the best parameters used in the FS setting of the method

## Advanced use
**Choosing explanation method**

We can choose between two explanation methods (SHAP[^1] and IME[^2]). If the explanation method Tree SHAP is chosen then flag variable treeSHAP must be set to **true** (treeSHAP=true), otherwise the explanation method IME is automatically chosen.

**Other settings**

When IME explanation method is chosen then we can choose between variations of sampling methods (equalSampling, adaptiveSamplingSS<sup>*</sup>, adaptiveSamplingAE<sup>**</sup>, aproxErrSampling).

```java
    method=IMEver.adaptiveSamplingSS;   //sampling method
    numerFeat=true;                     //enable generation of numerical features 
    explAllClasses=true;                //explain all (true) classes or just minority class (false)
    thrL=0.1;                           //lower weight threshold 
    thrU=0.8;                           //upper weight threshold
    step=0.1;                           //step for traversing all thresholds from thrL to thrU
    folds=10;                           //evaluation of models, folds=1 means no CV and using split in ratio listed below
    splitTrain=5;                       //5 ... 80%:20%, 4 ... 75%:25%, 3 ... 66%:33%; useful only when folds=1 (split validation)
```

<sup>*</sup>_stopping criteria is sum of samples_

<sup>**</sup>_stopping criteria is approximation error for all attributes_

Flag variables that allow additional options (_knowledge discovery_, _visualisation_, _FC based on exhaustive search_, _FC based on interaction information_) must be set to **false**.

```java
    justExplain=false;
    visualisation=false;
    exhaustive=false;
    jakulin=false;
```     

----- Exhaustive search (generate all possible combinations between attributes)

Flag variable exhaustive must be set to **true** and jakulin, justExplain and visualisation to **false**. Results are printed and saved in the logs folder.

```java
    justExplain=false;
    visualisation=false;
    exhaustive=true;
    jakulin=false;
```     
----- FC based on interaction information 

Flag variables jakulin and exhaustive must be set to **true** and flags justExplain and visualisation to **false**. Calculate interaction information between all combinations of attributes[^3] and construct features. Results are printed and saved in the logs folder.

```java
    justExplain=false;
    visualisation=false;
    exhaustive=true;
    jakulin=true;
```
----- Knowledge discovery (construct features from the whole dataset and evaluate them)

To activate knowledge discovery (KD), the flag variable justExplain must be set to **true** and visualisation **false**. New constructs of FC are evaluated by MDL scores. Results are printed and saved in kd subfolder (logs/kd).
-	_impGroups-"time-date"_ – the file that stores groups of attributes that co-occur in explanations
-	_attrImpListMDL-"time-date"_ – the file that stores MDL scores of attributes before and after FC; attribute/feature evaluation step
-	_discretizationIntervals-"time-date"_ – the file that stores discretization intervals of numerical attributes for calculating logical features

```java          
    justExplain=true;
    visualisation=false;
```    

----- Visualisation

If you want to visualise explanations of instances from visFrom to visTo flag variable visualisation must be set to **true**. Besides that, attribute importance is also visualised. The default prediction algorithm is Random Forest and for explanations IME method is used. For each explained instance only topHigh (default: 6) attributes with the highest absolute contributions are shown. All images are saved in visualisation folder. For attribute importance[^4], which is based on instance explanations, we draw (max.) 20 the most important attributes.
         

```java
    justExplain=false;
    visualisation=true;     //visualisation of explanations using IME method
    visFrom=1, visTo=10;    //visualise instances from visFrom to visTo
    drawLimit=20;           //we draw (max.) 20 the most important attributes (attribute importance visualisation)
    topHigh=10;             //visualise features with highest contributions (instance explanation visualisation)
    RESOLUTION=100;         //density for model visualisation
    N_SAMPLES=100;          //if we use equalSampling ... number of samples  
    pdfPng=true;            //besides eps, print also pdf and png format
```
## Authors
EFC was created by Boštjan Vouk, Marko Robnik-Šikonja and Matej Guid.

[^1]: Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in neural information processing systems 30 (NIPS 2017) (pp. 4765–4774). Curran Associates, Inc. https://bit.ly/3zhk5Is
[^2]: Štrumbelj, E., & Kononenko, I. (2010). An efficient explanation of individual classifications using game theory. Journal of Machine Learning Research, 11, 1–18.
[^3]: Jakulin, A. (2005). Machine learning based on attribute interactions [Doctoral dissertation, University of Ljubljana]. ePrints.FRI. https://bit.ly/3eiJ18x
[^4]: Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and individual predictions with feature contributions. Knowledge and Information Systems, 41(3), 647–665. https://doi.org/f6pnsr
