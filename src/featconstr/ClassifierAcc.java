package featconstr;

import weka.classifiers.Classifier;

/**
 *
 * @author bostjan
 */
public class ClassifierAcc {
    Classifier c;
    double acc;
   
    public ClassifierAcc(Classifier c, double acc) {
       this.c = c;
       this.acc = acc;
    }  
}
