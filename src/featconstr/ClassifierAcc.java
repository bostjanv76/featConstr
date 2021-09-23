/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
