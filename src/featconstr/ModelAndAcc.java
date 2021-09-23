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
public class ModelAndAcc {

private Classifier model;
private double acc; //accuracy
    public ModelAndAcc(){   
    }
    public ModelAndAcc(Classifier model, double acc){   
        this.model=model;
        this.acc=acc;
    }
    public void setClassifier(Classifier model){
        this.model=model;
    }
    public void setAcc(double acc){
        this.acc=acc;
    }
    public Classifier getClassifier(){
        return model;
    }
    public double getAcc(){
        return acc;
    }

}
