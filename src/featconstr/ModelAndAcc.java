package featconstr;

import weka.classifiers.Classifier;

/**
 *
 * @author bostjan
 */
public class ModelAndAcc {

private Classifier model;
    private double acc; //accuracy
    private double mae; //mean absolute error
    private double rmse; //root mean squared error
    public ModelAndAcc(){   
    }
    public ModelAndAcc(Classifier model, double acc){   
        this.model=model;
        this.acc=acc;
    }
    public ModelAndAcc(Classifier model, double mae, double rmse){   
        this.model=model;
        this.mae=mae;
        this.rmse=rmse;
    }
    public void setClassifier(Classifier model){
        this.model=model;
    }
    public void setAcc(double acc){
        this.acc=acc;
    }
    public void setMae(double mae){
        this.mae=mae;
    }
    public void setRmse(double rmse){
        this.rmse=rmse;
    }
    public Classifier getClassifier(){
        return model;
    }
    public double getAcc(){
        return acc;
    }
    public double getMae(){
        return mae;
    }
    public double getRmse(){
        return rmse;
    }

}
