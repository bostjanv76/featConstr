package featconstr;

/**
 *
 * @author bostjan
 */
public class ParamSearchEval{
    private double acc;
    private double mae;
    private double rmse;
    private int feat[]=new int[6];                  //0-logical, 1-threshold, 2-furia, 3-cartesian, 4-relational, 5-numerical
    private int tree[]=new int[3];                  //0-tree size, 1-number of leaves, 3-sum of constructs
    private int complexityFuria[]=new int[2];       //0-number of rules, 1-sum of constructs
    private int numOfLogFeatInTree[]=new int[2];    //0-number of logical feat, 1-sum of constructs
    private int numOfCartesian[]=new int[2];        //0-number of cartesian features in tree, 1-sum of constructs (cartesian features) in tree
    private int numOfRelational[]=new int[2];       //0-number of relational features in tree, 1-sum of constructs (relational feat) in tree
    private int numOfNumerical[]=new int[2];        //0-number of numerical features in tree, 1-sum of constructs (numerical feat) in tree
    private int furiaThrInTree[]=new int[4];        //0-number of Furia feat, 1-sum of constructs in Furia feat, 2-number of Thr feat, 3-sum of construct in Thr
    private long time[]=new long[2];                //0-feature construction time, 1-learning time

    public ParamSearchEval(){   
    
    }
    public ParamSearchEval(double acc, int feat[], int tree[], int complexityFuria[], int furiaThrInTree[]){    //for classification problems   
        this.acc=acc;
        this.feat=feat;
        this.tree=tree;
        this.complexityFuria=complexityFuria;
        this.furiaThrInTree=furiaThrInTree;
    }
    public ParamSearchEval(double mae, double rmse){    //for regression problems   
        this.mae=mae;
        this.rmse=rmse;
    }
    //setter methods
    public void setAcc(double acc){
        this.acc=acc;
    }
    public void setMae(double mae){
        this.mae=mae;
    }
    public void setRmse(double rmse){
        this.rmse=rmse;
    }
    public void setFeat(int feat[]){
        this.feat=feat;
    }
    public void setTree(int tree[]){
        this.tree=tree;
    }
    public void setComplexityFuria(int complexityFuria[]){
        this.complexityFuria=complexityFuria;
    }
    public void setFuriaThrComplx(int furiaThrInTree[]){
        this.furiaThrInTree=furiaThrInTree;
    }
    public void setTime(long time[]){
        this.time=time;
    }
    public void setNumLogFeatInTree(int numOfLogFeatInTree[]){
        this.numOfLogFeatInTree=numOfLogFeatInTree;
    }
    public void setCartFeatInTree(int numOfCartesian[]){
        this.numOfCartesian=numOfCartesian;
    }
    public void setRelFeatInTree(int numOfRelational[]){
        this.numOfRelational=numOfRelational;
    }
    public void setNumFeatInTree(int numOfNumerical[]){
        this.numOfNumerical=numOfNumerical;
    }
    //getter methods
    public double getAcc(){
        return acc;
    }
    public double getMae(){
        return mae;
    }
    public double getRmse(){
        return rmse;
    }
    public int[] getFeat(){
        return feat;
    }
    public int[] getTree(){
        return tree;
    }
    public int[] getComplexityFuria(){
        return complexityFuria;
    }
    public int[] getFuriaThrComplx(){
        return furiaThrInTree;
    }
    public long[] getTime(){
        return time;
    }
    public int[] getNumLogFeatInTree(){
        return numOfLogFeatInTree;
    }
    public int[] getCartFeatInTree(){
        return numOfCartesian;
    } 
    public int[] getRelFeatInTree(){
        return numOfRelational;
    }
    public int[] getNumFeatInTree(){
        return numOfNumerical;
    } 
}
