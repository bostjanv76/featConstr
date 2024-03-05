package featconstr;

/**
 *
 * @author bostjan
 */
public class Parameters{
   private int numOfAttr;
   private double acc;
   private double mae;
   private double rmse;
   private String evalMeth;
   
   public Parameters( double acc, String evalMeth, int numOfAttr) {
       this.acc = acc;
       this.evalMeth = evalMeth;
       this.numOfAttr=numOfAttr;
   }
   
    public Parameters(double mae, double rmse, String evalMeth, int numOfAttr) {
       this.mae = mae;
       this.rmse = rmse;
       this.evalMeth = evalMeth;
       this.numOfAttr=numOfAttr;
   }
   
   public double getAcc(){
       return acc;
   }
   
   public int getNumOfAttr(){
       return numOfAttr;
   }
   
   public String getEvalMeth(){
       return evalMeth;
   }  
    
   public double getMae(){
       return mae;
   }
    
   public double getRmse(){
       return rmse;
   }
}
