/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package featconstr;

/**
 *
 * @author bostjan
 */
public class Parameters {
   private int numOfAttr;
   private double acc;
   private String evalMeth;
   public Parameters( double acc, String evalMeth, int numOfAttr) {
       this.acc = acc;
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
}
