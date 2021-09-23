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
public class SortRec implements Comparable<SortRec>{
   private int value;
   private double key;

   public SortRec(){
    this.value=0;
    this.key=0.0;
   }
   
   public SortRec(int value, double key){
       this.value = value;
       this.key = key;
   }
   public SortRec(int value){
       this.value = value;
   }
   
   public SortRec(double key){
       this.key = key;
   }
   
   public double getKey(){
       return this.key;
   }
   
      
   public void setKey(double key){
       this.key=key;
   }

   public void setValue(int value){
       this.value=value;
   }
   
   public int getValue(){
       return this.value;
   }
   
@Override
    public int compareTo(SortRec rec) {
        //double compareKey=((SortRec)o).getKey();
        /* For Ascending order*/
        //return this.key-compareKey;
        
        return new Double(key).compareTo(rec.key);

        /* For Descending order do like this */
        //return compareKey-this.key;
        //return compareage-this.studentage;
    }


}
