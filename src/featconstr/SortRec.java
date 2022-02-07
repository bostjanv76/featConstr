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
        return new Double(key).compareTo(rec.key);
    }


}
