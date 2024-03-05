package featconstr;

import java.util.concurrent.TimeUnit;

/**
 *
 * @author bostjan
 */
public class Timer{
  long startTime, stopTime;

  boolean running;

  public Timer(){
    //startTime = System.currentTimeMillis();
    startTime = System.nanoTime();
    running = true;
  }

  public void start(){
    //startTime = System.currentTimeMillis();
    startTime = System.nanoTime();
    running = true;
  }

  public void stop(){
    //stopTime = System.currentTimeMillis();
    stopTime = System.nanoTime();
    running = false;
  }

  public long diff(){
    if (running) 
      //return System.currentTimeMillis()-startTime;
      return System.nanoTime()-startTime;
    else 
    return TimeUnit.NANOSECONDS.toMillis(stopTime-startTime);  
    //return System.nanoTime()-startTime;
  }

  public String toString(){    //for conversion from milli seconds
    long diff = diff();
    long millis = diff%1000;
    long secs = (diff/1000)%60;
    long mins = (diff/(1000*60))%60;
    long hs = (diff/(1000*3600))%24;
    long days = diff/(1000*3600*24);

    if (days > 0) 
      return days+"d "+hs+"h "+mins+"m "+secs+"s "+millis+"ms";

    if (hs > 0)
      return hs+"h "+mins+"m "+secs+"s "+millis+"ms";

    if (mins > 0)
      return mins+"m "+secs+"s "+millis+"ms";

    if (secs > 0)
      return secs+"s "+millis+"ms";

    return millis+"ms";
  } 
    
}
