/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package featconstr;

import java.util.concurrent.TimeUnit;

/**
 *
 * @author bostjan
 */
public class Timer {
  long startTime, stopTime;

  boolean running;


  public Timer() {
    //startTime = System.currentTimeMillis();
    startTime = System.nanoTime();
    running = true;
  }


  public void start() {
    //startTime = System.currentTimeMillis();
    startTime = System.nanoTime();
    running = true;
  }


  public void stop() {
    //stopTime = System.currentTimeMillis();
    stopTime = System.nanoTime();
    running = false;
  }


  public long diff() {  //return time in milli seconds
    if (running) 
      //return System.currentTimeMillis()-startTime;
      return System.nanoTime()-startTime;
    else 
      return TimeUnit.NANOSECONDS.toMillis(stopTime-startTime);    
  }

  
  public String toString() {    //not goog for nano seconds
    long diff = diff();
    
	//long x=diff;
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
	//return x+"diferenca";
  } 
    
}
