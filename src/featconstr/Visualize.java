/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package featconstr;

import org.sourceforge.jlibeps.epsgraphics.*;
import java.awt.*;
import java.io.*;
import java.text.DecimalFormat;
import weka.core.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;


public class Visualize 
{
	final static float DASH[] = {10.0f};
	final static BasicStroke DASHED = new BasicStroke(1.0f,BasicStroke.CAP_BUTT,BasicStroke.JOIN_MITER,10.0f, DASH, 0.0f);
	final static BasicStroke ROUNDED = new BasicStroke(1.0f,BasicStroke.CAP_ROUND,BasicStroke.JOIN_MITER,4.0f);
	final static BasicStroke THICK = new BasicStroke(1.0f);
	final static BasicStroke NORMAL = new BasicStroke(1.0f);
	final static BasicStroke THIN = new BasicStroke(0.5f);
	final static BasicStroke BOLD = new BasicStroke(2.0f);
	static int VIS_SIZE = 500;	
	
	public Visualize()
	{
		
		
	}

        
public static String roundDecimal2(double d) {
DecimalFormat twoDForm = new DecimalFormat("#.##");
    if (twoDForm.format(d) == "-0")
        return "0";
    else
        return twoDForm.format(d);
}

public static String roundDecimal3(double d) {
DecimalFormat twoDForm = new DecimalFormat("#.###");
    if (twoDForm.format(d) == "-0")
        return "0";
    else
        return twoDForm.format(d);
}
	
private static int getY(double y){
    return (int)(y);
}

private static int getX(double y){
    return (int)((VIS_SIZE / 2) + y);
}	
	
public static void modelVisualizationToFile(String file, String modelName, String datasetName, Instances data, ArrayList[] dotsA, ArrayList dotsB[], boolean classification, int classValueToExplain, int resolution){
    int VIS_SIZE2 = 400;
    int xBB = 595;        //width of bounding box (A4)
    int yBB = 842;        //height of bounding box (A4)
    String fontName = Font.SANS_SERIF;
    Font myFont14 = new Font(fontName, Font.BOLD, 14);
    Font myFont10 = new Font(fontName, Font.BOLD, 10);

    // start drawing the picture
    try{
        int coordX = 45; // left/right margin
        int coordXlength = VIS_SIZE2 - 2 * coordX;
        int coordY = 150;
        int coordYlength = 100;           
        int sign_size = 2;  //orig. 4 ... velikost pike pri risanju prispevkov in stdev prispevka
        int sign_step = (int)(sign_size/2.0);  		
            
        //int xBB = VIS_SIZE2+20;        //width of bounding box (A4)
        int yImg = (data.numAttributes()-1)*(coordYlength+20)+34;      //height of bounding box (A4) ...	data.numAttributes()-1	... we draw just attributes withouth class 10 Dataset 10 Model 14 space between Model and first attr  
            //System.out.println("height of image "+yImg);
        FileOutputStream finalImage = new FileOutputStream(file);
        //EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0, 0, VIS_SIZE2+20, data.numAttributes()*(coordYlength+20));
        //EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0, 0, xBB, data.numAttributes()*(coordYlength+20));
        EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0, 0, xBB, yBB);
    		
        //center picture to bounding box
        //int wBox=530;       //width of the box for drawing
        int xT=xBB/2-(VIS_SIZE2+20)/2;
        int yT=yBB/2-yImg/2;
        //g.translate(xT,yT); //because of later transformation to pdf and png  - to be in the center of the page
            
        g.setFont(myFont14);
        g.setColor(Color.BLACK);               
        // *************************
        // outer graph visualization     
        
        // dataset & model print
        int width1,width2;
        g.drawString("Dataset: "+datasetName,10,10);

        g.drawString("Model: "+modelName,10,25);
        width1 =g.getFontMetrics().stringWidth(("Model: "+modelName));        
        
        if(!classification){
            width1=g.getFontMetrics().stringWidth(("Dataset: "+datasetName));
            g.drawString("Resolution: " + resolution,10+width1+5,25);  
        }
        else{
            if(data.attribute(1).isNumeric()){
                width2=g.getFontMetrics().stringWidth(("Explaining class: "+(new Instances(data,1,1)).instance(0).classAttribute().value(classValueToExplain)));
                g.drawString("Resolution: " + resolution,10+width1+width2+5,25);
            }
            g.drawString("Explaining class: " + (new Instances(data,1,1)).instance(0).classAttribute().value(classValueToExplain),10+width1+5,25);
        }        
                //instance.instance(0).classAttribute().value(classValueToExplain));

        //g.drawString("Explanation time (sec): "+time,220,20);
			
        double max_val = Double.MIN_VALUE;
        double min_val = Double.MAX_VALUE;
             
        for(int i = 0; i < dotsA.length; i++){
            ArrayList temp = dotsA[i];
            ArrayList temp2 = dotsB[i];
            for (int j = 0; j < temp.size() / 2; j++){
                double d = (Double)(temp.get(j*2+1));
                //System.out.println(d);
		if(d > max_val) 
                    max_val = d;
		if(d < min_val) 
                    min_val = d;
                d = (Double)(temp2.get(j*2+1));
		//System.out.println(d);
		if(d > max_val) 
                    max_val = d;
		if(d < min_val) 
                    min_val = d;
            }
        }
		
               
        for(int i = 0; i < data.numAttributes() - 1; i++){
            double maxX = Double.MIN_VALUE;
            double minX = Double.MAX_VALUE;
            g.setFont(myFont10);
            g.setColor(Color.GRAY);
            g.drawString(data.attribute(i).name(),coordX + 2,(i-1)*(coordYlength+20) + coordY+20 - 5);
            ArrayList temp = dotsA[i];
            ArrayList temp2 = dotsB[i];
            if (data.attribute(i).isNominal()){
                maxX = data.attribute(i).numValues() - 1;
                minX = 0;
            }
            else
                for (int j = 0; j < temp.size() / 2; j++){
                    double d = (Double)(temp.get(j*2));
                    if(d > maxX) 
                        maxX = d;
                    if(d < minX) 
                        minX = d;
            }
    			
            // attribute text
            g.setColor(Color.BLACK);
            g.setFont(myFont10);
				
            double chunkSize = coordYlength/(max_val - min_val);
            	
            // base lines
	    g.setStroke(NORMAL);
	    g.drawLine(coordX,coordY+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20));   //spodnja črta
	    g.drawLine(coordX,coordY+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));
	    g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20));
	    g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));    //zgornja črta
	 
	    // zero axis
	    double axisOffSet = 0;
	    if(min_val < 0) 
                axisOffSet = min_val*chunkSize;
            g.setStroke(DASHED);
	    g.drawLine(coordX,coordY+i*(coordYlength+20)+(int)axisOffSet,coordX+coordXlength,coordY+i*(coordYlength+20)+(int)axisOffSet);
	            
	            
            // zero axis start and end numbers
	    if (data.attribute(i).isNominal()){
                g.setFont(myFont10);
	        for (int j = 0; j < data.attribute(i).numValues();j++){
                    g.drawString(data.attribute(i).value(j),(int)(((j - minX) / (maxX - minX)) * +coordXlength) + coordX, coordY+i*(coordYlength+20)+(int)axisOffSet +10);
	    	}
            }
	    else{
                g.setFont(myFont10);
                g.drawString(roundDecimal2(minX),coordX -10,coordY+i*(coordYlength+20)+(int)axisOffSet +10);	      
                g.drawString(roundDecimal2(maxX),coordX+coordXlength  - 20,coordY+i*(coordYlength+20)+(int)axisOffSet +10);
   	}
	           
	        
	// horizontal and vertical limiters
	g.setFont(myFont10);
	int ylabels = 4;
	for(int j = 0; j < ylabels+1; j++){
            g.drawLine(coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet-5, coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet+5);
	            	
	    // vertical limiters
	    g.drawLine(coordX + coordXlength-5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize), coordX + coordXlength+5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
	    g.drawString(roundDecimal2(j*((max_val - min_val)/ylabels)+min_val), coordX + coordXlength+15,  coordY + 3 + i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
	}
	            
	            
	for (int j = 0; j < temp.size() / 2; j++){
            double x = (Double)(temp.get(j*2));
	    double y = (Double)(temp.get(j*2+1));    		
	        		
            if (!data.attribute(i).isNominal()){
                g.setColor(Color.BLACK);
                g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
		g.setColor(Color.GRAY);
                y = (Double)(temp2.get(j*2+1));
                g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
	    }
	    else{
                sign_size+=2;
                g.setStroke(BOLD);
                g.setColor(Color.BLACK);
		g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
		g.setColor(Color.GRAY);
		y = (Double)(temp2.get(j*2+1));
		g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
                sign_size-=2;
	    }

	}
	     				
        // horizontal line
	g.setColor(Color.GRAY);
	//g.drawLine(0, 260+attribute*(coordYlength+20), VIS_SIZE2+20, 260+attribute*(coordYlength+20));
        }
           

        g.flush();
        g.close();
        finalImage.close();
    }
    catch(Exception e){
        System.err.println("ERROR: "+e);
    }			
}
public static void modelVisualToFileAttrImptLine(String file, String modelName, String datasetName, Instances data, ArrayList[] dotsA, ArrayList dotsB[], boolean classification, int resolution, int classValueToExplain, String format){
    int VIS_SIZE2 = 400;
    int xBB = 595;        //width of bounding box (A4)
    int yBB = 842;        //height of bounding box (A4)
    String fontName = Font.SANS_SERIF;
    Font myFont14 = new Font(fontName, Font.BOLD, 14);
    Font myFont10 = new Font(fontName, Font.BOLD, 10);
    Font myFont8 = new Font(fontName, Font.BOLD, 8);

    // start drawing the picture
    try{
        int coordX = 45; // left/right margin
        int coordXlength = VIS_SIZE2 - 2 * coordX;
        int coordY = 150;
        int coordYlength = 100;           
        int sign_size = 2;  //orig. 4 ... velikost pike pri risanju prispevkov in stdev prispevka
        int sign_step = (int)(sign_size/2.0);  		
            
        //int xBB = VIS_SIZE2+20;        //width of bounding box (A4)
        int yImg = (data.numAttributes()-1)*(coordYlength+20)+34;      //height of bounding box (A4) ...	data.numAttributes()-1	... we draw just attributes withouth class 10 Dataset 10 Model 14 space between Model and first attr  
            //System.out.println("height of image "+yImg);
        FileOutputStream finalImage = new FileOutputStream(file);
        EpsGraphics2D g;
        if(format.equals("A4"))
            g = new EpsGraphics2D("Title", finalImage, 0, 0, xBB, yBB);
        else
            g = new EpsGraphics2D("Title", finalImage, 0, 0, VIS_SIZE2+20, data.numAttributes()*(coordYlength+20));
        //EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0, 0, xBB, data.numAttributes()*(coordYlength+20));
        
    		
        //center picture to bounding box
        //int wBox=530;       //width of the box for drawing
        int xT=xBB/2-(VIS_SIZE2+20)/2;
        int yT=yBB/2-yImg/2;
        g.translate(xT,yT); //because of later transformation to pdf and png  - to be in the center of the page
            
        g.setFont(myFont14);
        g.setColor(Color.BLACK);               
        // *************************
        // outer graph visualization     
        
        // dataset & model print
        int width1,width2;
        g.drawString("Dataset: "+datasetName,10,10);

        g.drawString("Model: "+modelName,10,25);
        width1 =g.getFontMetrics().stringWidth(("Model: "+modelName));        
        
        if(!classification){
            width1=g.getFontMetrics().stringWidth(("Dataset: "+datasetName));
            g.drawString("Resolution: " + resolution,10+width1+5,25);  
        }
        else{
            if(data.attribute(1).isNumeric()){
                width2=g.getFontMetrics().stringWidth(("Explaining class: "+(new Instances(data,1,1)).instance(0).classAttribute().value(classValueToExplain)));
                g.drawString("Resolution: " + resolution,10+width1+width2+5,25);
            }
            g.drawString("Explaining class: " + (new Instances(data,1,1)).instance(0).classAttribute().value(classValueToExplain),10+width1+5,25);
        }        
                //instance.instance(0).classAttribute().value(classValueToExplain));

        //g.drawString("Explanation time (sec): "+time,220,20);
			
        double max_val = -Double.MAX_VALUE;//Double.MIN_VALUE;
        double min_val = Double.MAX_VALUE;
             
        for(int i = 0; i < dotsA.length; i++){
            ArrayList temp = dotsA[i];
            ArrayList temp2 = dotsB[i]; //v dotsB so shranjene vrednsosti za informativnost atributa, načeloma nebi potrebovali tabelo ArrayLista, predelana je predhodna različica, ki vsebuje točke standardne deviacije za vsak psi
            for (int j = 0; j < temp.size() / 2; j++){
                double d = (Double)(temp.get(j*2+1));
                //System.out.println(d);
		if(d > max_val) 
                    max_val = d;
		if(d < min_val) 
                    min_val = d;
                d = (Double)(temp2.get(0));
		//System.out.println(d);
		if(d > max_val) 
                    max_val = d;
		if(d < min_val) 
                    min_val = d;
            }
        }
	max_val+=max_val*5/100.0;   //povečamo maksimalno vrednost osi X zaradi boljše vidljivosti	
        min_val+=min_val*5/100.0;   //"povečamo" min vrednost X osi v negativni smeri - povečamo razpon
        
        for(int i = 0; i < data.numAttributes() -1; i++){            
            double maxX = Double.MIN_VALUE;
            double minX = Double.MAX_VALUE;
            g.setFont(myFont10);
            g.setColor(Color.GRAY);
            g.drawString(data.attribute(i).name(),coordX + 2,(i-1)*(coordYlength+20) + coordY+20 - 5);
            ArrayList temp = dotsA[i];
            ArrayList temp2 = dotsB[i];
            if (data.attribute(i).isNominal()){
                maxX = data.attribute(i).numValues() - 1;
                minX = 0;
            }
            else
                for (int j = 0; j < temp.size() / 2; j++){
                    double d = (Double)(temp.get(j*2));
                    if(d > maxX) 
                        maxX = d;
                    if(d < minX) 
                        minX = d;
            }
    			
            // attribute text
            g.setColor(Color.GRAY);
            g.setFont(myFont10);
				
            double chunkSize = coordYlength/(max_val - min_val);
            	
            // base lines
	    g.setStroke(NORMAL);
	    g.drawLine(coordX,coordY+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20));   //spodnja črta
	    g.drawLine(coordX,coordY+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));    //leva črta pravokotnika
	    g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20)); //desna črta pravokotnika
	    g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));    //zgornja črta
	 
	    // zero axis
	    double axisOffSet = 0;
	    if(min_val < 0) 
                axisOffSet = min_val*chunkSize;
            g.setStroke(DASHED);
	    g.drawLine(coordX,coordY+i*(coordYlength+20)+(int)axisOffSet,coordX+coordXlength,coordY+i*(coordYlength+20)+(int)axisOffSet);
	            
	            
            // zero axis start and end numbers
	    if (data.attribute(i).isNominal()){
                g.setFont(myFont8);
	        for (int j = 0; j < data.attribute(i).numValues();j++){
                    g.drawString(data.attribute(i).value(j),(int)(((j - minX) / (maxX - minX)) * +coordXlength) + coordX, coordY+i*(coordYlength+20)+(int)axisOffSet +10);
	    	}
            }
	    else{
                g.setFont(myFont8);
                g.drawString(roundDecimal2(minX),coordX -10,coordY+i*(coordYlength+20)+(int)axisOffSet +10);	      
                g.drawString(roundDecimal2(maxX),coordX+coordXlength  - 20,coordY+i*(coordYlength+20)+(int)axisOffSet +10);
   	}
	           
	        
	// horizontal and vertical limiters
	g.setFont(myFont10);
	int ylabels = 4;
	for(int j = 0; j < ylabels+1; j++){
            //izris oznak na X osi
            g.drawLine(coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet-5, coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet+5);
	            	
	    //izris oznak in vrednosti na Y osi
	    g.drawLine(coordX + coordXlength-5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize), coordX + coordXlength+5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
	    g.drawString(roundDecimal2(j*((max_val - min_val)/ylabels)+min_val), coordX + coordXlength+15,  coordY + 3 + i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
	}
	            
	            
	for (int j = 0; j < temp.size() / 2; j++){
            double x = (Double)(temp.get(j*2));
	    double y = (Double)(temp.get(j*2+1));    		
	        		
            if (!data.attribute(i).isNominal()){
                g.setColor(Color.BLACK);
                g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
                //g.setColor(Color.RED);
                //g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX + (int)(sign_size*0.40), 
                 //       (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY + (int)(sign_size*0.40), 
                 //       sign_size/2, sign_size/2);
		//g.setColor(Color.GRAY);
                //y = (Double)(temp2.get(0));
                //g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
                
	    }
	    else{
                sign_size+=2;
                g.setStroke(BOLD);
                g.setColor(Color.BLACK);
		g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
		//g.setColor(Color.GRAY);
		//y = (Double)(temp2.get(0));
		//g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
                sign_size-=2;
	    }

	}
        
        
        //g.drawLine(coordX,coordY+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20)); 
            //draw attribute importance
            double x1=(Double)(temp.get(0));
            double x2=(Double)(temp.get(temp.size()-2));
            double y = (Double)(temp2.get(0));
            g.setColor(Color.getHSBColor(121, 83, 54));
            g.setStroke(NORMAL);
            g.setFont(myFont8);
            width1=g.getFontMetrics().stringWidth("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)));
            if (data.attribute(i).isNominal()){ 
                //g.drawLine((int)(((x1 - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                //            (int)((((data.attribute(i).numValues()-1) - minX) / (maxX - minX)) * +coordXlength) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                g.drawLine(coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                            (int)((((data.attribute(i).numValues()-1) - minX) / (maxX - minX)) * +coordXlength) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                g.drawString("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)), 
                        coordX+(int)(((double)coordXlength/2)-width1/2),//(int)(((((data.attribute(i).numValues()-1) - minX) / (maxX - minX)) * +coordXlength) + coordX)/2,  
                        (int)((((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY)+15);
            }
            else{
                
                //g.drawLine((int)(((x1 - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                //            (int)(((x2 - minX) / (maxX - minX)) * +coordXlength) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                g.drawLine(coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                            (int)(((x2 - minX) / (maxX - minX)) * +coordXlength) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                g.drawString("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)), 
                        coordX+(int)(((double)coordXlength/2)-width1/2),//(int)(((((x2 - minX) / (maxX - minX)) * +coordXlength) + coordX)/2),  
                        (int)((((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY)+15);
            }
        // horizontal line
	g.setColor(Color.GRAY);
	//g.drawLine(0, 260+attribute*(coordYlength+20), VIS_SIZE2+20, 260+attribute*(coordYlength+20));
        
        
        //System.out.println("Attr. importance A"+(i+1)+" "+roundDecimal3((double)dotsB[i].get(0)));
        
        }
           

        g.flush();
        g.close();
        finalImage.close();
    }
    catch(Exception e){
        System.err.println("ERROR: "+e);
    }			
}
public static void attrImportanceVisualization(String file, String modelName, String datasetName, Instances data, ArrayList dotsB[], boolean classification, int resolution, String format){
    int xBB = 595;        //width of bounding box (A4)
    int yBB = 842;      //height of bounding box (A4)
    int wBox=530;       //width of the box for drawing ... VIS_SIZE is currently 500
    int fontSize2 = 14;
    int fontSize = 14;
    int leadinY = 80;//orig 130; //odmik med zgornjim robom in napisom Feature Contribution in Value
    int minY = leadinY;
    int leadoutY = 70; //odmik med koncem napisov (številke na osi) in spodnjim robom
    double perFeature = 30;
    double ratio = 0.6;
    int maxX = +150;
    int minX = 0;//orig. -150;
    int textLeft = -10;  //orig. 10 ... -10 because we then add +20
    String fontName = Font.SANS_SERIF;
    Font myFont = new Font(fontName, Font.BOLD, fontSize);
    double threshold=0.1;
        
    //added by BV ... calculate threshold
    /*double contrCp[]=contributions.clone();
        for (int i = 0; i < contrCp.length; i++){
            if (contrCp[i]<0)
                contrCp[i]=Math.abs(contrCp[i]);
        }
        Arrays.sort(contrCp);
        if(contrCp.length>topHigh)
            threshold=contrCp[contrCp.length-topHigh];*/
        /////////////////////////////////////////////
    
        int relevantFeatures = 0;
        double maxContrib = 0;
        for (int i = 0; i < dotsB.length; i++){
            //System.out.println("contribution "+ i+" "+contributions[i]);
        	if (Math.abs((double)dotsB[i].get(0)) >= threshold) 
                    relevantFeatures++;
        	if (Math.abs((double)dotsB[i].get(0)) >= maxContrib) 
                    maxContrib = Math.abs((double)dotsB[i].get(0));
        }

        
        int TOTAL_Y = (int)(leadinY + perFeature * relevantFeatures + leadoutY);
        double MAX_Y = leadinY + perFeature * relevantFeatures; //vertikalna črta med pozitivnim in negativnim delom

        
    try{
        FileOutputStream finalImage = new FileOutputStream(file);
        //EpsGraphics2D(String title, OutputStream outputStream, int minX, int minY, int maxX, int maxY)
        //EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0, 0, 500, TOTAL_Y); //orig.

        EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0,0, xBB, yBB); //A4 beacause if we later convert eps to pdf and png (parameters are set to center image on A4)
        
        //center picture to bounding box
        int xT=xBB/2-wBox/2;
        int yT=yBB/2-TOTAL_Y/2;
        g.translate(xT,yT); //because of later transformation to pdf and png  - to be in the center of the page
        
        g.setFont(myFont);
        g.setStroke(THICK);
        g.setColor(Color.BLACK);
        g.drawLine(1,1,1,TOTAL_Y);
        g.drawLine(wBox,1,wBox,TOTAL_Y);
        g.drawLine(1,TOTAL_Y,wBox,TOTAL_Y);
        g.drawLine(1,1,wBox,1);
        g.setColor(Color.BLACK);
        g.drawRect(380,3,148,22);
          
        g.drawString("Data: " + datasetName,10,20);
        g.drawString("Model: " + modelName,10,40);
        
        //g.drawString("Model: " + modelName,215,20); //orig. 
        g.drawString("Attribute importance",385,20);
        //g.drawString("Explanation time per instance : " + time + " seconds",10,40);
            


        g.drawString("Feature",15,getY(minY-10));
        g.drawString("Value",getX(maxX + 35) + 30,getY(minY-10));
        g.drawString("Importance",getX(0-42),getY(minY-10));
              
        g.setStroke(ROUNDED);
        int counter = 0;
        for (int i = 0; i < dotsB.length; i++){
            if (Math.abs((double)dotsB[i].get(0)) >= threshold){
                // text for feature
                int textSize = fontSize2;
	        Font tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
	        g.setFont(tempFont);
	
	        String attVal = dotsB[i].get(0).toString();
	        double yText = perFeature*(counter) + minY;
	        g.setColor(Color.BLACK);
	        g.drawString(data.attribute(i).name() + " ", textLeft+20,getY(yText+fontSize+3));
	        g.drawString(" " + formatValue(attVal,13), getX(maxX + 5) + 30,getY(yText+fontSize+3));

	        // bar for feature
	        double y = perFeature*(counter) + minY;
	        double y2 = perFeature*(counter+1) + minY;
	        g.setStroke(DASHED);
                g.drawLine(VIS_SIZE/4,getY(y2),getX(maxX),getY(y2)); //orig. g.drawLine(getX(minX),getY(y2),getX(maxX),getY(y2));
	        g.setStroke(NORMAL);
	        g.setColor(Color.GRAY);
	
                double barH = (int)(perFeature * ratio);
	        double barTop = y + (perFeature- barH) / 2;
	        double x1 = Math.min(((double)dotsB[i].get(0)/maxContrib) * (getX(maxX)-VIS_SIZE/4),0);
	        double x2 = Math.abs(((double)dotsB[i].get(0)/maxContrib) * (getX(maxX)-VIS_SIZE/4));
System.out.println((double)dotsB[i].get(0)+" "+maxContrib+" "+((double)dotsB[i].get(0)/maxContrib) +" "+ (getX(maxX)-VIS_SIZE/4)+" "+(int)Math.ceil(x2));
	  
	        if (Math.abs((double)dotsB[i].get(0)) >= 0.01){    //if threshold is applied then this is irrelevant
                    //g.fillRect(getX(x1),getY(barTop),(int)x2,(int)barH);  //orig.    
	            //g.fillRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH); //fillRect(int x, int y, int width, int height)
                    g.fillRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
                    //System.out.println(VIS_SIZE/4+" "+getY(barTop)+" "+(int)Math.ceil(x2) +" "+ (int)barH);
                    g.setColor(Color.BLACK);
	            //g.drawRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH); //orig. (int)x2
                    g.drawRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
	        }
            }
                	
            if (Math.abs((double)dotsB[i].get(0)) >= threshold) 
                counter++;
        }
        
        double y = perFeature*(0) + minY;
        g.setStroke(DASHED);
        g.drawLine(VIS_SIZE/4,getY(y),getX(maxX),getY(y)); //prva črtkana črta
        // axis & scale	
        g.setStroke(NORMAL);
        //g.drawLine(getX(0),getY(MAX_Y),getX(0),getY(minY)); //vertikalna črta za 0, v tem primeru je ne potrebujemo
        Font tempFont2 = new Font(Font.MONOSPACED, Font.BOLD, fontSize2-1);
        g.setFont(tempFont2);

        //g.drawLine(getX(minX),getY(MAX_Y + 20),getX(maxX),getY(MAX_Y + 20)); //orig. spodnja črna črta ... os kamor postavimo vrednosti
        g.drawLine(VIS_SIZE/4,getY(MAX_Y + 20),getX(maxX),getY(MAX_Y + 20)); //spodnja črna črta ... os kamor postavimo vrednosti
                           
        String[] tick = new String[3];
        tick[0] = ""+minX;  //"  0"
        tick[1] = roundDecimal2(maxContrib/2);
        tick[2] = roundDecimal2(maxContrib);
                
                
        //risanje oznak na X osi        
        //prva vertikalna črtica 0
        int width =g.getFontMetrics().stringWidth((tick[0]));
        g.drawLine(VIS_SIZE/4,getY(MAX_Y + 24),VIS_SIZE/4,getY(MAX_Y + 20)); //vertikalne črtice na osi
        g.drawString(tick[0],VIS_SIZE/4-width/2,getY(MAX_Y + 24+20) );
        //druga vertikalna črtica polovica
        width=g.getFontMetrics().stringWidth((tick[1]));
        g.drawLine(VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2,getY(MAX_Y + 24),VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2,getY(MAX_Y + 20)); //vertikalne črtice na osi
        g.drawString(tick[1],VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2-width/2,getY(MAX_Y + 24+20) );
        //tretja vertikalna črtica maksimum
        width=g.getFontMetrics().stringWidth((tick[2]));
        g.drawLine(getX(maxX),getY(MAX_Y + 24),getX(maxX),getY(MAX_Y + 20)); //vertikalne črtice na osi
        g.drawString(tick[2],getX(maxX)-width/2,getY(MAX_Y + 24+20) );  //-width/2 to center number to axis



        g.flush();
        g.close();
        finalImage.close();
    }
    
    catch (Exception e){
        System.out.println("ERROR: "+e);
    }			
}

public static void attrImportanceVisualizationSorted(String file, String modelName, String datasetName, Instances data, ArrayList dotsB[], boolean classification, int resolution,String format){
    int xBB = 595;        //width of bounding box (A4)
    int yBB = 842;      //height of bounding box (A4)
    int wBox=530;       //width of the box for drawing ... VIS_SIZE is currently 500
    int fontSize2 = 14;
    int fontSize = 14;
    int leadinY = 80;//orig 130; //odmik med zgornjim robom in napisom Feature Contribution in Value
    int minY = leadinY;
    int leadoutY = 70; //odmik med koncem napisov (številke na osi) in spodnjim robom
    double perFeature = 30;
    double ratio = 0.6;
    int maxX = +150;
    int minX = 0;//orig. -150;
    int textLeft = -10;  //orig. 10 ... -10 because we then add +20
    String fontName = Font.SANS_SERIF;
    Font myFont = new Font(fontName, Font.BOLD, fontSize);
    double threshold=0.03;
        
    //added by BV ... calculate threshold
    /*double contrCp[]=contributions.clone();
        for (int i = 0; i < contrCp.length; i++){
            if (contrCp[i]<0)
                contrCp[i]=Math.abs(contrCp[i]);
        }
        Arrays.sort(contrCp);
        if(contrCp.length>topHigh)
            threshold=contrCp[contrCp.length-topHigh];*/
        /////////////////////////////////////////////
        
        
    Map<Double, String> treemap = new TreeMap<>();    
    for(int i=0;i<dotsB.length;i++){
        treemap.put((double)dotsB[i].get(0), data.attribute(i).name());
    }
                    
//print Treemap            
/*for (Entry<Double, String> entry : treemap.entrySet()) {
    double key = entry.getKey();
    String value = entry.getValue();
    System.out.printf("%s : %s\n", key, value);
}*/

//sort (descending) map based on attr. importance
    Map<Double, String> newMap = new TreeMap<>(Collections.reverseOrder());
        newMap.putAll(treemap);
        if(newMap.size()>20){    //izrišemo samo 20 najpomembnejših atributov
            //System.out.println("Limit get"+newMap.get(newMap.keySet().toArray()[19]));   //ime atributa
            //System.out.println("Ime "+newMap.values().toArray()[19]);                    //ime atributa ... vrne enako kot zgoraj
            System.out.println("Limit za izris ime attr.: "+newMap.get(newMap.keySet().toArray()[19]));   //ime atributa
            System.out.println("Limit za izris vrednost: "+newMap.keySet().toArray()[19]);            //vrednost atributa
            threshold=Double.parseDouble(String.valueOf(newMap.keySet().toArray()[19]));
            newMap.keySet().removeAll(Arrays.asList(newMap.keySet().toArray()).subList(20, newMap.size()-1));
        }
        else
            threshold=Double.parseDouble(String.valueOf(newMap.keySet().toArray()[newMap.size()-2]));
        System.out.println("Velikost seznama-Map: "+newMap.size()); 
        System.out.println("Testni izpis limit: "+Double.parseDouble(String.valueOf(newMap.keySet().toArray()[newMap.size()-2]))); 
//print sorted map
/*for (Entry<Double, String> entry : newMap.entrySet()) {
    double key = entry.getKey();
    String value = entry.getValue();
    System.out.printf("%s : %s\n", key, value);
}*/

        int relevantFeatures = 0;
        double maxContrib = 0;
        for (int i = 0; i < dotsB.length; i++){
            //System.out.println("contribution "+ i+" "+contributions[i]);
        	if (Math.abs((double)dotsB[i].get(0)) >= threshold) 
                    relevantFeatures++;
        	if (Math.abs((double)dotsB[i].get(0)) >= maxContrib) 
                    maxContrib = Math.abs((double)dotsB[i].get(0));
        }

        
        int TOTAL_Y = (int)(leadinY + perFeature * relevantFeatures + leadoutY);
        double MAX_Y = leadinY + perFeature * relevantFeatures; //vertikalna črta med pozitivnim in negativnim delom

        
    try{
        FileOutputStream finalImage = new FileOutputStream(file);
        //EpsGraphics2D(String title, OutputStream outputStream, int minX, int minY, int maxX, int maxY)
        //EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0, 0, 500, TOTAL_Y); //orig.

        EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0,0, xBB, yBB); //A4 beacause if we later convert eps to pdf and png (parameters are set to center image on A4)
        
        //center picture to bounding box
        int xT=xBB/2-wBox/2;
        int yT=yBB/2-TOTAL_Y/2;
        g.translate(xT,yT); //because of later transformation to pdf and png  - to be in the center of the page
        
        g.setFont(myFont);
        g.setStroke(THICK);
        g.setColor(Color.BLACK);
        g.drawLine(1,1,1,TOTAL_Y);
        g.drawLine(wBox,1,wBox,TOTAL_Y);
        g.drawLine(1,TOTAL_Y,wBox,TOTAL_Y);
        g.drawLine(1,1,wBox,1);
        g.setColor(Color.BLACK);
        g.drawRect(380,3,148,22);
          
        g.drawString("Data: " + datasetName,10,20);
        g.drawString("Model: " + modelName,10,40);
        
        //g.drawString("Model: " + modelName,215,20); //orig. 
        g.drawString("Attribute importance",385,20);
        //g.drawString("Explanation time per instance : " + time + " seconds",10,40);
            


        g.drawString("Feature",15,getY(minY-10));
        g.drawString("Value",getX(maxX + 35) + 30,getY(minY-10));
        g.drawString("Importance",getX(0-42),getY(minY-10));
              
        g.setStroke(ROUNDED);
        int counter = 0;
        //for (int i = 0; i < dotsB.length; i++){
        for (Entry<Double, String> entry : newMap.entrySet()) {
            
                double value = entry.getKey();
                String attrName = entry.getValue();
            if (value >= threshold){
                // text for feature
                int textSize = fontSize2;
	        Font tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
	        g.setFont(tempFont);
	
	        String attVal = value+"";
	        double yText = perFeature*(counter) + minY;
	        g.setColor(Color.BLACK);
	        g.drawString(attrName + " ", textLeft+20,getY(yText+fontSize+3));
	        g.drawString(" " + formatValue(attVal,13), getX(maxX + 5) + 30,getY(yText+fontSize+3));

	        // bar for feature
	        double y = perFeature*(counter) + minY;
	        double y2 = perFeature*(counter+1) + minY;
	        g.setStroke(DASHED);
                g.drawLine(VIS_SIZE/4,getY(y2),getX(maxX),getY(y2)); //orig. g.drawLine(getX(minX),getY(y2),getX(maxX),getY(y2));
	        g.setStroke(NORMAL);
	        g.setColor(Color.GRAY);
	
                double barH = (int)(perFeature * ratio);
	        double barTop = y + (perFeature- barH) / 2;
	        double x1 = Math.min((value/maxContrib) * (getX(maxX)-VIS_SIZE/4),0);
	        double x2 = Math.abs((value/maxContrib) * (getX(maxX)-VIS_SIZE/4));
//System.out.println(value+" "+maxContrib+" "+(value/maxContrib) +" "+ (getX(maxX)-VIS_SIZE/4)+" "+(int)Math.ceil(x2));
	  
	        if (value >= 0.01){    //if threshold is applied then this is irrelevant
                    //g.fillRect(getX(x1),getY(barTop),(int)x2,(int)barH);  //orig.    
	            //g.fillRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH); //fillRect(int x, int y, int width, int height)
                    g.fillRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
                    //System.out.println(VIS_SIZE/4+" "+getY(barTop)+" "+(int)Math.ceil(x2) +" "+ (int)barH);
                    g.setColor(Color.BLACK);
	            //g.drawRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH); //orig. (int)x2
                    g.drawRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
	        }
            }
                	
            if (Math.abs(value) >= threshold) 
                counter++;
        }
        
        double y = perFeature*(0) + minY;
        g.setStroke(DASHED);
        g.drawLine(VIS_SIZE/4,getY(y),getX(maxX),getY(y)); //prva črtkana črta
        // axis & scale	
        g.setStroke(NORMAL);
        //g.drawLine(getX(0),getY(MAX_Y),getX(0),getY(minY)); //vertikalna črta za 0, v tem primeru je ne potrebujemo
        Font tempFont2 = new Font(Font.MONOSPACED, Font.BOLD, fontSize2-1);
        g.setFont(tempFont2);

        //g.drawLine(getX(minX),getY(MAX_Y + 20),getX(maxX),getY(MAX_Y + 20)); //orig. spodnja črna črta ... os kamor postavimo vrednosti
        g.drawLine(VIS_SIZE/4,getY(MAX_Y + 20),getX(maxX),getY(MAX_Y + 20)); //spodnja črna črta ... os kamor postavimo vrednosti
                           
        String[] tick = new String[3];
        tick[0] = ""+minX;  //"  0"
        tick[1] = roundDecimal2(maxContrib/2);
        tick[2] = roundDecimal2(maxContrib);
                
                
        //risanje oznak na X osi        
        //prva vertikalna črtica 0
        int width =g.getFontMetrics().stringWidth((tick[0]));
        g.drawLine(VIS_SIZE/4,getY(MAX_Y + 24),VIS_SIZE/4,getY(MAX_Y + 20)); //vertikalne črtice na osi
        g.drawString(tick[0],VIS_SIZE/4-width/2,getY(MAX_Y + 24+20) );
        //druga vertikalna črtica polovica
        width=g.getFontMetrics().stringWidth((tick[1]));
        g.drawLine(VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2,getY(MAX_Y + 24),VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2,getY(MAX_Y + 20)); //vertikalne črtice na osi
        g.drawString(tick[1],VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2-width/2,getY(MAX_Y + 24+20) );
        //tretja vertikalna črtica maksimum
        width=g.getFontMetrics().stringWidth((tick[2]));
        g.drawLine(getX(maxX),getY(MAX_Y + 24),getX(maxX),getY(MAX_Y + 20)); //vertikalne črtice na osi
        g.drawString(tick[2],getX(maxX)-width/2,getY(MAX_Y + 24+20) );  //-width/2 to center number to axis



        g.flush();
        g.close();
        finalImage.close();
    }
    
    catch (Exception e){
        System.out.println("ERROR: "+e);
    }			
}

public static void instanceVisualizationToFile(String file, String modelName, String datasetName, Instances instance, int id, double[] contributions, int topHigh, double prediction, int classValueToExplain, boolean isClassification){				
    int xBB = 595;        //width of bounding box (A4)
    int yBB = 842;      //height of bounding box (A4)
    int wBox=530;       //width of the box for drawing
    int fontSize2 = 14;
    int fontSize = 14;
    int leadinY = 170;//orig 130; //odmik med zgornjim robom in napisom Feature Contribution in Value
    int minY = leadinY;
    int leadoutY = 70; //odmik med koncem napisov (številke na osi) in spodnjim robom
    double perFeature = 30;
    double ratio = 0.6;
    int maxX = +150;
    int minX = -150;
    int textLeft = -10;  //orig. 10 ... -10 because we then add +20
    String fontName = Font.SANS_SERIF;
    Font myFont = new Font(fontName, Font.BOLD, fontSize);
    double threshold=-1;
        
    //added by BV ... calculate threshold
    double contrCp[]=contributions.clone();
        for (int i = 0; i < contrCp.length; i++){
            if (contrCp[i]<0)
                contrCp[i]=Math.abs(contrCp[i]);
        }
        Arrays.sort(contrCp);
        if(contrCp.length>topHigh)
            threshold=contrCp[contrCp.length-topHigh];
        /////////////////////////////////////////////
        
        int relevantFeatures = 0;
        double maxContrib = 0;
        for (int i = 0; i < contributions.length; i++){
            //System.out.println("contribution "+ i+" "+contributions[i]);
        	if (Math.abs(contributions[i]) >= threshold) 
                    relevantFeatures++;
        	if (Math.abs(contributions[i]) >= maxContrib) 
                    maxContrib = Math.abs(contributions[i]);
        }

        
        int TOTAL_Y = (int)(leadinY + perFeature * relevantFeatures + leadoutY);
        double MAX_Y = leadinY + perFeature * relevantFeatures; //vertikalna črta med pozitivnim in negativnim delom

        
    try{
        FileOutputStream finalImage = new FileOutputStream(file);
        //EpsGraphics2D(String title, OutputStream outputStream, int minX, int minY, int maxX, int maxY)
        //EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0, 0, 500, TOTAL_Y); //orig.

        EpsGraphics2D g = new EpsGraphics2D("Title", finalImage, 0,0, xBB, yBB); //A4 beacause if we later convert eps to pdf and png (parameters are set to center image on A4)
        
        //center picture to bounding box
        int xT=xBB/2-wBox/2;
        int yT=yBB/2-TOTAL_Y/2;
        g.translate(xT,yT); //because of later transformation to pdf and png  - to be in the center of the page
        
        g.setFont(myFont);
        g.setStroke(THICK);
        g.setColor(Color.BLACK);
        g.drawLine(1,1,1,TOTAL_Y);
        g.drawLine(wBox,1,wBox,TOTAL_Y);
        g.drawLine(1,TOTAL_Y,wBox,TOTAL_Y);
        g.drawLine(1,1,wBox,1);
        g.setColor(Color.BLACK);
        g.drawRect(380,3,148,22);
          
        g.drawString("Data: " + datasetName,10,20);
        g.drawString("Model: " + modelName,10,40);
        g.drawString("Instance No.: " + (id+1),10,60);
        //g.drawString("Model: " + modelName,215,20); //orig. 
        g.drawString("Instance Explanation",385,20);
        //g.drawString("Explanation time per instance : " + time + " seconds",10,40);
            
        if (isClassification){    
            String actValue= instance.instance(0).classAttribute().value((int)instance.instance(0).classValue()).replace(',','.');
            String predStr=""+((prediction==0 || prediction==1)? (int)prediction : FeatConstr.rnd3(prediction));
            g.drawString("Explaining class: " + instance.instance(0).classAttribute().value(classValueToExplain) +" Prediction: p(class = "+instance.instance(0).classAttribute().value(classValueToExplain)+"|x)= "+predStr,10,100);
            g.drawString("Actual value for this instance: class = " + actValue,10,120);
        }
        else{
            g.drawString("Prediction: p = " + roundDecimal2(prediction).replace(',','.'),10,100);    
            g.drawString("Actual value for this instance: " + roundDecimal2(instance.instance(0).value(contributions.length)).replace(',','.'),10,120);
        }

        g.drawString("Feature",15,getY(minY-10));
        g.drawString("Value",getX(maxX + 35) + 30,getY(minY-10));
        g.drawString("Contribution",getX(0-42),getY(minY-10));
              
        g.setStroke(ROUNDED);
        int counter = 0;
        for (int i = 0; i < contributions.length; i++){
            if (Math.abs(contributions[i]) >= threshold){
                // text for feature
                int textSize = fontSize2;
	        Font tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
	        g.setFont(tempFont);
	
	        String attVal = instance.instance(0).toString(i);
	        double yText = perFeature*(counter) + minY;
	        g.setColor(Color.BLACK);
	        g.drawString(instance.attribute(i).name() + " ", textLeft+20,getY(yText+fontSize+3));
	        g.drawString(" " + formatValue(attVal,13), getX(maxX + 5) + 30,getY(yText+fontSize+3));

	        // bar for feature
	        double y = perFeature*(counter) + minY;
	        double y2 = perFeature*(counter+1) + minY;
	        g.setStroke(DASHED);
	        g.drawLine(getX(minX),getY(y2),getX(maxX),getY(y2));
	        g.setStroke(NORMAL);
	        g.setColor(Color.GRAY);
	
                double barH = (int)(perFeature * ratio);
	        double barTop = y + (perFeature- barH) / 2;
	        double x1 = Math.min((contributions[i]/maxContrib) * maxX,0);
	        double x2 = Math.abs((contributions[i]/maxContrib) * maxX);
                //System.out.println("x1: "+x1+" x2: "+x2);
                //System.out.println("get(x1): "+getX(x1)+" (int)Math.ceil(x2): "+(int)Math.ceil(x2));
                /*if(contributions[i]<0)
                    x2=x2*(-1);*/
	        // System.out.println(x1);
	  
	        if (Math.abs(contributions[i]) >= 0.01){    //if threshold is applied then this is irrelevant
                    //g.fillRect(getX(x1),getY(barTop),(int)x2,(int)barH);  //orig.    
	            g.fillRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH); //fillRect(int x, int y, int width, int height)
                    g.setColor(Color.BLACK);
	            g.drawRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH); //orig. (int)x2
	        }
	                    
	        if (1==1){  //???????????
                    int costumOffsetX = -45;    //odmik za okvirček z vrednostmi / prispevki
	            g.setStroke(ROUNDED);
	            g.setColor(Color.WHITE);
	            g.fillRect(getX(maxX - 40-costumOffsetX),getY(yText+13), 48, 12);
	            g.setColor(Color.BLACK);
	            g.drawRect(getX(maxX - 40-costumOffsetX),getY(yText+13), 48, 12);
	            g.drawString(padLeft(roundDecimal3(contributions[i]).replace(',','.'), " ", 8),getX(maxX - 48-costumOffsetX),getY(yText+22)); //vrednost prispevka za posamezni atribut
	            g.setStroke(NORMAL);
	        }
            }
                	
            if (Math.abs(contributions[i]) >= threshold) 
                counter++;
        }
        
        double y = perFeature*(0) + minY;
        g.setStroke(DASHED);
        g.drawLine(getX(minX),getY(y),getX(maxX),getY(y));
        // axis & scale	
        g.setStroke(NORMAL);
        g.drawLine(getX(0),getY(MAX_Y),getX(0),getY(minY));
        Font tempFont2 = new Font(Font.MONOSPACED, Font.BOLD, fontSize2-1);
        g.setFont(tempFont2);

        g.drawLine(getX(minX),getY(MAX_Y + 20),getX(maxX),getY(MAX_Y + 20));
                           
        String[] tick = new String[5];
        tick[0] = "-"+roundDecimal2(maxContrib);
        tick[1] = "-"+roundDecimal2(maxContrib/2);
        tick[2] = "  0";
        tick[3] = roundDecimal2(maxContrib/2);
        tick[4] = roundDecimal2(maxContrib);
                
        for (int k = 0; k < 5; k++){
            g.drawLine(getX(((maxX - minX) / 4)*k-maxX),getY(MAX_Y + 24),getX(((maxX - minX) / 4)*k-maxX),getY(MAX_Y + 20));
            g.drawString(tick[k],getX(((maxX - minX) / 4)*(k-0.3)-maxX+3),getY(MAX_Y + 24+20) );
        }

        g.flush();
        g.close();
        finalImage.close();
    }
    
    catch (Exception e){
        System.out.println("ERROR: "+e);
    }
}

static String formatValue(String s, int size){
    boolean inDecimal = false;
    int[] remove = new int[s.length()];
    int counter = 0;
    for (int i = 0; i < s.length(); i++){
        if (inDecimal) 
            counter++;
        if (Character.isDigit(s.charAt(i))){
            if (counter > 2) remove[i] = 1;
	}
	else{
            inDecimal = false;
	    counter = 0;
	}

	if (s.charAt(i) == '.' && !inDecimal){
            inDecimal = true;
	    counter = 0;
	}
    }
    
    String sNew = "";
    for (int i = 0; i < s.length(); i++) 
        if (remove[i] != 1) 
            sNew += s.charAt(i);
    
    s = sNew.replace("\\", "").replace("'","");
    while (s.length() < size){
        s = " " + s + " ";
    }
    
    if (s.length() > size) 
        return s.substring(0,size);

    return s;
}	
	   
static String padLeft(String s, String c, int size){
    while (s.length() < size) 
        s = c + s;
    return s;
}

}
