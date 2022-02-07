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

public class Visualize{
    final static float DASH[] = {10.0f};
    final static BasicStroke DASHED = new BasicStroke(1.0f,BasicStroke.CAP_BUTT,BasicStroke.JOIN_MITER,10.0f, DASH, 0.0f);
    final static BasicStroke ROUNDED = new BasicStroke(1.0f,BasicStroke.CAP_ROUND,BasicStroke.JOIN_MITER,4.0f);
    final static BasicStroke THICK = new BasicStroke(1.0f);
    final static BasicStroke NORMAL = new BasicStroke(1.0f);
    final static BasicStroke THIN = new BasicStroke(0.5f);
    final static BasicStroke BOLD = new BasicStroke(2.0f);
    static int VIS_SIZE = 500;	

    public Visualize(){				
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
            int coordX = 45;    //left/right margin
            int coordXlength = VIS_SIZE2 - 2 * coordX;
            int coordY = 150;
            int coordYlength = 100;           
            int sign_size = 2;  //dot size when drawing contributions and stdev of contribution
            //int sign_step = (int)(sign_size/2.0);  		
            int yImg = (data.numAttributes()-1)*(coordYlength+20)+34;   //height of bounding box (A4) ... data.numAttributes()-1 ... we draw just attributes withouth class 
            FileOutputStream finalImage = new FileOutputStream(file);
            EpsGraphics2D g;
            if(format.equals("A4"))
                g = new EpsGraphics2D("Title", finalImage, 0, 0, xBB, yBB);
            else
                g = new EpsGraphics2D("Title", finalImage, 0, 0, VIS_SIZE2+20, data.numAttributes()*(coordYlength+20));

            //center picture to bounding box
            int xT=xBB/2-(VIS_SIZE2+20)/2;
            int yT=yBB/2-yImg/2;
            g.translate(xT,yT); //because of later transformation to pdf and png  - to be in the center of the page

            g.setFont(myFont14);
            g.setColor(Color.BLACK);               

            //outer graph visualization     
            //dataset & model print
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

            double max_val = -Double.MAX_VALUE;
            double min_val = Double.MAX_VALUE;

            for(int i = 0; i < dotsA.length; i++){
                ArrayList temp = dotsA[i];
                ArrayList temp2 = dotsB[i]; //values that represent ​​informativeness of attributes
                for (int j = 0; j < temp.size() / 2; j++){
                    double d = (Double)(temp.get(j*2+1));
                    if(d > max_val) 
                        max_val = d;
                    if(d < min_val) 
                        min_val = d;
                    d = (Double)(temp2.get(0));
                    if(d > max_val) 
                        max_val = d;
                    if(d < min_val) 
                        min_val = d;
                }
            }
            max_val+=max_val*5/100.0;   //we increase the maximum value of the x axis for better visibility
            min_val+=min_val*5/100.0;   //we increase the min value of the x axis in the negative direction - we increase the range

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

                //attribute text
                g.setColor(Color.GRAY);
                g.setFont(myFont10);

                double chunkSize = coordYlength/(max_val - min_val);

                //base lines
                g.setStroke(NORMAL);
                g.drawLine(coordX,coordY+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20));     //bottom line
                g.drawLine(coordX,coordY+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));    //left line of the rectangle
                g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20));   //right line of the rectangle
                g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));  //top line

                //zero axis
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

            //horizontal and vertical limiters
            g.setFont(myFont10);
            int ylabels = 4;
            for(int j = 0; j < ylabels+1; j++){
                //labels on the x axis
                g.drawLine(coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet-5, coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet+5);
                //labels and values on the y axis
                g.drawLine(coordX + coordXlength-5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize), coordX + coordXlength+5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
                g.drawString(roundDecimal2(j*((max_val - min_val)/ylabels)+min_val), coordX + coordXlength+15,  coordY + 3 + i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
            }

            for (int j = 0; j < temp.size() / 2; j++){
                double x = (Double)(temp.get(j*2));
                double y = (Double)(temp.get(j*2+1));    		

                if (!data.attribute(i).isNominal()){
                    g.setColor(Color.BLACK);
                    g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
                }
                else{
                    sign_size+=2;
                    g.setStroke(BOLD);
                    g.setColor(Color.BLACK);
                    g.fillOval((int)(((x - minX) / (maxX - minX)) * +coordXlength - (sign_size / 2.0)) + coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY, sign_size, sign_size);
                    sign_size-=2;
                }

            }

            //draw attribute importance
            double x1=(Double)(temp.get(0));
            double x2=(Double)(temp.get(temp.size()-2));
            double y = (Double)(temp2.get(0));
            g.setColor(Color.getHSBColor(121, 83, 54));
            g.setStroke(NORMAL);
            g.setFont(myFont8);
            width1=g.getFontMetrics().stringWidth("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)));
            if (data.attribute(i).isNominal()){ 
                g.drawLine(coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                        (int)((((data.attribute(i).numValues()-1) - minX) / (maxX - minX)) * +coordXlength) + coordX, 
                        (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                g.drawString("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)),
                        coordX+(int)(((double)coordXlength/2)-width1/2),
                        (int)((((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY)+15);
            }
            else{
                g.drawLine(coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                        (int)(((x2 - minX) / (maxX - minX)) * +coordXlength) + coordX, 
                        (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                g.drawString("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)), 
                        coordX+(int)(((double)coordXlength/2)-width1/2),
                        (int)((((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY)+15);
            }
            //horizontal line
            g.setColor(Color.GRAY);
            }
            g.flush();
            g.close();
            finalImage.close();
        }
        catch(Exception e){
            System.err.println("ERROR: "+e);
        }			
    }
    
    public static void attrImportanceVisualizationSorted(String file, String modelName, String datasetName, Instances data, ArrayList dotsB[], boolean classification, int resolution,String format){
        int xBB = 595;      //width of bounding box (A4)
        int yBB = 842;      //height of bounding box (A4)
        int wBox=530;       //width of the box for drawing ... VIS_SIZE is currently 500
        int fontSize2 = 14;
        int fontSize = 14;
        int leadinY = 80;   //the offset between the top edge and the label Feature Contribution in Value
        int minY = leadinY;
        int leadoutY = 70;  //the distance between the end of the inscriptions (numbers on the axis) and the bottom edge
        double perFeature = 30;
        double ratio = 0.6;
        int maxX = +150;
        int minX = 0;
        int textLeft = -10; //-10 because we then add +20
        int drawLimit=20;   //we draw only 20 the most important attributes
        String fontName = Font.SANS_SERIF;
        Font myFont = new Font(fontName, Font.BOLD, fontSize);
        double threshold=0.03;

        Map<Double, String> treemap = new TreeMap<>();    
        for(int i=0;i<dotsB.length;i++){
            treemap.put((double)dotsB[i].get(0), data.attribute(i).name());
        }

        //sort (descending) map based on attr. importance
        Map<Double, String> newMap = new TreeMap<>(Collections.reverseOrder());
            newMap.putAll(treemap);
            if(newMap.size()>drawLimit){    
                System.out.println("Drawing limit for attribute name: "+newMap.get(newMap.keySet().toArray()[drawLimit-1]));    //attribute name
                System.out.println("Drawing limit for value: "+newMap.keySet().toArray()[drawLimit-1]);                 //attribute value
                threshold=Double.parseDouble(String.valueOf(newMap.keySet().toArray()[drawLimit-1]));
                newMap.keySet().removeAll(Arrays.asList(newMap.keySet().toArray()).subList(drawLimit, newMap.size()-1));
            }
            else
                threshold=Double.parseDouble(String.valueOf(newMap.keySet().toArray()[newMap.size()-2]));
            
            System.out.println("List size - map: "+newMap.size()); 
            System.out.println("Test printout - limit: "+Double.parseDouble(String.valueOf(newMap.keySet().toArray()[newMap.size()-2]))); 

            int relevantFeatures = 0;
            double maxContrib = 0;
            for (int i = 0; i < dotsB.length; i++){
                if (Math.abs((double)dotsB[i].get(0)) >= threshold) 
                    relevantFeatures++;
                if (Math.abs((double)dotsB[i].get(0)) >= maxContrib) 
                    maxContrib = Math.abs((double)dotsB[i].get(0));
            }

            int TOTAL_Y = (int)(leadinY + perFeature * relevantFeatures + leadoutY);
            double MAX_Y = leadinY + perFeature * relevantFeatures; //vertical line between positive and negative part

        try{
            FileOutputStream finalImage = new FileOutputStream(file);
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
            g.drawString("Attribute importance",385,20);
            g.drawString("Feature",15,getY(minY-10));
            g.drawString("Value",getX(maxX + 35) + 30,getY(minY-10));
            g.drawString("Importance",getX(0-42),getY(minY-10));
            g.setStroke(ROUNDED);
            
            int counter = 0;
            for (Entry<Double, String> entry : newMap.entrySet()){
                double value = entry.getKey();
                String attrName = entry.getValue();
                if (value >= threshold){
                    //text for feature
                    int textSize = fontSize2;
                    Font tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
                    g.setFont(tempFont);

                    String attVal = value+"";
                    double yText = perFeature*(counter) + minY;
                    g.setColor(Color.BLACK);
                    g.drawString(attrName + " ", textLeft+20,getY(yText+fontSize+3));
                    g.drawString(" " + formatValue(attVal,13), getX(maxX + 5) + 30,getY(yText+fontSize+3));

                    //bar for feature
                    double y = perFeature*(counter) + minY;
                    double y2 = perFeature*(counter+1) + minY;
                    g.setStroke(DASHED);
                    g.drawLine(VIS_SIZE/4,getY(y2),getX(maxX),getY(y2));
                    g.setStroke(NORMAL);
                    g.setColor(Color.GRAY);

                    double barH = (int)(perFeature * ratio);
                    double barTop = y + (perFeature- barH) / 2;
                    double x1 = Math.min((value/maxContrib) * (getX(maxX)-VIS_SIZE/4),0);
                    double x2 = Math.abs((value/maxContrib) * (getX(maxX)-VIS_SIZE/4));

                    if (value >= 0.01){    //if threshold is applied then this is irrelevant
                        g.fillRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
                        g.setColor(Color.BLACK);
                        g.drawRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
                    }
                }

                if (Math.abs(value) >= threshold) 
                    counter++;
            }

            double y = perFeature*(0) + minY;
            g.setStroke(DASHED);
            g.drawLine(VIS_SIZE/4,getY(y),getX(maxX),getY(y)); //first dashed line
            //axis & scale	
            g.setStroke(NORMAL);
            Font tempFont2 = new Font(Font.MONOSPACED, Font.BOLD, fontSize2-1);
            g.setFont(tempFont2);
            g.drawLine(VIS_SIZE/4,getY(MAX_Y + 20),getX(maxX),getY(MAX_Y + 20)); //bottom black line ... the axis where we place values
            String[] tick = new String[3];
            tick[0] = ""+minX;
            tick[1] = roundDecimal2(maxContrib/2);
            tick[2] = roundDecimal2(maxContrib);

            //drawing labels on the X axis
            //first vertical line 0
            int width =g.getFontMetrics().stringWidth((tick[0]));
            g.drawLine(VIS_SIZE/4,getY(MAX_Y + 24),VIS_SIZE/4,getY(MAX_Y + 20)); //vertical lines on the axis
            g.drawString(tick[0],VIS_SIZE/4-width/2,getY(MAX_Y + 24+20) );
            //the second half of the vertical line
            width=g.getFontMetrics().stringWidth((tick[1]));
            g.drawLine(VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2,getY(MAX_Y + 24),VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2,getY(MAX_Y + 20)); //vertical lines on the axis
            g.drawString(tick[1],VIS_SIZE/4+(getX(maxX)-VIS_SIZE/4)/2-width/2,getY(MAX_Y + 24+20) );
            //third vertical line maximum
            width=g.getFontMetrics().stringWidth((tick[2]));
            g.drawLine(getX(maxX),getY(MAX_Y + 24),getX(maxX),getY(MAX_Y + 20)); //vertical lines on the axis
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
        int xBB = 595;      //width of bounding box (A4)
        int yBB = 842;      //height of bounding box (A4)
        int wBox=530;       //width of the box for drawing
        int fontSize2 = 14;
        int fontSize = 14;
        int leadinY = 170;  //the offset between the top edge and the label Feature Contribution in Value
        int minY = leadinY;
        int leadoutY = 70;  //the distance between the end of the inscriptions (numbers on the axis) and the bottom edge
        double perFeature = 30;
        double ratio = 0.6;
        int maxX = +150;
        int minX = -150;
        int textLeft = -10;  //-10 because we then add +20
        String fontName = Font.SANS_SERIF;
        Font myFont = new Font(fontName, Font.BOLD, fontSize);
        double threshold=-1;

        //threshold calculation
        double contrCp[]=contributions.clone();
        for (int i = 0; i < contrCp.length; i++){
            if (contrCp[i]<0)
                contrCp[i]=Math.abs(contrCp[i]);
        }
        Arrays.sort(contrCp);
        if(contrCp.length>topHigh)
            threshold=contrCp[contrCp.length-topHigh];

        int relevantFeatures = 0;
        double maxContrib = 0;
        for (int i = 0; i < contributions.length; i++){
            if (Math.abs(contributions[i]) >= threshold) 
                relevantFeatures++;
            if (Math.abs(contributions[i]) >= maxContrib) 
                maxContrib = Math.abs(contributions[i]);
        }

        int TOTAL_Y = (int)(leadinY + perFeature * relevantFeatures + leadoutY);
        double MAX_Y = leadinY + perFeature * relevantFeatures; //vertical line between positive and negative part

        try{
            FileOutputStream finalImage = new FileOutputStream(file);
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
            g.drawString("Instance Explanation",385,20);

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
                    //text for feature
                    int textSize = fontSize2;
                    Font tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
                    g.setFont(tempFont);

                    String attVal = instance.instance(0).toString(i);
                    double yText = perFeature*(counter) + minY;
                    g.setColor(Color.BLACK);
                    g.drawString(instance.attribute(i).name() + " ", textLeft+20,getY(yText+fontSize+3));
                    g.drawString(" " + formatValue(attVal,13), getX(maxX + 5) + 30,getY(yText+fontSize+3));

                    //bar for feature
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

                    if (Math.abs(contributions[i]) >= 0.01){    //if threshold is applied then this is irrelevant
                        g.fillRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH);
                        g.setColor(Color.BLACK);
                        g.drawRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH);
                    }

                    int costumOffsetX = -45;    //offset for box with values/contributions
                    g.setStroke(ROUNDED);
                    g.setColor(Color.WHITE);
                    g.fillRect(getX(maxX - 40-costumOffsetX),getY(yText+13), 48, 12);
                    g.setColor(Color.BLACK);
                    g.drawRect(getX(maxX - 40-costumOffsetX),getY(yText+13), 48, 12);
                    g.drawString(padLeft(roundDecimal3(contributions[i]).replace(',','.'), " ", 8),getX(maxX - 48-costumOffsetX),getY(yText+22)); //contribution for each attribute
                    g.setStroke(NORMAL);
                }

                if (Math.abs(contributions[i]) >= threshold) 
                    counter++;
            }

            double y = perFeature*(0) + minY;
            g.setStroke(DASHED);
            g.drawLine(getX(minX),getY(y),getX(maxX),getY(y));
            //axis & scale	
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

    public static String formatValue(String s, int size){
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
	   
    public static String padLeft(String s, String c, int size){
        while (s.length() < size) 
            s = c + s;
        return s;
    }
}
