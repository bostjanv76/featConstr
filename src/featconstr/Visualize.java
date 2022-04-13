package featconstr;

import org.sourceforge.jlibeps.epsgraphics.*;
import java.awt.*;
import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
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
    static int numDec=3;        //number of decimal places in attribute importance image
  
    //credits to Erik Štrumbelj
    public static void modelVisualToFileAttrImptLine(String file, String modelName, String datasetName, Instances data, ArrayList<Double>[] dotsA, ArrayList<Double> dotsB[], boolean classification, int resolution, int classValueToExplain, String format, boolean fc){
        int VIS_SIZE2 = 400;
        int xBB = 595;        //width of bounding box (A4)
        int yBB = 842;        //height of bounding box (A4)
        String fontName = Font.SANS_SERIF;
        String tmpVal;
        Font myFont14 = new Font(fontName, Font.BOLD, 14);
        Font myFont10 = new Font(fontName, Font.BOLD, 10);
        Font myFont8 = new Font(fontName, Font.BOLD, 8);
        Font myFont5 = new Font(fontName, Font.BOLD, 5);

        // start drawing the picture
        FileOutputStream finalImage;
        EpsGraphics2D g;
        try{
            int coordX = 45;    //left/right margin
            int coordXlength = VIS_SIZE2 - 2 * coordX;
            int coordY = 150;
            int coordYlength = 100;           
            int sign_size = 2;  //dot size when drawing contributions and stdev of contributions 		
            int yImg = (data.numAttributes()-1)*(coordYlength+20)+34;   //height of bounding box (A4) ... data.numAttributes()-1 ... we draw just attributes withouth class 
            finalImage = new FileOutputStream(file);            
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
            
            ArrayList<Double> temp, temp2;
            double d;
            for(int i = 0; i < dotsA.length; i++){
                temp = dotsA[i];
                temp2 = dotsB[i]; //values that represent ​​informativeness of attributes
                for(int j = 0; j < temp.size() / 2; j++){
                    d = (temp.get(j*2+1));
                    if(d > max_val) 
                        max_val = d;
                    if(d < min_val) 
                        min_val = d;
                    d = (temp2.get(0));
                    if(d > max_val) 
                        max_val = d;
                    if(d < min_val) 
                        min_val = d;
                }
            }
            max_val+=max_val*5/100.0;   //we increase the maximum value of the x axis for better visibility
            min_val+=min_val*5/100.0;   //we increase the min value of the x axis in the negative direction - we increase the range

            double maxX, minX, chunkSize, axisOffSet, y, x, x2;
            int ylabels;
            for(int i = 0; i < data.numAttributes() -1; i++){            
                maxX = Double.MIN_VALUE;
                minX = Double.MAX_VALUE;
                if(fc)
                    g.setFont(myFont8);
                else
                    g.setFont(myFont10);
                g.setColor(Color.GRAY);
                g.drawString(data.attribute(i).name(),coordX + 2,(i-1)*(coordYlength+20) + coordY+20 - 5);
                temp = dotsA[i];
                temp2 = dotsB[i];
                if(data.attribute(i).isNominal()){
                    maxX = data.attribute(i).numValues() - 1;
                    minX = 0;
                }
                else
                    for(int j = 0; j < temp.size() / 2; j++){
                        d = (temp.get(j*2));
                        if(d > maxX) 
                            maxX = d;
                        if(d < minX) 
                            minX = d;
                }

                //attribute text
                g.setColor(Color.GRAY);
                g.setFont(myFont10);

                chunkSize = coordYlength/(max_val - min_val);

                //base lines
                g.setStroke(NORMAL);
                g.drawLine(coordX,coordY+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20));     //bottom line
                g.drawLine(coordX,coordY+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));    //left line of the rectangle
                g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX + coordXlength,coordY+i*(coordYlength+20));   //right line of the rectangle
                g.drawLine(coordX + coordXlength,coordY - coordYlength+i*(coordYlength+20),coordX ,coordY - coordYlength+i*(coordYlength+20));  //top line

                //zero axis
                axisOffSet = 0;
                if(min_val < 0) 
                    axisOffSet = min_val*chunkSize;
                g.setStroke(DASHED);
                g.drawLine(coordX,coordY+i*(coordYlength+20)+(int)axisOffSet,coordX+coordXlength,coordY+i*(coordYlength+20)+(int)axisOffSet);

                //zero axis start and end numbers
                if(data.attribute(i).isNominal()){
                    g.setFont(myFont5);
                    for(int j = 0; j < data.attribute(i).numValues();j++){
                        if(data.attribute(i).value(j).contains("-inf") || data.attribute(i).value(j).contains("All") || data.attribute(i).value(j).contains("_x_")){
                            tmpVal = data.attribute(i).value(j).replace("\\", "").replace("'","");
                            g.drawString(tmpVal,(int)(((j - minX) / (maxX - minX)) * +coordXlength) + coordX, coordY+i*(coordYlength+20)+(int)axisOffSet +10);
                        }
                        else
                            g.drawString(data.attribute(i).value(j),(int)(((j - minX) / (maxX - minX)) * +coordXlength) + coordX, coordY+i*(coordYlength+20)+(int)axisOffSet +10);
                    }
                }
                else{
                    g.setFont(myFont5);
                    g.drawString(roundDecimal2(minX),coordX -10,coordY+i*(coordYlength+20)+(int)axisOffSet +10);	      
                    g.drawString(roundDecimal2(maxX),coordX+coordXlength  - 20,coordY+i*(coordYlength+20)+(int)axisOffSet +10);
                }

            //horizontal and vertical limiters
            g.setFont(myFont8);
            ylabels = 4;
            for(int j = 0; j < ylabels+1; j++){
                //labels on the x axis
                g.drawLine(coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet-5, coordX+(int)(j*((double)coordXlength/ylabels)), coordY+i*(coordYlength+20)+(int)axisOffSet+5);
                //labels and values on the y axis
                g.drawLine(coordX + coordXlength-5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize), coordX + coordXlength+5, coordY+i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
                g.drawString(roundDecimal2(j*((max_val - min_val)/ylabels)+min_val), coordX + coordXlength+15,  coordY + 3 + i*(coordYlength+20)-(int)(j*((max_val - min_val)/ylabels)*chunkSize));
            }

            for(int j = 0; j < temp.size() / 2; j++){
                x = (temp.get(j*2));
                y = (temp.get(j*2+1));    		

                if(!data.attribute(i).isNominal()){
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
            x2=(temp.get(temp.size()-2));
            y =(temp2.get(0));
            g.setColor(Color.getHSBColor(121, 83, 54));
            g.setStroke(NORMAL);
            g.setFont(myFont8);
            if(fc)
                width1=g.getFontMetrics().stringWidth("Feat. importance: "+roundDecimal3((double)dotsB[i].get(0)));
            else
                width1=g.getFontMetrics().stringWidth("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)));
            if(data.attribute(i).isNominal()){ 
                g.drawLine(coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                        (int)((((data.attribute(i).numValues()-1) - minX) / (maxX - minX)) * +coordXlength) + coordX, 
                        (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                if(fc)
                    g.drawString("Feat. importance: "+roundDecimal3((double)dotsB[i].get(0)),
                        coordX+(int)(((double)coordXlength/2)-width1/2),
                        (int)((((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY)+15);
                else
                    g.drawString("Attr. importance: "+roundDecimal3((double)dotsB[i].get(0)),
                        coordX+(int)(((double)coordXlength/2)-width1/2),
                        (int)((((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY)+15);
            }
            else{
                g.drawLine(coordX, (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY,
                        (int)(((x2 - minX) / (maxX - minX)) * +coordXlength) + coordX, 
                        (int)(((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY);
                if(fc)
                    g.drawString("Feat. importance: "+roundDecimal3((double)dotsB[i].get(0)), 
                        coordX+(int)(((double)coordXlength/2)-width1/2),
                        (int)((((y - min_val) / (max_val - min_val)) * -coordYlength - sign_size / 2.0) + (i)*(coordYlength+20) + coordY)+15);
                else
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
        catch(IOException e){
            System.err.println("ERROR: "+e.toString());
        }			
    }
    
    public static void attrImportanceVisualizationSorted(String file, String modelName, String datasetName, Instances data, int drawLimit, ArrayList<Double> dotsB[], boolean classification, int resolution,String format, boolean fc){
        int xBB = 595;      //width of bounding box (A4)
        int yBB = 842;      //height of bounding box (A4)
        int wBox=530;       //width of the box for drawing ... VIS_SIZE is currently 500
        int fontSize2 = 9;
        int fontSize = 14;
        int leadinY = 80;   //the offset between the top edge and the label Feature Contribution in Value
        int minY = leadinY;
        int leadoutY = 70;  //the distance between the end of the inscriptions (numbers on the axis) and the bottom edge
        double perFeature = 30;
        double ratio = 0.6;
        int maxX = +150;
        int minX = 0;
        int textLeft = -10; //-10 because we then add +20
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
                System.out.printf("Drawing limit for value: %.4f\n",newMap.keySet().toArray()[drawLimit-1]); //attribute value
                threshold=Double.parseDouble(String.valueOf(newMap.keySet().toArray()[drawLimit-1]));
                newMap.keySet().removeAll(Arrays.asList(newMap.keySet().toArray()).subList(drawLimit, newMap.size()-1)); //subList(fromIndex - inclusive, toIndex - exclusive)
            }
            else
                threshold=Double.parseDouble(String.valueOf(newMap.keySet().toArray()[newMap.size()-1]));
            
            int relevantFeatures = 0;
            double maxContrib = 0;
            for(int i = 0; i < dotsB.length; i++){
                if(Math.abs((double)dotsB[i].get(0)) >= threshold) 
                    relevantFeatures++;
                if(Math.abs((double)dotsB[i].get(0)) >= maxContrib) 
                    maxContrib = Math.abs((double)dotsB[i].get(0));
            }

            int TOTAL_Y = (int)(leadinY + perFeature * relevantFeatures + leadoutY);
            double MAX_Y = leadinY + perFeature * relevantFeatures; //vertical line between positive and negative part
        
            FileOutputStream finalImage;
            EpsGraphics2D g;
        try{
            finalImage= new FileOutputStream(file);
            g = new EpsGraphics2D("Title", finalImage, 0,0, xBB, yBB); //A4 beacause if we later convert eps to pdf and png (parameters are set to center image on A4)

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
            if(fc){
                g.drawString("Feature importance",385,20);
                g.drawString("Feature",15,getY(minY-10));
            }
            else{    
                g.drawString("Attribute importance",385,20);
                g.drawString("Attribute",15,getY(minY-10));
            }
            g.drawString("Value",getX(maxX + 35) + 30,getY(minY-10));
            g.drawString("Importance",getX(0-42),getY(minY-10));
            g.setStroke(ROUNDED);
            
            int counter = 0;
            BigDecimal bd;
            Font tempFontA, tempFontV;
            String attrName, padded, attVal;
            double value, yText, y, y2, barH, barTop, x2;
            for(Entry<Double, String> entry : newMap.entrySet()){
                value = entry.getKey();
                attrName = entry.getValue();
                if(value >= threshold){
                    //text for feature
                    int textSize = fontSize2;
                    tempFontA = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
                    tempFontV = new Font(Font.MONOSPACED, Font.BOLD, textSize);
   
                    bd = new BigDecimal(value).setScale(numDec, RoundingMode.HALF_UP);
                    padded = String.format("%-5s", bd.doubleValue()).replace(' ', '0'); //rpad
                    attVal = padded;
                    yText = perFeature*(counter) + minY;
                    g.setColor(Color.BLACK);
                    g.setFont(tempFontV);
                    if(fc)
                        g.drawString("Feat " +(counter+1)+ " ", textLeft+20,getY(yText+fontSize+3));
                    else
                        g.drawString(attrName + " ", textLeft+20,getY(yText+fontSize+3));
                    g.setFont(tempFontA);
                    if(fc){
                        g.setFont(tempFontA);
                        g.drawString(attrName + " ", textLeft+40,getY(yText+fontSize+17));
                    }
                    g.setFont(tempFontV);
                    g.drawString(" " + formatValue(attVal,13,numDec), getX(maxX + 5) + 30,getY(yText+fontSize+3));
                                       
                    //bar for feature
                    y = perFeature*(counter) + minY;
                    y2 = perFeature*(counter+1) + minY;
                    g.setStroke(DASHED);
                    g.setColor(Color.GRAY);
                    if(!fc)
                        g.drawLine(VIS_SIZE/4,getY(y2),getX(maxX),getY(y2));
                    g.setStroke(NORMAL);
                   
                    barH = (int)(perFeature * ratio);
                    barTop = y + (perFeature- barH) / 2;
                    x2 = Math.abs((value/maxContrib) * (getX(maxX)-VIS_SIZE/4));

                    g.fillRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
                    g.setColor(Color.BLACK);
                    g.drawRect(VIS_SIZE/4,getY(barTop),(int)Math.ceil(x2),(int)barH);
                }

                if(Math.abs(value) >= threshold) 
                    counter++;
            }

            y = perFeature*(0) + minY;
            g.setStroke(DASHED);
            if(!fc)
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

        catch (IOException e){
            System.out.println("ERROR: "+e.toString());
        }			
    }
    
    //credits to Erik Štrumbelj
    public static void instanceVisualizationToFile(String file, String modelName, String datasetName, Instances instance, int id, int topHigh, double[] contributions, double prediction, int classValueToExplain, boolean isClassification, boolean fc){				
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
        for(int i = 0; i < contrCp.length; i++){
            if(contrCp[i]<0)
                contrCp[i]=Math.abs(contrCp[i]);
        }
        Arrays.sort(contrCp);
        if(contrCp.length>topHigh)
            threshold=contrCp[contrCp.length-topHigh];

        int relevantFeatures = 0;
        double maxContrib = 0;
        for(int i = 0; i < contributions.length; i++){
            if(Math.abs(contributions[i]) >= threshold) 
                relevantFeatures++;
            if(Math.abs(contributions[i]) >= maxContrib) 
                maxContrib = Math.abs(contributions[i]);
        }

        int TOTAL_Y = (int)(leadinY + perFeature * relevantFeatures + leadoutY);
        double MAX_Y = leadinY + perFeature * relevantFeatures; //vertical line between positive and negative part

        FileOutputStream finalImage;
        EpsGraphics2D g;
        try{
            finalImage = new FileOutputStream(file);
            g = new EpsGraphics2D("Title", finalImage, 0,0, xBB, yBB); //A4 beacause if we later convert eps to pdf and png (parameters are set to center image on A4)

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
            g.drawString("Instance No.: " + id,10,60);             
            g.drawString("Instance Explanation",385,20);

            if(isClassification){    
                String actValue= instance.instance(0).classAttribute().value((int)instance.instance(0).classValue()).replace(',','.');
                String predStr=""+((prediction==0 || prediction==1)? (int)prediction : FeatConstr.rnd3(prediction));
                g.drawString("Explaining class: " + instance.instance(0).classAttribute().value(classValueToExplain) +" Prediction: p(class = "+instance.instance(0).classAttribute().value(classValueToExplain)+"|x)= "+predStr,10,100);
                g.drawString("Actual value for this instance: class = " + actValue,10,120);
            }
            else{
                g.drawString("Prediction: p = " + roundDecimal2(prediction).replace(',','.'),10,100);    
                g.drawString("Actual value for this instance: " + roundDecimal2(instance.instance(0).value(contributions.length)).replace(',','.'),10,120);
            }
            
            if(fc)
                g.drawString("Feature",15,getY(minY-10));
            else
                g.drawString("Attribute",15,getY(minY-10));
            g.drawString("Value",getX(maxX + 35) + 30,getY(minY-10));
            g.drawString("Contribution",getX(0-42),getY(minY-10));

            double yA = perFeature*(0) + minY;
            g.setStroke(DASHED);
            g.setColor(Color.GRAY);
            g.drawLine(getX(minX),getY(yA),getX(maxX),getY(yA)); //first dashed line
            //axis & scale	
            g.setStroke(NORMAL);
            g.drawLine(getX(0),getY(MAX_Y),getX(0),getY(minY));   //y-axis
            Font tempFont2 = new Font(Font.MONOSPACED, Font.BOLD, fontSize2-1);
            g.setFont(tempFont2);
            g.setColor(Color.BLACK);
            g.drawLine(getX(minX),getY(MAX_Y + 20),getX(maxX),getY(MAX_Y + 20));    //x-axis

            String[] tick = new String[5];
            tick[0] = "-"+roundDecimal2(maxContrib);
            tick[1] = "-"+roundDecimal2(maxContrib/2);
            tick[2] = "  0";
            tick[3] = roundDecimal2(maxContrib/2);
            tick[4] = roundDecimal2(maxContrib);

            for(int k = 0; k < 5; k++){
                g.drawLine(getX(((maxX - minX) / 4)*k-maxX),getY(MAX_Y + 24),getX(((maxX - minX) / 4)*k-maxX),getY(MAX_Y + 20));
                g.drawString(tick[k],getX(((maxX - minX) / 4)*(k-0.3)-maxX+3),getY(MAX_Y + 24+20) );
            }                        
            
            g.setStroke(ROUNDED);
            int counter = 0;
            int textSize;
            String attVal;
            boolean writeOnce=true;
            Font tempFont, tempFontA, tempFontV, tempFontZ;
            double yText, y, y2, barH, barTop, x1, x2 ;
            int costumOffsetX;
            for(int i = 0; i < contributions.length; i++){
                if(Math.abs(contributions[i]) >= threshold){
                    //text for feature
                    textSize = fontSize2;
                    tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
                    g.setFont(tempFont);

                    attVal = instance.instance(0).toString(i);
                    yText = perFeature*(counter) + minY;
                    g.setColor(Color.BLACK);
                    if(fc){
                        textSize=8;
                        tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);
                        g.setFont(tempFont);
                    }
                    textSize = 9;
                    tempFontA = new Font(Font.MONOSPACED, Font.BOLD, textSize-4);
                    tempFontV = new Font(Font.MONOSPACED, Font.BOLD, textSize);                    
                    
                    g.setFont(tempFontV);
                    if(fc)
                        g.drawString("Feat " +(counter+1)+ " ", textLeft+20,getY(yText+fontSize+3));
                    else
                        g.drawString(instance.attribute(i).name() + " ", textLeft+20,getY(yText+fontSize+3));
                    g.setFont(tempFontA);
                    if(fc){
                        g.setFont(tempFontA);
                        g.drawString(instance.attribute(i).name() + " ", textLeft+40,getY(yText+fontSize+15));
                    }
                    g.setFont(tempFontV);
                    
                    textSize=14;
                    tempFont = new Font(Font.MONOSPACED, Font.BOLD, textSize-3);

                    if(attVal.contains("-inf") || attVal.contains("All") || attVal.contains("_x_")){ //Cartesian product value
                        if(writeOnce){
                            tempFontZ = new Font(Font.MONOSPACED, Font.BOLD, 6);
                            g.setFont(tempFontZ);
                            g.drawString("*",10,getY(MAX_Y + 60) );
                            g.setColor(Color.GRAY);
                            g.drawString("Cartesian product value",14,getY(MAX_Y + 63) );
                            writeOnce=false;
                        }
    g.setFont(tempFont);
                        g.setColor(Color.BLACK);
                        attVal = attVal.replace("\\", "").replace("'","");
                        g.drawString("*", getX(maxX + 5) + 68,getY(yText+fontSize+4)); //value e.g. 1-2 is from Cartesian product after FC '-' is instead of 'x'
                        tempFontZ = new Font(Font.MONOSPACED, Font.BOLD, 4);
                        g.setFont(tempFontZ);
                        g.drawString(attVal, getX(maxX + 5)+69-(attVal.length()),getY(yText+fontSize+15)); //value e.g. 1-2 is from Cartesian product after FC '-' is instead of 'x'
                    }
                    else
                        g.drawString(" " + formatValue(attVal,13,numDec), getX(maxX + 5) + 30,getY(yText+fontSize+4)); //value e.g. 1-2 is from Cartesian product after FC '-' is instead of 'x'

                    g.setFont(tempFont);
                    //bar for feature
                    y = perFeature*(counter) + minY;
                    y2 = perFeature*(counter+1) + minY;
                    g.setStroke(DASHED);
                    g.setColor(Color.GRAY);
                    g.drawLine(getX(minX),getY(y2),getX(maxX),getY(y2));
                    g.setStroke(NORMAL);
                    g.setColor(Color.GRAY);

                    barH = (int)(perFeature * ratio);
                    barTop = y + (perFeature- barH) / 2;
                    x1 = Math.min((contributions[i]/maxContrib) * maxX,0);
                    x2 = Math.abs((contributions[i]/maxContrib) * maxX);

                    g.fillRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH);
                    g.setColor(Color.BLACK);
                    g.drawRect(getX(x1),getY(barTop),(int)Math.ceil(x2),(int)barH);
                    
                    costumOffsetX = -45;    //offset for box with values/contributions
                    g.setStroke(ROUNDED);
                    g.setColor(Color.WHITE);
                    g.fillRect(getX(maxX - 40-costumOffsetX),getY(yText+9), 48, 12);
                    g.setColor(Color.BLACK);
                    g.drawRect(getX(maxX - 40-costumOffsetX),getY(yText+9), 48, 12);
                    g.drawString(padLeft(roundDecimal3(contributions[i]).replace(',','.'), " ", 8),getX(maxX - 48-costumOffsetX),getY(yText+18.5)); //contribution for each attribute
                    g.setStroke(NORMAL);
                }

                if(Math.abs(contributions[i]) >= threshold) 
                    counter++;
            }

            g.flush();
            g.close();
            finalImage.close();
        }

        catch (IOException e){
            System.out.println("ERROR: "+e.toString());
        }
    }

    public static String roundDecimal2(double d) {
        DecimalFormat twoDForm = new DecimalFormat("#.##");
        if(twoDForm.format(d).equals("-0"))
            return "0";
        else
            return twoDForm.format(d).replace(",", ".");
    }

    public static String roundDecimal3(double d) {
        DecimalFormat twoDForm = new DecimalFormat("#.###");
        if(twoDForm.format(d).equals("-0"))
            return "0";
        else
            return twoDForm.format(d).replace(",", ".");
    }

    private static int getY(double y){
        return (int)(y);
    }

    private static int getX(double y){
        return (int)((VIS_SIZE / 2) + y);
    }
    
    public static String formatValue(String s, int size, int decPlaces){
        boolean inDecimal = false;
        int[] remove = new int[s.length()];
        int counter = 0;
        for(int i = 0; i < s.length(); i++){
            if(inDecimal) 
                counter++;
            if(Character.isDigit(s.charAt(i))){
                if (counter > decPlaces) remove[i] = 1;
            }
            else{
                inDecimal = false;
                counter = 0;
            }

            if(s.charAt(i) == '.' && !inDecimal){
                inDecimal = true;
                counter = 0;
            }
        }

        String sNew = "";
        for(int i = 0; i < s.length(); i++) 
            if(remove[i] != 1) 
                sNew += s.charAt(i);

        s = sNew.replace("\\", "").replace("'","");
        while(s.length() < size){
            s = " " + s + " ";
        }

        if(s.length() > size) 
            return s.substring(0,size);

        return s;
    }	
	   
    public static String padLeft(String s, String c, int size){
        while(s.length() < size) 
            s = c + s;
        return s;
    }
}