/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import java.io.FileWriter;
import java.io.IOException;


/**
 *
 * @author phillipperks
 * @param <C>
 */
public class ClassifierWrapper <C extends weka.classifiers.Classifier> {
    
    private C classifier;
    private int correct;
    private double accuracy;
    private Instances test;
    private Instances train;
    
    public ClassifierWrapper(C classifier, Instances test, Instances train ) throws Exception{
        this.classifier = classifier;
        this.correct = 0;
        this.train = train;
        this.test = test;
        
        this.classifier.buildClassifier(this.train);
        for(Instance i: test){
            double result = this.classifier.classifyInstance(i);
            if(result == i.value(this.test.numAttributes()-1)){
                this.correct++;
            }
            
        }
        this.accuracy = (double)this.correct/test.numInstances();
        
        
    }
    
    //returns the classifier
    public C getClassifier(){
        return classifier;
    }
    
    //classifies Instance i using this classifier
    public double classifyInstance(Instance i) throws Exception{
        return classifier.classifyInstance(i);
    }
    
    //returns the training Instances
    public Instances getTrain(){
        return train;
    }
    
    //returns the test Instances
    public Instances getTest(){
        return test;
    }
    
    //returns accuracy of classifier
    public double getAccuracy(){
        return accuracy;
    }
    
    //returns number of times the correct action was classified
    public double getCorrect(){
        return correct;
    }
    
    //classifies all instances and prints the accuracy as well as a confusion matrix
    public void classifyAllInstances(){
        StringBuilder str = new StringBuilder();
        for(int i=0; i<50; i++){
            str.append("=");
        }
        
        str.append("\n\n").append("Class Name: ");
        str.append(classifier.getClass().getName()).append("\n");
        
        str.append("Accuracy: ").append((double)correct/test.numInstances()*100);
        str.append("% \n\n");
        
        int classes = test.numClasses();
        int[][] matrix = new int [classes][classes];
        for(Instance i: test){
            try {
                matrix[(int)classifier.classifyInstance(i)][(int)i.classValue()] += 1;
            } catch (Exception ex) {
                Logger.getLogger(ClassifierWrapper.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        str.append("\t");
        
        for(int i=0; i< classes; i++){
            str.append((double)i).append("\t");
        }
        
        str.append("\n");
        for(int i=0; i<20; i++){
            str.append("-");
        }
         str.append("\n");
        
        for(int i=0; i< classes; i++){
            str.append((double)i).append("|").append("\t");
            for(int j=0; j< classes; j++){
                str.append(matrix[i][j]).append("\t");
            }
            if(i < classes-1){
                str.append("\n   |\n"); 
            }
            else{
                str.append("\n\n");
            }
        }
         System.out.println(str.toString());
    }
    
}
