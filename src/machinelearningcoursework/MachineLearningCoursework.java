/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;

import java.io.FileReader;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class MachineLearningCoursework {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        Instances train = loadData("/Users/phillipperks/Documents/Year 3/Machine Learning/MachineLearningCoursework/question1.arff");
        //LinearPerceptron lp = new LinearPerceptron();
        //train.setClassIndex(train.numAttributes()-1);
        //lp.buildClassifier(train);
        
        EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
        //train.setClassIndex(train.numAttributes()-1);
        elp.buildClassifier(train);
        
    }
        public static Instances loadData(String filePath){
        String dataLocation=filePath;
        Instances i = null;
        try{
            FileReader reader = new FileReader(dataLocation);
            i = new Instances(reader);
            i.setClassIndex(i.numAttributes()-1);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
        }
        return i;
    }
}
