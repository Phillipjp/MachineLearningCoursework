/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import weka.core.Instance;
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
        Instances all = loadData("/Users/phillipperks/Documents/Year 3/Machine Learning/MachineLearningCoursework/question1.arff");
        LinearPerceptron lp = new LinearPerceptron();
        all.setClassIndex(all.numAttributes()-1);
        lp.buildClassifier(all);
        
//        EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
//        //train.setClassIndex(train.numAttributes()-1);
//        elp.buildClassifier(data);
        
        
        
        
    } 
    
    public static double [] learningAlgorithmComparison(int folds, Instances all) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper online = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,false),test, train);
            online.classifyAllInstances();
            accuracies[0] = online.getAccuracy();
            ClassifierWrapper offline = new ClassifierWrapper(new EnhancedLinearPerceptron(false,false,false),test, train);
            offline.classifyAllInstances();
            accuracies[1] = offline.getAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        
        return accuracies;
    }
    
    public static double [] onlineStandardisedComparison(int folds, Instances all) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper standardiesed = new ClassifierWrapper(new EnhancedLinearPerceptron(true,true,false),test, train);
            standardiesed.classifyAllInstances();
            accuracies[0] = standardiesed.getAccuracy();
            ClassifierWrapper online = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,false),test, train);
            online.classifyAllInstances();
            accuracies[1] = online.getAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        
        return accuracies;
    }
    
    public static double [] offlineStandardisedComparison(int folds, Instances all) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper standardiesed = new ClassifierWrapper(new EnhancedLinearPerceptron(true,true,false),test, train);
            standardiesed.classifyAllInstances();
            accuracies[0] = standardiesed.getAccuracy();
            ClassifierWrapper offline = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,false),test, train);
            offline.classifyAllInstances();
            accuracies[1] = offline.getAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        
        return accuracies;
    }
    
    public static double [] crossValidationComparison(int folds, Instances all) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper linearPerceptron = new ClassifierWrapper(new LinearPerceptron(),test, train);
            linearPerceptron.classifyAllInstances();
            accuracies[0] = linearPerceptron.getAccuracy();
            ClassifierWrapper crossValidation = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,true),test, train);
            crossValidation.classifyAllInstances();
            accuracies[1] = crossValidation.getAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        
        return accuracies;
    }
    
    public static double [] ensembleComparison(int folds, Instances all) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper ensemble = new ClassifierWrapper(new LinearPerceptronEnsemble(),test, train);
            ensemble.classifyAllInstances();
            accuracies[0] = ensemble.getAccuracy();
            ClassifierWrapper single = new ClassifierWrapper(new EnhancedLinearPerceptron(),test, train);
            single.classifyAllInstances();
            accuracies[1] = single.getAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        
        return accuracies;
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
        
    public static Instances [] splitData(Instances data){
        Instances [] split = new Instances [2];
        

        ArrayList<Integer> indexes = new ArrayList<>();
        for(int i=0; i<data.numInstances(); i++){
            indexes.add(i);
        }
        //shuffle the list so it's randomised
        Collections.shuffle(indexes);
        
        //create a random subset of Instances from the original data 
        int subsetSize = (int)(indexes.size() * 0.5);
        Instances train = new Instances(data, subsetSize);
        for(int i=0; i<subsetSize; i++){
            train.add(data.instance(indexes.get(i)));
        }
        
        Instances test = new Instances(data, subsetSize);
        
        for(int i=subsetSize; i<indexes.size(); i++){
            test.add(data.instance(indexes.get(i)));
        }
        
        split[0] = train;
        split[1] = test;
        
        return split;

    }
}
