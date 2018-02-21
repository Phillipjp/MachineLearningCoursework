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
        Instances data = loadData("/Users/phillipperks/Documents/Year 3/Machine Learning/MachineLearningCoursework/question1.arff");
        //LinearPerceptron lp = new LinearPerceptron();
        //train.setClassIndex(train.numAttributes()-1);
        //lp.buildClassifier(train);
        
        EnhancedLinearPerceptron elp = new EnhancedLinearPerceptron();
        //train.setClassIndex(train.numAttributes()-1);
        elp.buildClassifier(data);
        
        int folds = 30;
        
        // needs to be done for each data set
        double [] learningAlgorithmComparisonAccuracies = learningAlgorithmComparison(folds,data);
        double [] standardisiedDataComparisonAccuracies = standardisiedDataComparison(folds,data);
        double [] crossValidationComparisonAccuracies = crossValidationComparison(folds,data);
        
        
    } 
    
    public static double linearPerceptronAccuracy(Instances train, Instances test, boolean standardise, boolean online, boolean selectModel) throws Exception{
        EnhancedLinearPerceptron linearPerceptron = new EnhancedLinearPerceptron(standardise, online, selectModel);
        linearPerceptron.buildClassifier(train);
        double accuracy = 0;
        int correct = 0;
        for(Instance i: test){
            if(linearPerceptron.classifyInstance(i)==i.value(test.numAttributes()-1)){
                correct++;
            }
        }
        accuracy = correct/test.numInstances();
        return accuracy;
    }
    
    public static double [] learningAlgorithmComparison(int folds, Instances data) throws Exception{
        double [] averages = new double [2];
        
        double [][] learningAlgorithim = new double [2][folds];
        
        
        
        for(int i=0; i<folds; i++){
            Instances [] allData = splitData(data);
            Instances train = allData[0];
            Instances test = allData[1];
            
            double onlineAve = linearPerceptronAccuracy(train, test, false, true, false);
            double offlineAve = linearPerceptronAccuracy(train, test, false, false, false);
            
            learningAlgorithim[0][folds] = onlineAve;
            averages[0] += onlineAve;
            learningAlgorithim[1][folds] = offlineAve;
            averages[1] += offlineAve;
            
        }
        
        averages[0]/=folds;
        averages[1]/=folds;
        
        return averages;
    }
    
    public static double [] standardisiedDataComparison(int folds, Instances data) throws Exception{
        double [] averages = new double [4];
        
        double [][] learningAlgorithim = new double [4][folds];
        
        
        
        for(int i=0; i<folds; i++){
            Instances [] allData = splitData(data);
            Instances train = allData[0];
            Instances test = allData[1];
            
            double onlineAve = linearPerceptronAccuracy(train, test, false, true, false);
            double onlineAveStd = linearPerceptronAccuracy(train, test, true, true, false);
            double offlineAve = linearPerceptronAccuracy(train, test, false, false, false);
            double offlineAveStd = linearPerceptronAccuracy(train, test, true, false, false);
            
            learningAlgorithim[0][folds] = onlineAve;
            learningAlgorithim[1][folds] = onlineAveStd;
            averages[0] += onlineAve;
            averages[1] += onlineAveStd;
            
            learningAlgorithim[2][folds] = offlineAve;
            learningAlgorithim[3][folds] = offlineAveStd;
            averages[2] += offlineAve;
            averages[3] += offlineAveStd;
            
        }
        
        for(int i=0; i<averages.length; i++){
            averages[i]/=folds;
        }
        
        return averages;
    }
    
    public static double [] crossValidationComparison(int folds, Instances data) throws Exception{
        double [] averages = new double [2];
        
        double [][] learningAlgorithim = new double [2][folds];
        
        
        
        for(int i=0; i<folds; i++){
            Instances [] allData = splitData(data);
            Instances train = allData[0];
            Instances test = allData[1];
            
            double cvAve = linearPerceptronAccuracy(train, test, false, true, true);
            double defaultAve = linearPerceptronAccuracy(train, test, true, true, false);
            
            learningAlgorithim[0][folds] = cvAve;
            averages[0] += cvAve;
            learningAlgorithim[1][folds] = defaultAve;
            averages[1] += defaultAve;
            
        }
        
        averages[0]/=folds;
        averages[1]/=folds;
        
        return averages;
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
