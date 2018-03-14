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
         String problem = "acute-inflammation";
//        String problem = "acute-nephritis";
//        String problem = "balloons";
//        String problem = "bank";
//        String problem = "blood";
//        String problem = "breast-cancer";
//        String problem = "breast-cancer-wisc";
//        String problem = "breast-cancer-wisc-diag";
//        String problem = "chess-krvkp";
//        String problem = "congressional-voting";
//        String problem = "conn-bench-sonar-mines-rocks";
//        String problem = "credit-approval";

        Instances all = loadData("\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Machine Learning\\ForML\\" + problem + ".arff");
        all.setClassIndex(all.numAttributes()-1);
        
        int folds = 30;
        
        double [] learningAlgorithm =  learningAlgorithmComparison(30, all, problem);
        System.out.println("Online Accuracy:\t\t" + learningAlgorithm[0]);
        System.out.println("Online Balanced Accuracy:\t" + learningAlgorithm[1]);
        System.out.println("Offline Accuracy:\t\t" + learningAlgorithm[2]);
        System.out.println("Offline Balanced Accuracy:\t" + learningAlgorithm[3]);
        double accuracy = EvaluationMetrics.accuracy("\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Machine Learning\\MachineLearningCoursework\\results\\learningAlgorithmComparison\\acute-inflammation\\offline\\offline", folds);
        double NLL = EvaluationMetrics.NLL("\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Machine Learning\\MachineLearningCoursework\\results\\learningAlgorithmComparison\\acute-inflammation\\offline\\offline", folds);
        System.out.println("NLL:\t" + NLL);
        System.out.println("Statistics Accuracy:\t" + accuracy);
        double balancedAccuracy = EvaluationMetrics.balancedAccuracy("\\\\ueahome4\\stusci3\\xju14zpu\\data\\Documents\\Machine Learning\\MachineLearningCoursework\\results\\learningAlgorithmComparison\\acute-inflammation\\offline\\offline", folds);
        System.out.println("Statistics Balanced Accuracy:\t" + balancedAccuracy);

        
//        double [] elpAVE = new double [2];
//        for (int i = 0; i < 30; i++) {
//            
//
//        
//        Instances [] data = splitData(all);
//        Instances train = data[0];
//        Instances test = data [1];
//        
//        ClassifierWrapper elp = new ClassifierWrapper(new EnhancedLinearPerceptron(true,false,false),test, train);
//        System.out.println("");
//        elp.confusionMatrix();
//        elpAVE[0] += elp.getAccuracy();
//        elpAVE[1] += elp.getBalancedAccuracy();
//        
//        }
//        
//        System.out.println("Accuracy:\t\t" + elpAVE[0]/30);
//        System.out.println("Balanced Accuracy:\t" + elpAVE[1]/30);
    } 
    
    public static double [] learningAlgorithmComparison(int folds, Instances all, String problem) throws Exception{
        double [] accuracies = new double [4];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper online = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,false),test, train);
            online.confusionMatrix();
            online.writeCsvFile("learningAlgorithmComparison\\" +problem+"\\online\\online" + i, "EnhancedLinearPerceptron", problem);
            accuracies[0] += online.getAccuracy();
            accuracies[1] += online.getBalancedAccuracy();
            ClassifierWrapper offline = new ClassifierWrapper(new EnhancedLinearPerceptron(false,false,false),test, train);
            offline.writeCsvFile("learningAlgorithmComparison\\" +problem+"\\offline\\offline" + i, "EnhancedLinearPerceptron", problem);
            offline.confusionMatrix();
            accuracies[2] += offline.getAccuracy();
            accuracies[3] += offline.getBalancedAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        accuracies[2]/=folds;
        accuracies[3]/=folds;
        
        return accuracies;
    }
    
    public static double [] onlineStandardisedComparison(int folds, Instances all, String problem) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper standardiesed = new ClassifierWrapper(new EnhancedLinearPerceptron(true,true,false),test, train);
            standardiesed.confusionMatrix();
            standardiesed.writeCsvFile("onlineStandardisedComparison\\" +problem+"\\onlineStandardised" + i, "EnhancedLinearPerceptron", problem);
            accuracies[0] += standardiesed.getAccuracy();
            ClassifierWrapper online = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,false),test, train);
            online.writeCsvFile("onlineStandardisedComparison\\" +problem+"\\online" + i, "EnhancedLinearPerceptron", problem);
            online.confusionMatrix();
            accuracies[1] += online.getAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        
        return accuracies;
    }
    
    public static double [] offlineStandardisedComparison(int folds, Instances all, String problem) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper standardiesed = new ClassifierWrapper(new EnhancedLinearPerceptron(true,true,false),test, train);
            standardiesed.confusionMatrix();
            standardiesed.writeCsvFile("offlineStandardisedComparison\\" +problem+"\\offlineStandardised" + i, "EnhancedLinearPerceptron", problem);
            accuracies[0] += standardiesed.getAccuracy();
            ClassifierWrapper offline = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,false),test, train);
            offline.writeCsvFile("offlineStandardisedComparison\\" +problem+"\\offline" + i, "EnhancedLinearPerceptron", problem);
            offline.confusionMatrix();
            accuracies[1] += offline.getAccuracy();
        }
        
        accuracies[0]/=folds;
        accuracies[1]/=folds;
        
        return accuracies;
    }
    
    public static double [] crossValidationComparison(int folds, Instances all, String problem) throws Exception{
        double [] accuracies = new double [2];
        for (int i = 0; i < folds; i++) {
            Instances [] data  = splitData(all);
            Instances train = data[0];
            Instances test = data[1];
            
            ClassifierWrapper linearPerceptron = new ClassifierWrapper(new LinearPerceptron(),test, train);
            linearPerceptron.confusionMatrix();
            linearPerceptron.writeCsvFile("crossValidationComparison\\" +problem+"\\linearPerceptron" + i, "LinearPerceptron", problem);
            accuracies[0] += linearPerceptron.getAccuracy();
            ClassifierWrapper crossValidation = new ClassifierWrapper(new EnhancedLinearPerceptron(false,true,true),test, train);
            crossValidation.writeCsvFile("crossValidationComparison\\" +problem+"\\crossValidation" + i, "EnhancedLinearPerceptron", problem);
            crossValidation.confusionMatrix();
            accuracies[1] += crossValidation.getAccuracy();
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
            ensemble.confusionMatrix();
            accuracies[0] += ensemble.getAccuracy();
            ClassifierWrapper single = new ClassifierWrapper(new EnhancedLinearPerceptron(),test, train);
            single.confusionMatrix();
            accuracies[1] += single.getAccuracy();
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
