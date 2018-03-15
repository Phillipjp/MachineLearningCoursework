
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author xju14zpu
 */
public class EvaluationMetrics {
    
    public static double NLL(String dataLocation, int folds) throws FileNotFoundException, IOException, Exception{
        double NLL = 0;
        for (int i = 0; i < folds; i++) {
            File inputFile = new File(dataLocation + i + ".csv");
            FileReader reader = new FileReader(inputFile);
            BufferedReader br = new BufferedReader(reader);
            String lineString ;

            br.readLine();
            br.readLine();
            br.readLine();
            while((lineString = br.readLine()) != null){
                String [] lineArr = lineString.split(",");

                if(Double.parseDouble(lineArr[0]) == 0){
                    NLL += Math.log(Double.parseDouble(lineArr[3]))/Math.log(2);
                }
                else if(Double.parseDouble(lineArr[0]) == 1){
                    NLL += Math.log(Double.parseDouble(lineArr[4]))/Math.log(2);
                }
                else{
                    throw new Exception();
                }
            }
        }
        NLL /= folds;
        return NLL;
    }
    
    public static double accuracy(String dataLocation, int folds) throws FileNotFoundException, IOException{
        double accuracy = 0;
        for (int i = 0; i < folds; i++) {
            File inputFile = new File(dataLocation + i + ".csv");
            FileReader reader = new FileReader(inputFile);
            BufferedReader br = new BufferedReader(reader);
            String lineString ;
            br.readLine();
            br.readLine();
            lineString = br.readLine();
            String [] lineArr = lineString.split(",");
            accuracy += Double.parseDouble(lineArr[0]);
        }
        accuracy /= folds;
        return accuracy;
    }
    
    public static double balancedAccuracy(String dataLocation, int folds) throws FileNotFoundException, IOException{
        double balancedAccuracy = 0;
        for (int i = 0; i < folds; i++) {
            File inputFile = new File(dataLocation + i + ".csv");
            FileReader reader = new FileReader(inputFile);
            BufferedReader br = new BufferedReader(reader);
            String lineString ;
            br.readLine();
            br.readLine();
            lineString = br.readLine();
            String [] lineArr = lineString.split(",");
            balancedAccuracy += Double.parseDouble(lineArr[1]);
        }
        balancedAccuracy /= folds;
        return balancedAccuracy;
    }
}
