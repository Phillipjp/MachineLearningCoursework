/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;

import java.util.ArrayList;
import java.util.Collections;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class LinearPerceptronEnsemble implements Classifier{

    private EnhancedLinearPerceptron [] ensemble;
    private int [][]  ensembleData;
    private double proportion;
    private int size;
    
    public LinearPerceptronEnsemble(){
        super();
        this.ensemble = new EnhancedLinearPerceptron [50];
        this.ensembleData = new int [50][];
        this.proportion = 0.5;
    }
    
    public LinearPerceptronEnsemble(int size){
        super();
        this.ensemble = new EnhancedLinearPerceptron [size];
        this.ensembleData = new int [size][];
        this.proportion = 0.5;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        for(int i=0; i< this.ensemble.length; i++){
            this.ensemble[i] = new EnhancedLinearPerceptron();
            this.ensembleData[i] = getSubsetIndexes(data);
            Instances train = getSubsetOfAttributes(data, ensembleData[i]);
            this.ensemble[i].buildClassifier(train);
        }
    }
    
    public void setSize(int size, Instances data) throws Exception{
        int [][] newEnsembleData = new int [size][];
        EnhancedLinearPerceptron [] newEnsemble = new EnhancedLinearPerceptron [size];
        if(size<ensemble.length){
            for (int i = 0; i < ensemble.length; i++) {
                newEnsemble[i] = ensemble [i];
                newEnsembleData[i] = ensembleData[i];
            }
            ensemble = null;
            ensembleData = null;
            ensemble = newEnsemble;
            ensembleData = newEnsembleData;
        }
        else if(size > ensemble.length){
            for (int i = 0; i < ensemble.length; i++) {
                newEnsemble[i] = ensemble [i];
                newEnsembleData[i] = ensembleData[i];
            }
            for(int i = ensemble.length; i<size; i++){
                newEnsemble[i] = new EnhancedLinearPerceptron();
                newEnsembleData[i] = getSubsetIndexes(data);
                Instances train = getSubsetOfAttributes(data, ensembleData[i]);
                newEnsemble[i].buildClassifier(train);
            }
            ensemble = null;
            ensembleData = null;
            ensemble = newEnsemble;
            ensembleData = newEnsembleData;
            
        }
    }
    private Instances getSubsetOfAttributes(Instances data, int [] indexes){
        //create a new Instnces object 
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i = 0; i < indexes.length; i++) {
            attributes.add(data.attribute(i));
        }
        attributes.add(data.classAttribute());
        Instances subset = new Instances(data.relationName(), attributes ,data.numInstances());
        //create a new set of Instanes that only contain specific attributes 
        for(int i=0; i<data.numInstances(); i++){
            Instance sub = new DenseInstance(indexes.length+1);
            double [] inst = data.instance(i).toDoubleArray();
            for (int j = 0; j < indexes.length; j++) {
                sub.setValue(j, inst[j]);
            }
            sub.setValue(indexes.length, data.instance(i).classValue());
            sub.setDataset(subset);
            subset.add(i, sub);
        }
        subset.setClassIndex(subset.numAttributes()-1);
        return subset;
    }
    
    private int [] getSubsetIndexes(Instances data){
        //create a list the size of the number of attributes
        ArrayList<Integer> indexesList = new ArrayList<>();
        for(int i=0; i<data.numAttributes()-1; i++){
            indexesList.add(i);
        }
        //shuffle the list so it's randomised
        Collections.shuffle(indexesList);
        //put the list into an array the size of the prortoion  of attributes to be used
        int subsetSize = (int)(indexesList.size() * proportion);
        int [] indexes = new int [subsetSize];
        for (int i = 0; i < subsetSize; i++) {
            indexes[i] = indexesList.get(i);
        }
        return indexes;
    }
    
    private Instance getClassifyInstanceAttributes(Instance instnc, int [] indexes, int p){
        //get only the attibutes the classifer was trained on
        Instance sub = new DenseInstance(indexes.length+1);
        double [] inst = instnc.toDoubleArray();
        for (int j = 0; j < indexes.length; j++) {
            sub.setValue(j, inst[j]);
        }
        sub.setValue(indexes.length, instnc.classValue());
        return sub;
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double [] votes = new double [2];
        
        for (int i = 0; i < ensemble.length; i++) {
            Instance sub = getClassifyInstanceAttributes(instnc, ensembleData[i], i);
            double prediction = ensemble[i].classifyInstance(sub);
            if(prediction == 0)
                votes[0]++;
            else
                votes[1]++;
        }
        if(votes[0]>votes[1])
            return 0;
        else
            return 1;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        double [] votes = new double [2];
        
        for (int i = 0; i < ensemble.length; i++) {
            double prediction = ensemble[i].classifyInstance(instnc);
            if(prediction == 0)
                votes[0]++;
            else
                votes[1]++;
        }
        
        votes[0]/=(votes[0]+votes[1]);
        votes[1]/=(votes[0]+votes[1]);
        return votes;
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
