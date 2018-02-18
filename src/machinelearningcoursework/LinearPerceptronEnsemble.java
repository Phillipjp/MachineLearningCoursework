/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;

import java.util.ArrayList;
import java.util.Collections;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author phillipperks
 */
public class LinearPerceptronEnsemble implements Classifier{

    LinearPerceptron [] ensemble;
    Instances [] ensembleData;
    double proportion;
    
    public LinearPerceptronEnsemble(){
        super();
        this.ensemble = new LinearPerceptron [50];
        this.ensembleData = new Instances [50];
        this.proportion = 0.5;
    }
    
    public LinearPerceptronEnsemble(int size){
        super();
        this.ensemble = new LinearPerceptron [size];
        this.ensembleData = new Instances [size];
        this.proportion = 0.5;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        for(int i=0; i< this.ensemble.length; i++){
            this.ensemble[i] = new LinearPerceptron();
            this.ensembleData[i] = getSubset(data);
            this.ensemble[i].buildClassifier(ensembleData[i]);
        }
    }
    
    private Instances getSubset(Instances data){
        //create a list the size of the number of instances
        ArrayList<Integer> indexes = new ArrayList<>();
        for(int i=0; i<data.numInstances()-1; i++){
            indexes.add(i);
        }
        //shuffle the list so it's randomised
        Collections.shuffle(indexes);
        
        //create a random subset of Instances from the original data 
        int subsetSize = (int)(indexes.size() * proportion);
        Instances subset = new Instances(data, subsetSize);
        for(int i=0; i<subsetSize; i++){
            subset.set(i, data.instance(indexes.get(i)));
        }
        
        
        return subset;
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
}
