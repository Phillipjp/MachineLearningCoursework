/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;
import java.util.Arrays;
import weka.classifiers.*;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
/**
 *
 * @author phillipperks
 */
public class LinearPerceptron implements Classifier {

    double bias;
    int max_iterations;
    int num_instances;
    int num_attributes;
    double [] w;
    
    //defualt constructor
    public LinearPerceptron(){
        super();
        this.bias = 1;
        this.max_iterations = 100;
    }
    
    //constructor that sets the bias
    public LinearPerceptron(double bias){
        super();
        this.bias = bias;
        this.max_iterations = 100;
    }
    
    //constructor that sets the  max iterations
    public LinearPerceptron(int i){
        super();
        this.bias = 1;
        this.max_iterations = i;
    }
    
    //constructor that sets the bias and max iterations
    public LinearPerceptron(double bias, int i){
        super();
        this.bias = bias;
        this.max_iterations = i;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {  
        this.num_instances = data.numInstances();
        this.num_attributes = data.numAttributes()-1;
        for(int i=0; i<num_attributes; i++){
            if(!data.attribute(i).isNumeric()){
                throw new AttributesNotContinuousException("All attribute values"
                        + "need to be continuous. They cannot be descrete.");
            }
        }
        this.w = new double[num_attributes];
        //calcualte weights using the on-line perceptron rule
        this.w = onlinePerceptronRule(data);
    }
    
    protected double [] onlinePerceptronRule(Instances data) throws Exception{
        boolean stop = false;
        int iteration = 0;
        int lastUpdate = -1;
        double [] weights = new double[num_attributes];
        //initialise all weights to 1
        Arrays.fill(weights,1);
        
        do{
            for(int i=0; i<num_instances; i++){
                //if this was the last instance where an update was required stop
                if(i == lastUpdate){
                    stop = true;
                    break;
                }
                //else update weightings
                else{
                    double y;
                    //calculate if the prediction is positive or negative
                    y = calculateY(data.instance(i), weights);
                    //if the prediction is wrong update the weights
                    if(y != data.instance(i).classValue()){
                        lastUpdate = i;
                        double classValue = 1;
                        if(data.instance(i).classValue() == 0){
                            classValue = -1;
                        }
                        if(y == 0){
                            y = -1;
                        }
                        //update weights
                        for(int j=0; j<weights.length; j++){
                            //delta = 0.5 * learning rate * (class value - predicted value) * attribute value
                            double delta = (0.5*bias)*(classValue-y)* data.instance(i).value(j);
                            weights[j] += delta;
                        }
                    }
                } 
                iteration++;
            }
        //stop if stop = true or the maximum number of iterations has been reached
        }while(!stop && iteration < max_iterations);
        return weights;
    }

    protected double calculateY(Instance instnc, double [] weights){
        double y = 0;
        //find if the instance is positive or negative
        for(int j=0; j<weights.length;j++){
            //System.out.println(w[j] + " x " + instnc.value(j));
            y+=weights[j]*instnc.value(j);
        }
        //if the instance is negative set y to -1
        if(y<0){
            y = 0;
        }
        //else set the y to +1
        else{
            y = 1;
        }
        return y;
    }
    
    public void setBias(double bias){
        this.bias=bias;
    }
    
    public void setMaxIterations(int i){
        this.max_iterations = i;
    }
    
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        return calculateY(instnc, w);
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
