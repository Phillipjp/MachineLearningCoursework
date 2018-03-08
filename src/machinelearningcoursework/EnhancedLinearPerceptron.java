/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;
import java.util.Arrays;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
/**
 *
 * @author phillipperks
 */
public class EnhancedLinearPerceptron extends LinearPerceptron {

    
    double [] w;
    boolean selectModel;
    boolean onlineModel;
    boolean standardise;
    double [] means;
    double [] standard_deviations;
    
    //enum for the two types of model that can be used
    enum Model{online, offline};
    
    //defualt constructor
    public EnhancedLinearPerceptron(){
        super();
        this.bias = 1;
        this.max_iterations = 100;
        this.selectModel = false;
        this.onlineModel = true;
        this.standardise = true;
    }
    
    public EnhancedLinearPerceptron(boolean standardise){
        super();
        this.bias = 1;
        this.max_iterations = 100;
        this.selectModel = false;
        this.onlineModel = true;
        this.standardise = standardise;
    }
    
    public EnhancedLinearPerceptron(boolean standardise, boolean online, boolean selectModel){
        super();
        this.bias = 1;
        this.max_iterations = 100;
        this.selectModel = selectModel;
        this.onlineModel = online;
        this.standardise = standardise;
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
        //standardise attributes
        if(standardise){
            this.means = new double[num_attributes];
            this.standard_deviations = new double[num_attributes];
            data = standardiseAttributes(data);
        }
        //set what learing algorithim to use via cross validation
        if(selectModel){
            Model model = modelSelection(data);
            if(model.equals(Model.online)){
                this.w = onlinePerceptronRule(data);
            }
            else{
                this.w = offlinePerceptronRule(data); 
            }
        }
        else{
            if(onlineModel){
                this.w = onlinePerceptronRule(data);
            }
            else{
               this.w = offlinePerceptronRule(data); 
            }
        }
    }
    
    private Instances standardiseAttributes(Instances data){
  
        //calculate means of each attribute
        for(Instance i: data){
            for(int j=0; j<num_attributes; j++){
                means[j]+=i.value(j);
            }
        }
        for(int j=0; j<num_attributes; j++){
                means[j]/=num_instances;
        }
        
        //calculate standard deviate for each attribute
        for(Instance i: data){
            for(int j=0; j<num_attributes; j++){
                standard_deviations[j]+=Math.pow(i.value(j)-means[j],2);
            }
        }
        for(int j=0; j<num_attributes; j++){
                standard_deviations[j]/=num_instances;
                standard_deviations[j] = Math.sqrt(standard_deviations[j]);
        }
        
        //standardise each instance
        for(Instance i: data){
            i = standardiseInstance(i);
        }
        
        return data;
    }
    
    private Instance standardiseInstance(Instance i){
        //go through each attribute and standardise it
        for(int j=0; j<num_attributes; j++){
            i.setValue(j, (i.value(j)-means[j])/standard_deviations[j]);
        }
        return i;
    }
    
    private double [] offlinePerceptronRule(Instances data) throws Exception{
        boolean stop = false;
        int iteration = 0;
        int lastUpdate = -1;
        double [] weights = new double[num_attributes];
        //initialise all weights to 1
        Arrays.fill(weights,1);
        
        do{
            double [] delta = new double [num_attributes];
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
                    //if the prediction is wrong
                    if(y != data.instance(i).classValue()){
                        lastUpdate = i;
                        double classValue = 1;
                        if(data.instance(i).classValue() == 0){
                            classValue = -1;
                        }
                        if(y == 0){
                            y = -1;
                        }
                        //calcualte new deltas for each weight
                        for(int j=0; j<delta.length; j++){
                            //delta[j] += 0.5 * learning rate * (class value - predicted value) * attribute value
                            delta[j] += (0.5*bias)*(classValue-y)* data.instance(i).value(j);
                        }
                    }
                } 
            }
            //update weights
            for(int j=0; j<weights.length; j++){
                weights[j] += delta[j];
            }
            iteration++;
        //stop if stop = true or the maximum number of iterations has been reached
        }while(!stop && iteration < max_iterations);
        
        return weights;
    }
    
    private Model modelSelection(Instances data) throws Exception{
        //count of the number correct classifications
        int online = 0;
        int offline = 0;
        //the model to be used
        Model model;
        //weights for each model
        double [] online_w;
        double [] offline_w;
        //for all instances
        for(int i=0; i<num_instances; i++){
            //use all the data except one instance
            Instances cvdata = data;
            cvdata.delete(i);
            
            //calcualte weights using the on-line perceptron rule
            online_w = onlinePerceptronRule(cvdata);
            //calcualte weights using the off-line perceptron rule
            offline_w = offlinePerceptronRule(cvdata);
            
            //classify the instance that was left out
            double onlineResult = calculateY(data.instance(i), online_w);
            double offlineResult = calculateY(data.instance(i), offline_w);
            
            //if the instance was correctly classified by the on-line perceptron rule
            if(onlineResult == data.instance(i).classValue()){
                online++;
            }
            //if the instance was correctly classified by the off-line perceptron rule
            if(offlineResult == data.instance(i).classValue()){
                offline++;
            }
        }
        //set the model to use the perceptron rule that made the most correct
        //classifications. If there's a tie use the on-line perceptron rule
        if(online>=offline){
            model = Model.online;
        }
        else{
            model = Model.offline;
        }
        
        return model;
    }
    
    public void setSelectModel(boolean flag){
        this.selectModel = flag;
    }
    
    public void setOnlineModel(boolean flag){
        this.onlineModel = flag;
    }
    
    public void setStandardise(boolean flag){
        this.standardise=flag;
    }
    
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        //standardise attributes
        if(standardise){
            instnc = standardiseInstance(instnc);
        }
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
