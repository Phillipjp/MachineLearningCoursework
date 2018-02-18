/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearningcoursework;

/**
 *
 * @author phillipperks
 */
public class AttributesNotContinuousException extends Exception {

    /**
     * Creates a new instance of <code>AttributesNotContinuousException</code>
     * without detail message.
     */
    public AttributesNotContinuousException() {
    }

    /**
     * Constructs an instance of <code>AttributesNotContinuousException</code>
     * with the specified detail message.
     *
     * @param msg the detail message.
     */
    public AttributesNotContinuousException(String msg) {
        super(msg);
    }
}
