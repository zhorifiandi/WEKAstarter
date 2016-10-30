/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;

/**
 *
 * @author user-ari
 */
public class WEKA {


    /**
     * @param args the command line arguments
     */
    
       
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("C:\\Users\\user-ari\\Documents\\ITB\\IF\\ai\\WEKA\\src\\weka\\iris.arff"));
        
        Instances train = new Instances (breader);
        train.setClassIndex(train.numAttributes() -1);
        
        breader.close();
        
        NaiveBayes nB = new NaiveBayes();
        nB.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(nB, train, 10, new Random(1));
        System.out.println(eval.toSummaryString("\nResults\n=======\n",true));
        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
        
    }
    
}
