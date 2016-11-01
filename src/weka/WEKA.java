/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.filters.Filter;
import weka.filters.supervised.attribute.*;

/**
 *
 * @author user-ari
 */
public class WEKA {
    
    void saveModel(Classifier C, String namaFile) throws Exception {
        //SAVE 
         // serialize model
                ObjectOutputStream oos = new ObjectOutputStream(
                                           new FileOutputStream(namaFile));
                oos.writeObject(C);
                oos.flush();
                oos.close();
    }

    public static void main(String[] args) throws Exception {
        // IMPORT file *.arff
        WEKA w = new WEKA();
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("src\\weka\\iris.arff"));
        
        Instances inputTrain = new Instances (breader);
        inputTrain.setClassIndex(inputTrain.numAttributes() -1);
        
        breader.close();
        
    
        //FILTER
        Discretize filter = new Discretize();
        filter.setInputFormat(inputTrain);
        Instances outputTrain = Filter.useFilter(inputTrain,filter);
        
        //ALGORITMA YANG DIGUNAKAN
        NaiveBayes nB = new NaiveBayes();
        nB.buildClassifier(outputTrain);
        
        
        Evaluation eval = new Evaluation(outputTrain);
       
        
        //Pilihan SKEMA
        boolean validasi = false;
        do {
            Scanner scan = new Scanner(System.in);
            System.out.println("\n\n=================\n==== OPTION ====");
            System.out.println("1. Full Training Scheme");
            System.out.println("2. 10 Fold Validation Scheme");
            System.out.println("3. Load");
            System.out.println("4. Create new instance");
            System.out.println("5. Exit");
            System.out.print("Enter your option (1/2/3/4) : ");
            int pilihan = scan.nextInt();

            switch (pilihan) {
                case 1:
                    {
                        eval.evaluateModel(nB,outputTrain);
                        //OUTPUT
                        System.out.println(eval.toSummaryString("\nResults\n======================\n",true));
                        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                        System.out.println("\nDo you want to save this model(1/0)? ");
                        int c = scan.nextInt();
                        if (c == 1 ){
                            System.out.print("Please enter your file name (*.model) : ");
                            String infile = scan.next();
                            w.saveModel(nB,infile);
                        }
                        else {
                            System.out.print("Model not saved.");
                        }       break;
                    }
                case 2:
                    {
                        eval.crossValidateModel(nB, outputTrain, 10, new Random(1));
                        //OUTPUT
                        System.out.println(eval.toSummaryString("\nResults\n======================\n",true));
                        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                        System.out.println("\nDo you want to save this model(1/0)? ");
                        int c = scan.nextInt();
                        if (c == 1 ){
                            System.out.print("Please enter your file name (*.model) : ");
                            String infile = scan.next();
                            w.saveModel(nB,infile);
                        }
                        else {
                            System.out.print("Model not saved.");
                        }       break;
                    }
                case 3:
                    //LOAD
                    // deserialize model
                    System.out.print("Please enter the file name : ");
                    String namaFile = scan.next();
                    Classifier cls = (Classifier) weka.core.SerializationHelper.read(namaFile);
                    eval.crossValidateModel(cls, outputTrain, 10, new Random(1));
                    System.out.println(eval.toSummaryString("\nResults\n======================\n",true));
                    System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                    break;
                case 4:
                    DenseInstance buffer = new DenseInstance(inputTrain.firstInstance());
                    buffer.setDataset(inputTrain);
                    buffer.setMissing(inputTrain.classIndex());
                    for (int i = 0; i < inputTrain.classIndex(); i++){
                        System.out.print("Enter the value for " + buffer.attribute(i).name() + ": ");
                        double val = scan.nextDouble();
                        buffer.setValue(i, val);
                    }
                    double res = nB.classifyInstance(buffer);
                    buffer.setValue(inputTrain.classIndex(), res);
                    inputTrain.add(buffer);
                    System.out.println(inputTrain);
                    break;
                case 5:
                    validasi = true;
                    break;
                default:
                    System.out.println("Wrong input!");
                    break;
            }
        }
        while (!validasi);
        
        
        
        
        
    }
    
}
