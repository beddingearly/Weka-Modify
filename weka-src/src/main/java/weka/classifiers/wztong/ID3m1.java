package weka.classifiers.wztong;

/*
 *    ID3m1.java
 *    Copyright (C) 2018 Zitong Wang
 *
 */


import weka.classifiers.*;
import weka.core.*;

/**
 * Implement jlx_ID3 classifier.
 */
public class ID3m1 extends Classifier {

    /** The node's successors.
     * 保存决策树节点的数组 */
    private ID3m1[] m_Successors;

    /** Attribute used for splitting.
     * 分裂属性*/
    private Attribute m_Attribute;

    /** The instances of the leaf node.
     * */
    private Instances m_Instances;

    /**
     * Builds jlx_ID3 decision tree classifier.
     *
     * @param data the training data
     * @exception Exception if classifier can't be built successfully
     */
    public void buildClassifier(Instances data) throws Exception {

        makeTree(data);
    }

    public ID3m1[] getSuccessors(){
        return m_Successors;
    }

    public Attribute getAttribute(){
        return m_Attribute;
    }

    public Instances getInstances(){
        return m_Instances;
    }

    /**
     * Method building jlx_ID3 tree using information gain measure
     * need to modify
     * @param data the training data
     * @exception Exception if decision tree can't be built successfully
     */
    private void makeTree(Instances data) throws Exception {
        // Check if no instances have reached this node
        if (data.numInstances() == 0) {
            m_Attribute = null;
            m_Instances=new Instances(data);
            return;
        }
        // Compute attribute with maximum split value.
        double impurityReduce = 0.0;
        double maxValue=0;
        int maxIndex=-1;
        for (int i = 0; i < data.numAttributes(); i++){
            // 不访问到最后一列
            if(i == data.classIndex()) continue;
            // 计算每一列的信息熵
            impurityReduce = computeInfoGainDegree(data, data.attribute(i));
            if (impurityReduce > maxValue){
                maxValue = impurityReduce;
                maxIndex = i;
            }
        }
//        System.out.print("maxIndex: ");
//        System.out.println(maxIndex);
//        System.out.print("maxValue: ");
//        System.out.println(maxValue);

        // Make leaf if information gain is zero, otherwise create successors.
        if(Utils.eq(maxValue, 0)){
            m_Attribute = null;
            m_Instances=new Instances(data);
            return;
        }
        else {
            // 返回最大的索引
            m_Attribute = data.attribute(maxIndex);
            Instances[] splitData = splitData(data, m_Attribute);
            m_Successors = new ID3m1[m_Attribute.numValues()];
            for (int j = 0; j < m_Attribute.numValues(); j++) {
                m_Successors[j] = new ID3m1();
                m_Successors[j].makeTree(splitData[j]);
            }
        }
//        System.out.println("m_Instance: " + getInstances());
//        System.out.println("m_attribute: " + getAttribute());
//        System.out.println(getSuccessors());
    }

    /**
     * Splits a dataset according to the values of a nominal attribute.
     * Split into two heaps based on class
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    private Instances[] splitData(Instances data, Attribute att) {

        int numAttValues=att.numValues();
        Instances[] splitData = new Instances[numAttValues];
        for (int j = 0; j < numAttValues; j++) {
            splitData[j] = new Instances(data,0);
        }
        int numInstances=data.numInstances();
        for (int i=0;i<numInstances;i++){
            int attVal=(int)data.instance(i).value(att);
            splitData[attVal].add(data.instance(i));
        }
//        System.out.print("splitData: ");
//        System.out.println(Arrays.toString(splitData));
        return splitData;
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     */
    private double computeEntropyReduce(Instances data, Attribute att) throws Exception {

        double entropyReduce = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                entropyReduce-=((double)splitData[j].numInstances()/(double) data.numInstances())*computeEntropy(splitData[j]);
            }
        }

        // add a ad
        return entropyReduce;
    }

    /**
     * Computes the entropy of a dataset.
     * 计算信息熵
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data's class distribution
     */
    private double computeEntropy(Instances data) throws Exception {
        int numClasses=data.numClasses();

        int numInstances=data.numInstances();
        double[] classCounts=new double[numClasses];
        for (int i=0;i<numInstances;i++){
            int classVal=(int)data.instance(i).classValue();
            classCounts[classVal]++;
        }
        for (int i=0;i<numClasses;i++){
            classCounts[i]/=numInstances;
        }
        double Entropy=0;
        for (int i=0;i<numClasses;i++){
            Entropy-=classCounts[i]*log2(classCounts[i],1);
        }
        return Entropy;
    }

    /**
     * compute the logarithm whose base is 2.
     * 换底公式
     * @param /args x,y are numerator and denominator of the fraction.
     * @return the natual logarithm of this fraction.
     */
    private double log2(double x,double y){

        if(x<1e-6||y<1e-6)
            return 0.0;
        else
            return Math.log(x/y)/Math.log(2);
    }

    /**
     * Computes class distribution for instance using decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    public double[] distributionForInstance(Instance instance) throws Exception{

        if (m_Attribute == null) {
            return computeDistribution(m_Instances);
        }
        else {
            return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
        }
    }

    /**
     * Compute the distribution.
     *
     * @param data the training data
     * @exception Exception if classifier can't be built successfully
     */
    private double[] computeDistribution(Instances data) throws Exception {
        int numClasses=data.numClasses();
        double[] probs=new double[numClasses];
        // 用于统计每个class的个数
        double[] classCounts=new double[numClasses];
        int numInstances=data.numInstances();
        for (int i=0;i<numInstances;i++){
            int classVal=(int)data.instance(i).classValue();
//            0 1
//            System.out.print("classVal: ");
//            System.out.println(classVal);
            classCounts[classVal] ++;
        }
        // System.out.println(Arrays.toString(classCounts));
        //System.out.println("classCounts");
//        for (double a : classCounts)
//            System.out.println(a);

        for (int i=0;i<numClasses;i++){
            probs[i]=(classCounts[i]+1.0)/(numInstances+numClasses);
        }
        Utils.normalize(probs);
        //System.out.println("Probs");
//        for(double a:probs)
//            System.out.println(a);
        return probs;
    }

    private double computeInfoGainDegree(Instances data, Attribute att) throws Exception{
        double infoGain = computeEntropyReduce(data, att);
        Instances[] splitData = splitData(data, att);
        double IVa = 0.0;
        for(int j = 0; j < att.numValues(); j++){
            IVa -= ((double)splitData[j].numInstances()/(double) data.numInstances())*log2((double)splitData[j].numInstances(), (double) data.numInstances());
        }

        return infoGain / IVa;
    }



    /**
     * Main method.
     *
     * @param args the options for the classifier
     */
    public static void main(String[] args) {

        try {
            System.out.println(Evaluation.evaluateModel(new ID3m1(), args));
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

}

