package moa.classifiers.trees;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import java.util.*;

import com.yahoo.labs.samoa.instances.Instance;
import moa.AbstractMOAObject;
import moa.classifiers.Regressor;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;

import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.HoeffdingNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NominalAttributeBinaryTest;
import moa.classifiers.core.conditionaltests.NominalAttributeMultiwayTest;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.splitcriteria.SDRSplitCriterion;
import moa.core.MiscUtils;
import moa.core.*;
import moa.options.ClassOption;
import weka.classifiers.trees.ht.Split;
import java.io.Serializable;

public class EFDTR extends EFDT implements Regressor {

    protected Node treeRoot;
    protected double examplesSeen = 0.0;
    protected double sumOfValues = 0.0;
    protected double sumOfSquares = 0.0;
    public int maxID = 0;
    protected int leafNodeCount = 0;
    protected int splitNodeCount = 0;
    protected DoubleVector sumOfAttrValues = new DoubleVector();
    protected DoubleVector sumOfAttrSquares = new DoubleVector();
    public static boolean regressionTree ;
    //private EFDTSplitNodeRegression parent = null;
    protected DoubleVector splitRatioStatistics = new DoubleVector();

    protected int alternateTrees;

    protected int prunedAlternateTrees;

    protected int switchedAlternateTrees;

    //region ================ OPTIONS ================

    public FlagOption learningRatioConstOption = new FlagOption(
            "learningRatioConstPerceptron", 'v', "Keep learning rate constant instead of decaying.");
    public FloatOption learningRatioPerceptronOption = new FloatOption(
            "learningRatioPerceptron", 'w', "Learning ratio to used for training the Perceptrons in the leaves.",
            0.02, 0, 1.00);
    public FloatOption learningRateDecayFactorOption = new FloatOption(
            "learningRatioDecayFactorPerceptron", 'x', "Learning rate decay factor (not used when learning rate is constant).",
            0.001, 0, 1.00);
    public FlagOption meanPredictionNodeOption = new FlagOption(
            "regressionTree", 'k', "Build a regression tree instead of a model tree.");



    //endregion ================ OPTIONS ================


    //region ================ CLASSES ================

    public static class InactiveLearningNodeForRegression extends InactiveLearningNode{
        EFDTRegressionPerceptron learningModel;
        public InactiveLearningNodeForRegression (double[] initialClassObservations,EFDTRegressionPerceptron p) {
            super(initialClassObservations);
            this.learningModel = p ;
        }
        public void learnFromInstance(Instance inst,EFDTR ht)  {

            //**
            // The observed class distribution contains the number of instances seen by the node in the first slot ,
            // the sum of values in the second and the sum of squared values in the third
            // these statistics are useful to calculate the mean and to calculate the variance reduction
            //**

            this.observedClassDistribution.addToValue(0,
                    inst.weight());
            this.observedClassDistribution.addToValue(1,
                    inst.weight() * inst.classValue());
            this.observedClassDistribution.addToValue(2,
                    inst.weight() * inst.classValue() * inst.classValue());

            if (regressionTree==false) {
                learningModel.updatePerceptron(inst);
            }
        }
    }
    public static class ActiveLearningNodeForRegression extends ActiveLearningNode{

        EFDTRegressionPerceptron learningModel;

        public ActiveLearningNodeForRegression(double[] initialClassObservations,EFDTRegressionPerceptron p) {
            super(initialClassObservations);
            this.learningModel=p;
            this.weightSeenAtLastSplitEvaluation = getWeightSeen();
            this.isInitialized = false;
        }

        public void learnFromInstance(Instance inst,EFDTR ht){
            nodeTime++;
            if (this.isInitialized == false) {
                this.attributeObservers = new AutoExpandVector<AttributeClassObserver>(inst.numAttributes());
                this.isInitialized = true;
            }
            //**
            // The observed class distribution contains the number of instances seen by the node in the first slot ,
            // the sum of values in the second and the sum of squared values in the third
            // these statistics are useful to calculate the mean and to calculate the variance reduction
            //**

            this.observedClassDistribution.addToValue(0,
                    inst.weight());
            this.observedClassDistribution.addToValue(1,
                    inst.weight() * inst.classValue());
            this.observedClassDistribution.addToValue(2,
                    inst.weight() * inst.classValue() * inst.classValue());

            if (regressionTree==false)  {
                learningModel.updatePerceptron(inst);
            }


            for (int i = 0; i < inst.numAttributes() - 1; i++) {
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    //we use HoeffdingNominalAttributeClassObserver for nominal attributes and HoeffdingNUmericAttributeClassObserver for numeric attributes
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();

                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeTarget(inst.value(instAttIndex),  inst.classValue());
            }
        }

        public double getWeightSeen() {
            return this.observedClassDistribution.getValue(0);
        }

    }
    public class EFDTRegressionPerceptron implements Serializable{
        private static final long serialVersionUID = 1L;
        protected EFDTR tree;
        // The Perception weights
        protected DoubleVector weightAttribute = new DoubleVector();

        // The number of instances contributing to this model
        protected double instancesSeen = 0;
        protected double sumOfValues;
        protected double sumOfSquares;
        // If the model should be reset or not
        protected boolean reset;

        public EFDTRegressionPerceptron(EFDTRegressionPerceptron original){

            this.instancesSeen = original.instancesSeen;
            weightAttribute = (DoubleVector) original.weightAttribute.copy();
            reset = false;
        }

        public EFDTRegressionPerceptron() {
            reset = true;
        }
        public DoubleVector getWeights() {
            return weightAttribute;
        }
        public void updatePerceptron(Instance inst) {

            // Initialize perceptron if necessary
            if (reset == true) {
                reset = false;
                weightAttribute = new DoubleVector();
                instancesSeen = 0;
                for (int j = 0; j < inst.numAttributes(); j++) { // The last index corresponds to the constant b
                    weightAttribute.setValue(j, 2 * classifierRandom.nextDouble() - 1);
                }
            }

            // Update attribute statistics
            instancesSeen += inst.weight();
            // Update weights
            double learningRatio = 0.0;
            if (learningRatioConstOption.isSet()) {
                learningRatio = learningRatioPerceptronOption.getValue();
            } else {
                learningRatio = learningRatioPerceptronOption.getValue() / (1 + instancesSeen * learningRateDecayFactorOption.getValue());
            }
            sumOfValues += inst.weight() * inst.classValue();
            sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();
            // Loop for compatibility with bagging methods
            for (int i = 0; i < (int) inst.weight(); i++) {
                updateWeights(inst, learningRatio);
            }
        }
        public void updateWeights(Instance inst, double learningRatio) {
            // Compute the normalized instance and the delta
            DoubleVector normalizedInstance = normalizedInstance(inst);
            double normalizedPrediction = prediction(normalizedInstance);
            double normalizedValue = normalizeTargetValue(inst.classValue());
            double delta = normalizedValue - normalizedPrediction;
            normalizedInstance.scaleValues(delta * learningRatio);
            weightAttribute.addValues(normalizedInstance);

        }
        public DoubleVector normalizedInstance(Instance inst) {
            // Normalize Instance
            DoubleVector normalizedInstance = new DoubleVector();

            for (int j = 0; j < inst.numAttributes() - 1; j++) {
                int l ;
                DoubleVector  v = new DoubleVector() ;
                int index =0;

                int instAttIndex = modelAttIndexToInstanceAttIndex(j, inst);
                double mean = sumOfAttrValues.getValue(j) / examplesSeen;
                double sd = computeSD(sumOfAttrSquares.getValue(j), sumOfAttrValues.getValue(j), examplesSeen);
                if (inst.attribute(instAttIndex).isNumeric() && examplesSeen > 1 && sd > 0)
                    normalizedInstance.setValue(j, (inst.value(instAttIndex) - mean) / (3 * sd));
                else
                    normalizedInstance.setValue(j, 0);
            }
            if (examplesSeen > 1)
                normalizedInstance.setValue(inst.numAttributes() - 1, 1.0); // Value to be multiplied with the constant factor
            else
                normalizedInstance.setValue(inst.numAttributes() - 1, 0.0);
            return normalizedInstance;
        }
        public double prediction(DoubleVector instanceValues) {
            return scalarProduct(weightAttribute, instanceValues);
        }

        protected double prediction(Instance inst) {
            DoubleVector normalizedInstance = normalizedInstance(inst);
            double normalizedPrediction = prediction(normalizedInstance);
            return denormalizePrediction(normalizedPrediction);
        }

        private double denormalizePrediction(double normalizedPrediction) {
            double mean = sumOfValues / examplesSeen;
            double sd = computeSD(sumOfSquares, sumOfValues, examplesSeen);
            if (examplesSeen > 1)
                return normalizedPrediction * sd * 3 + mean;
            else
                return 0.0;
        }

        public void getModelDescription(StringBuilder out, int indent) {
            StringUtils.appendIndented(out, indent, getClassNameString() + " =");
            if (getModelContext() != null) {
                for (int j = 0; j < getModelContext().numAttributes() - 1; j++) {
                    if (getModelContext().attribute(j).isNumeric()) {
                        out.append((j == 0 || weightAttribute.getValue(j) < 0) ? " " : " + ");
                        out.append(String.format("%.4f", weightAttribute.getValue(j)));
                        out.append(" * ");
                        out.append(getAttributeNameString(j));
                    }
                }
                out.append(" + " + weightAttribute.getValue((getModelContext().numAttributes() - 1)));
            }
            StringUtils.appendNewline(out);
        }
    }
    public static class MeanLearningNode extends ActiveLearningNodeForRegression{

        public MeanLearningNode(double[] initialClassObservations, EFDTRegressionPerceptron p) {
            super(initialClassObservations, p);
        }

        public double[] getClassVotes(Instance inst,EFDTR ht) {

            double numberOfExamplesSeen = 0;

            double sumOfValues = 0;

            double prediction = 0;

            double V[] = super.getClassVotes(inst, ht);
            sumOfValues = V[1];
            numberOfExamplesSeen = V[0];
            prediction = sumOfValues / numberOfExamplesSeen;
            return new double[]{prediction};
        }
    }
    //Implementation of the perceptron learning node
    public static class PerceptronLearningNode extends ActiveLearningNodeForRegression{

        public PerceptronLearningNode(double[] initialClassObservations, EFDTRegressionPerceptron p) {
            super(initialClassObservations, p);
        }
        public double[] getClassVotes(Instance inst, EFDTR ht) {
            return new double[] {learningModel.prediction(inst)};
        }
    }





    //endregion ================ CLASSES ================


    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.treeRoot != null) {
            FoundNode foundNode   = this.treeRoot.filterInstanceToLeaf(inst,
                    null, -1);
            Node leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }
            return leafNode.getClassVotes(inst, this);

        } else {
            return new double[]{0};        }
    }

    @Override
    public void resetLearningImpl() {
        this.treeRoot = null;
        this.decisionNodeCount = 0;
        this.activeLeafNodeCount = 0;
        this.inactiveLeafNodeCount = 0;
        this.inactiveLeafByteSizeEstimate = 0.0;
        this.activeLeafByteSizeEstimate = 0.0;
        this.byteSizeEstimateOverheadFraction = 1.0;
        this.growthAllowed = true;
        this.examplesSeen = 0;
        this.sumOfValues = 0.0;
        this.sumOfSquares = 0.0;

        if (this.leafpredictionOption.getChosenIndex()>0) {
            this.removePoorAttsOption = null;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        examplesSeen += inst.weight();
        sumOfValues += inst.weight() * inst.classValue();
        sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            int aIndex = modelAttIndexToInstanceAttIndex(i, inst);
            sumOfAttrValues.addToValue(i, inst.weight() * inst.value(aIndex));
            sumOfAttrSquares.addToValue(i, inst.weight() * inst.value(aIndex) * inst.value(aIndex));
        }
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            this.activeLeafNodeCount = 1;
        }
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;

        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
        if (leafNode instanceof EFDTLearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (this.growthAllowed
                    && (learningNode instanceof ActiveLearningNodeForRegression)) {
                ActiveLearningNodeForRegression activeLearningNode = (ActiveLearningNodeForRegression) learningNode;
                double weightSeen = activeLearningNode.getWeightSeen();
                if (weightSeen
                        - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
                    attemptToSplit(activeLearningNode, foundNode.parent,
                            foundNode.parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }

    }

    protected void attemptToSplit(ActiveLearningNodeForRegression node,SplitNode parent,int parentIndex) {
        if (!node.observedClassDistributionIsPure()) {
            // Set the split criterion to use to the SDR split criterion as described by Ikonomovska et al.
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            // Using this criterion, find the best split per attribute and rank the results
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);
            // Declare a variable to determine if any of the splits should be performed
            boolean shouldSplit = false;
            for (int i = 0; i < bestSplitSuggestions.length; i++){

                if (bestSplitSuggestions[i].splitTest != null){
                    if (!node.getVarianceRatiosum().containsKey((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0])))
                    {
                        node.getVarianceRatiosum().put((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0]), 0.0);
                    }
                    double currentSum = node.getVarianceRatiosum().get((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0]));
                    node.getVarianceRatiosum().put((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0]), currentSum + bestSplitSuggestions[i].merit);
                }

                else { // handle the null attribute
                    double currentSum = node.getVarianceRatiosum().get(-1); // null split
                    node.getVarianceRatiosum().put(-1, currentSum + bestSplitSuggestions[i].merit);
                }

            }
            // If only one split was returned, use it
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            }
            // Otherwise, consider which of the splits proposed may be worth trying

            else {
                // Determine the hoeffding bound value, used to select how many instances should be used to make a test decision
                // to feel reasonably confident that the test chosen by this sample is the same as what would be chosen using infinite examples
                double hoeffdingBound = computeHoeffdingBound(1,
                        this.splitConfidenceOption.getValue(), node.getWeightSeen());

                // Determine the top two ranked splitting suggestions

                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];

                // splitRatioStatistics.addToValue(0,1);
                //  splitRatioStatistics.addToValue(1,secondBestSuggestion.merit / bestSuggestion.merit);
                //  if ((((splitRatioStatistics.getValue(1)/splitRatioStatistics.getValue(0)) + hoeffdingBound)  < 1) || (hoeffdingBound < this.tieThresholdOption.getValue())) {


                if ((  secondBestSuggestion.merit/bestSuggestion.merit < 1 - hoeffdingBound)
                        || (hoeffdingBound < this.tieThresholdOption.getValue())) {
                    shouldSplit = true;
                }
                if ((this.removePoorAttsOption != null)
                        && this.removePoorAttsOption.isSet()) {
                    Set<Integer> poorAtts = new HashSet<Integer>();
                    // scan 1 - add any poor to set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
                                    poorAtts.add(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    // scan 2 - remove good ones from set
                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
                        if (bestSplitSuggestions[i].splitTest != null) {
                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
                            if (splitAtts.length == 1) {
                                if (bestSuggestion.merit
                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
                                    poorAtts.remove(new Integer(splitAtts[0]));
                                }
                            }
                        }
                    }
                    for (int poorAtt : poorAtts) {
                        node.disableAttribute(poorAtt);
                    }
                }
            }

            if (shouldSplit) {
                splitCount++;
                AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                if (splitDecision.splitTest == null) {
                    // preprune - null wins
                    deactivateLearningNode(node, parent, parentIndex);
                }
                else {
                    SplitNode newSplit = newSplitNode(splitDecision.splitTest,
                            node.getObservedClassDistribution(),splitDecision.numSplits() );
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i),new EFDTRegressionPerceptron((EFDTRegressionPerceptron) node.learningModel));
                        newSplit.setChild(i, newChild);
                    }
                    this.activeLeafNodeCount--;
                    this.decisionNodeCount++;
                    this.activeLeafNodeCount += splitDecision.numSplits();
                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }
                }

                // manage memory
                enforceTrackerLimit();
            }

        }
    }





    protected LearningNode newLearningNode(double[]initialClassObservations , EFDTRegressionPerceptron p) {

        if (meanPredictionNodeOption.isSet())
        {
            regressionTree=true ;
            return new MeanLearningNode(initialClassObservations, p);
        }
        regressionTree=false ;
        return new PerceptronLearningNode(initialClassObservations, p);
    }

    protected LearningNode newLearningNode() {
        EFDTRegressionPerceptron p = new EFDTRegressionPerceptron();
        return newLearningNode(new double[0],p);
    }
    @Override
    protected Measurement[] getModelMeasurementsImpl() {

        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    public double computeSD(double squaredVal, double val, double size) {
        if (size > 1)
            return Math.sqrt((squaredVal - ((val * val) / size)) / size);
        else
            return 0.0;
    }
    public double scalarProduct(DoubleVector u, DoubleVector v) {
        double ret = 0.0;
        for (int i = 0; i < Math.max(u.numValues(), v.numValues()); i++) {
            ret += u.getValue(i) * v.getValue(i);
        }
        return ret;
    }
    public double normalizeTargetValue(double value) {
        if (examplesSeen > 1) {
            double sd = Math.sqrt((sumOfSquares - ((sumOfValues * sumOfValues)/examplesSeen))/examplesSeen);
            double average = sumOfValues / examplesSeen;
            if (sd > 0 && examplesSeen > 1)
                return (value - average) / (3 * sd);
            else
                return 0.0;
        }
        return 0.0;
    }

    protected void deactivateLearningNode(ActiveLearningNode toDeactivate,
                                          SplitNode parent, int parentBranch) {

    }
    private int argmax(double[] array){

        double max = array[0];
        int maxarg = 0;

        for (int i = 1; i < array.length; i++){

            if(array[i] > max){
                max = array[i];
                maxarg = i;
            }
        }
        return maxarg;
    }
    @Override
    public void enforceTrackerLimit() {

    }
}
