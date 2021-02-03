package moa.classifiers.trees;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import java.util.*;


import com.yahoo.labs.samoa.instances.Instance;
import moa.AbstractMOAObject;
import moa.classifiers.Regressor;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.HoeffdingNumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NominalAttributeBinaryTest;
import moa.classifiers.core.conditionaltests.NominalAttributeMultiwayTest;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.core.AttributeSplitSuggestion;


import moa.core.*;
//import weka.classifiers.trees.ht.Split;

import java.io.Serializable;

public class adaEFDTRegression extends VFDT implements Regressor {

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
    private EFDTRegressionSplitNode parent = null;
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

    //region ================ interface ================
    public interface EFDTRegressionNode {

        boolean isRoot();
        public void filterInstanceToLeaves(Instance inst, SplitNode mygressionparent, int parentBranch, List<FoundNode> foundNodes, boolean updateSplitterCounts);

        void setRoot(boolean isRoot);

        void learnFromInstance(Instance inst, adaEFDTRegression ht,EFDTRegressionSplitNode  parent, int parentBranch);

        void setParent(EFDTRegressionSplitNode parent);
        EFDTRegressionSplitNode getParent();
        // Change for adwin
        //public boolean getErrorChange();
        public int numberLeaves();

        public double getErrorEstimation();

        public double getErrorWidth();

        public boolean isNullError();
        public void killTreeChilds(adaEFDTRegression ht);


    }
    //region ================ CLASSES ================


    /*public static class FoundNode {

        public Node node;s

        public SplitNode parent;

        public int parentBranch;

        public FoundNode(Node node, SplitNode parent, int parentBranch) {
            this.node = node;
            this.parent = parent;
            this.parentBranch = parentBranch;
        }
    }*/
    public static class adaInactiveLearningNodeForRegression extends InactiveLearningNode{
        EFDTRegressionPerceptron learningModel;
        public adaInactiveLearningNodeForRegression(double[] initialClassObservations,EFDTRegressionPerceptron p) {
            super(initialClassObservations);
            this.learningModel = p ;
        }
        public void learnFromInstance(Instance inst,adaEFDTRegression ht)  {

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

        public void learnFromInstance(Instance inst,adaEFDTRegression ht){
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
        protected adaEFDTRegression tree;
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
    //Implementation of the mean learning node
    public static class MeanLearningNode extends ActiveLearningNodeForRegression{

        public MeanLearningNode(double[] initialClassObservations, EFDTRegressionPerceptron p) {
            super(initialClassObservations, p);
        }

        public double[] getClassVotes(Instance inst,adaEFDTRegression ht) {

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
        public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
            return new double[] {learningModel.prediction(inst)};
        }
    }
    public class EFDTRegressionLearningNode extends PerceptronLearningNode implements EFDTRegressionNode{

        private EFDTRegressionSplitNode parent = null;
        protected ADWIN estimationErrorWeight;
        protected Random classifierRandom;
        public boolean ErrorChange = false;
        protected int randomSeed = 1;


        public EFDTRegressionLearningNode(double[] initialClassObservations, EFDTRegressionPerceptron p) {

            super(initialClassObservations, p);
            this.classifierRandom = new Random(this.randomSeed); }

        @Override
        public boolean isRoot() {
            return false;
        }

        @Override
        public void filterInstanceToLeaves(Instance inst, SplitNode splitparent, int parentBranch, List<FoundNode> foundNodes, boolean updateSplitterCounts) {
            foundNodes.add(new FoundNode(this, splitparent, parentBranch));

        }

        @Override
        public void setRoot(boolean isRoot) {

        }

        public void learnFromInstance(Instance inst, VFDT ht) {
            super.learnFromInstance(inst, ht);

        }

        @Override
        public void learnFromInstance(Instance inst, adaEFDTRegression ht,EFDTRegressionSplitNode  parent, int parentBranch) {

            learnFromInstance(inst, ht);

            if (ht.growthAllowed
                    && (this instanceof ActiveLearningNode)) {
                ActiveLearningNode activeLearningNode = this;
                double weightSeen = activeLearningNode.getWeightSeen();
                if (activeLearningNode.nodeTime % ht.gracePeriodOption.getValue() == 0) {
                    attemptToSplit(activeLearningNode, parent,
                            parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }
        public double[] getClassVotes(Instance inst, VFDT ht) {
            double[] dist;
            boolean predictionOption = ((adaEFDTRegression) ht).meanPredictionNodeOption.isSet();
            if (predictionOption == true) { //MeanClass
                dist = this.observedClassDistribution.getArrayCopy();
                double prediction = this.observedClassDistribution.getValue(1)/this.observedClassDistribution.getValue(0);
                //System.out.println(prediction);
                return new double[] {prediction};

            } else { //Perceptron
                return new double[] {this.learningModel.prediction(inst)};

            }


        }

        @Override
        public void setParent(EFDTRegressionSplitNode parent) {
            this.parent = parent;

        }

        @Override
        public EFDTRegressionSplitNode getParent() {
            return this.parent;
        }

        @Override
        public int numberLeaves() {
            return 1;
        }

        @Override
        public double getErrorEstimation() {
            if (this.estimationErrorWeight != null) {
                return this.estimationErrorWeight.getEstimation();
            } else {
                return 0;
            }
        }

        @Override
        public double getErrorWidth() {
            return this.estimationErrorWeight.getWidth();
        }

        @Override
        public boolean isNullError() {
            return (this.estimationErrorWeight == null);        }

        @Override
        public void killTreeChilds(adaEFDTRegression ht) {

        }
    }

    public class EFDTRegressionSplitNode extends SplitNode implements EFDTRegressionNode {
        protected ADWIN estimationErrorWeight;
        //public boolean isAlternateTree = false;
        protected Node alternateTree;
        protected int randomSeed = 1;

        protected Random classifierRandom;
        public boolean ErrorChange = false;
        protected AutoExpandVector<HoeffdingNumericAttributeClassObserver> attributeObservers = new AutoExpandVector<>();

        public EFDTRegressionSplitNode(InstanceConditionalTest splitTest, double[] classObservations, int size) {
            super(splitTest, classObservations, size);
            this.classifierRandom = new Random(this.randomSeed);
        }
        public EFDTRegressionSplitNode (InstanceConditionalTest splitTest, double[] classObservations) {
            super(splitTest, classObservations);
            this.classifierRandom = new Random(this.randomSeed);
        }

        @Override
        public void getDescription(StringBuilder sb, int indent) {

        }

        @Override
        public boolean isRoot() {
            return false;
        }

        @Override
        public void filterInstanceToLeaves(Instance inst, SplitNode myparent, int parentBranch, List<FoundNode> foundNodes, boolean updateSplitterCounts) {
            if (updateSplitterCounts) {
                this.observedClassDistribution.addToValue(0, inst.weight());
                this.observedClassDistribution.addToValue(1, inst.weight()*inst.classValue());
                this.observedClassDistribution.addToValue(2, inst.weight()*inst.classValue()*inst.classValue());

            }
            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                Node child = getChild(childIndex);
                if (child != null) {
                    ((EFDTRegressionNode) child).filterInstanceToLeaves(inst, this, childIndex,
                            foundNodes, updateSplitterCounts);
                } else {
                    foundNodes.add(new FoundNode(null, this, childIndex));
                }
            }
            if (this.alternateTree != null) {
                ((EFDTRegressionNode) this.alternateTree).filterInstanceToLeaves(inst, this, -999,
                        foundNodes, updateSplitterCounts);
            }
        }


        @Override
        public void setRoot(boolean isRoot) {

        }
        public void killSubtree(adaEFDTRegression ht) {
            for (Node child : this.children) {
                if (child != null) {

                    //Recursive delete of SplitNodes
                    if (child instanceof SplitNode) {
                        ((EFDTRegressionSplitNode) child).killSubtree(ht);
                    }
                    else if (child instanceof ActiveLearningNode) {
                        child = null;
                        ht.activeLeafNodeCount--;
                    }
                    else if (child instanceof InactiveLearningNode) {
                        child = null;
                        ht.inactiveLeafNodeCount--;
                    }
                    else{

                    }
                }
            }
        }

        @Override
        public void learnFromInstance(Instance inst, adaEFDTRegression ht, EFDTRegressionSplitNode parent, int parentBranch) {
            nodeTime++;
            double trueValue =  inst.classValue();
            //New option vore
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            Instance weightedInst = (Instance) inst.copy();
            if (k > 0) {
                //weightedInst.setWeight(inst.weight() * k);
            }
            //Compute ClassPrediction using filterInstanceToLeaf
            //int ClassPrediction = Utils.maxIndex(filterInstanceToLeaf(inst, null, -1).node.getClassVotes(inst, ht));
            double targetPrediction = 0;
            double absError=0;
            if (filterInstanceToLeaf(inst, parent, parentBranch).node != null) {
                double[] distribution = (filterInstanceToLeaf(inst, parent, parentBranch).node.getClassVotes(inst, ht));
                targetPrediction = distribution[0];
                //System.out.println(targetPrediction);
                absError = Math.abs((targetPrediction-trueValue)/ht.examplesSeen);

            }
            //boolean blCorrect = (trueClass == ClassPrediction);

            if (this.estimationErrorWeight == null) {
                this.estimationErrorWeight = new ADWIN();
            }
            double oldError = this.getErrorEstimation();
            this.ErrorChange = this.estimationErrorWeight.setInput(absError);
            if (this.ErrorChange == true && oldError > this.getErrorEstimation()) {
                //if error is decreasing, don't do anything
                this.ErrorChange = false;
            }
            // Check condition to build a new alternate tree
            //if (this.isAlternateTree == false) {
            if ((this.ErrorChange == true) && (this.alternateTree == null)) {
                //Start a new alternative tree : learning node
                this.alternateTree = ht.newLearningNode();


                //this.alternateTree.isAlternateTree = true;
                ht.alternateTrees++;
            }
            // Check condition to replace tree
            else if (this.alternateTree != null && ((EFDTRegressionNode) this.alternateTree).isNullError() == false) {
                if (this.getErrorWidth() > 300 && ((EFDTRegressionNode) this.alternateTree).getErrorWidth() > 300) {
                    double oldErrorRate = this.getErrorEstimation();
                    double altErrorRate = ((EFDTRegressionNode) this.alternateTree).getErrorEstimation();
                    double fDelta = .05;
                    //if (gNumAlts>0k) fDelta=fDelta/gNumAlts;
                    double fN = 1.0 / ((double) ((EFDTRegressionNode) this.alternateTree).getErrorWidth()) + 1.0 / ((double) this.getErrorWidth());
                    double Bound = (double) Math.sqrt((double) 2.0 * oldErrorRate * (1.0 - oldErrorRate) * Math.log(2.0 / fDelta) * fN);
                    if (Bound < oldErrorRate - altErrorRate) {
                        // Switch alternate tree
                        ht.activeLeafNodeCount -= this.numberLeaves();
                        ht.activeLeafNodeCount += ((EFDTRegressionNode) this.alternateTree).numberLeaves();
                        killTreeChilds(ht);
                        if (parent != null) {
                            parent.setChild(parentBranch, this.alternateTree);
                            //((AdaSplitNode) parent.getChild(parentBranch)).alternateTree = null;
                        } else {
                            // Switch root tree
                            ht.treeRoot = ((EFDTRegressionSplitNode) ht.treeRoot).alternateTree;
                        }
                        ht.switchedAlternateTrees++;
                    } else if (Bound < altErrorRate - oldErrorRate) {
                        // Erase alternate tree
                        if (this.alternateTree instanceof ActiveLearningNodeForRegression) {
                            this.alternateTree = null;
                            //ht.activeLeafNodeCount--;
                        } else if (this.alternateTree instanceof adaInactiveLearningNodeForRegression) {
                            this.alternateTree = null;
                            //ht.inactiveLeafNodeCount--;
                        } else {
                            ((EFDTRegressionSplitNode) this.alternateTree).killTreeChilds(ht);
                        }
                        ht.prunedAlternateTrees++;
                    }
                }
            }


            // check if a better split is available. if so, chop the tree at this point, copying likelihood. predictors for children are from parent likelihood.
            if (this.alternateTree != null) {
                ((EFDTRegressionNode) this.alternateTree).learnFromInstance(weightedInst, ht, parent, parentBranch);
            }

            int childBranch = this.instanceChildIndex(inst);
            Node child = this.getChild(childBranch);
            if (child != null) {
                ((EFDTRegressionNode) child).learnFromInstance(weightedInst, ht, this, childBranch);
            }
        }

        @Override
        public void setParent(EFDTRegressionSplitNode parent) {

        }

        @Override
        public EFDTRegressionSplitNode getParent() {
            return null;
        }

        @Override
        public int numberLeaves() {
            int numLeaves = 0;
            for (Node child : this.children) {
                if (child != null) {
                    numLeaves += ((EFDTRegressionNode) child).numberLeaves();
                }
            }
            return numLeaves;
        }

        @Override
        public double getErrorEstimation() {
            return this.estimationErrorWeight.getEstimation();

        }

        @Override
        public double getErrorWidth() {
            double w = 0.0;
            if (isNullError() == false) {
                w = this.estimationErrorWeight.getWidth();
            }
            return w;
        }

        @Override
        public boolean isNullError() {
            return (this.estimationErrorWeight == null);

        }

        @Override
        public void killTreeChilds(adaEFDTRegression ht) {
            for (Node child : this.children) {
                if (child != null) {
                    //Delete alternate tree if it exists
                    if (child instanceof EFDTRegressionSplitNode && ((EFDTRegressionSplitNode) child).alternateTree != null) {
                        ((EFDTRegressionNode) ((EFDTRegressionSplitNode) child).alternateTree).killTreeChilds(ht);
                        ht.prunedAlternateTrees++;
                    }
                    //Recursive delete of SplitNodes
                    if (child instanceof EFDTRegressionSplitNode) {
                        ((EFDTRegressionNode) child).killTreeChilds(ht);
                    }
                    if (child instanceof ActiveLearningNodeForRegression) {
                        child = null;
                        ht.activeLeafNodeCount--;
                    } else if (child instanceof adaInactiveLearningNodeForRegression) {
                        child = null;
                        ht.inactiveLeafNodeCount--;
                    }
                }
            }

        }

        public AttributeSplitSuggestion[] getBestSplitSuggestions(
                SplitCriterion criterion, adaEFDTRegression ht) {
            List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
            double[] nodeSplitDist = new double[] {examplesSeen, sumOfValues, sumOfSquares};

            for (int i = 0; i < this.attributeObservers.size(); i++) {
                HoeffdingNumericAttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs != null) {

                    // AT THIS STAGE NON-NUMERIC ATTRIBUTES ARE IGNORED
                    AttributeSplitSuggestion bestSuggestion = null;
                    if (obs instanceof HoeffdingNumericAttributeClassObserver) {
                        bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion, nodeSplitDist, i, true);
                    }

                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }


        protected void reEvaluateBestSplit(EFDTRegressionSplitNode node,EFDTRegressionSplitNode parent, int parentIndex){
            node.addToSplitAttempts(1);
            int currentSplit = -1;
            if(this.splitTest != null){
                currentSplit = this.splitTest.getAttsTestDependsOn()[0];
                // given the current implementations in MOA, we're only ever expecting one int to be returned
            } else{ // there is no split, split is null
                currentSplit = -1;
            }
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(adaEFDTRegression.this.splitCriterionOption);
            double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getClassDistributionAtTimeOfCreation()),
                    adaEFDTRegression.this.splitConfidenceOption.getValue(), node.observedClassDistribution.sumOfValues());

            // get best split suggestions
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, adaEFDTRegression.this);
            Arrays.sort(bestSplitSuggestions);
            Set<Integer> poorAtts = new HashSet<Integer>();
            AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];

            for (int i = 0; i < bestSplitSuggestions.length; i++){

                if (bestSplitSuggestions[i].splitTest != null){
                    if (!node.getVarianceRatiosum().containsKey((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0])))
                    {
                        node.getVarianceRatiosum().put((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0]), 0.0);
                    }
                    double currentSum = node.getVarianceRatiosum().get((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0]));
                    node.getVarianceRatiosum().put((bestSplitSuggestions[i].splitTest.getAttsTestDependsOn()[0]), currentSum + bestSplitSuggestions[i].merit);
                }

                else { // handle the null attribute. this is fine to do- it'll always average zero, and we will use this later to potentially burn bad splits.
                    double currentSum = node.getVarianceRatiosum().get(-1); // null split
                    node.getVarianceRatiosum().put(-1, currentSum + bestSplitSuggestions[i].merit);
                }

            }
            // get the average merit for best and current splits

            double bestSuggestionAverageMerit = 0.0;
            double currentAverageMerit = 0.0;
            if(node.splitTest == null) { // current is null- shouldn't happen, check for robustness
                currentAverageMerit = node.getVarianceRatiosum().get(-1)/node.getNumSplitAttempts();

            }
            else {

                bestSuggestionAverageMerit = node.getVarianceRatiosum().get(bestSuggestion.splitTest.getAttsTestDependsOn()[0])/node.getNumSplitAttempts();
            }

            if(node.splitTest == null) { // current is null- shouldn't happen, check for robustness
                currentAverageMerit = node.getVarianceRatiosum().get(-1)/node.getNumSplitAttempts();
            } else {
                currentAverageMerit = node.getVarianceRatiosum().get(node.splitTest.getAttsTestDependsOn()[0])/node.getNumSplitAttempts();
            }

            double tieThreshold =adaEFDTRegression.this.tieThresholdOption.getValue();

            // compute the average deltaG
            double deltaG = bestSuggestionAverageMerit - currentAverageMerit;

            if (deltaG > hoeffdingBound
                    || (hoeffdingBound < tieThreshold && deltaG > tieThreshold / 2)) {

                System.err.println(numInstances);

                AttributeSplitSuggestion splitDecision = bestSuggestion;

                // if null split wins
                if(splitDecision.splitTest == null){

                    node.killSubtree(adaEFDTRegression.this);
                    EFDTRegressionLearningNode replacement = (EFDTRegressionLearningNode)newLearningNode();
                    replacement.setVarianceRatiosum(node.getVarianceRatiosum()); // transfer infogain history, split to replacement leaf
                    if(node.getParent() != null){
                        node.getParent().setChild(parentIndex, replacement);
                    } else {
                        assert(node.getParent().isRoot());
                        node.setRoot(true);
                    }
                }

                else {

                    Node newSplit = newSplitNode(splitDecision.splitTest,
                            node.getObservedClassDistribution(), splitDecision.numSplits());

                    ((EFDTRegressionSplitNode)newSplit).attributeObservers = node.attributeObservers; // copy the attribute observers
                    newSplit.setVarianceRatiosum(node.getVarianceRatiosum());  // transfer infogain history, split to replacement split

                    if (node.splitTest == splitDecision.splitTest
                            && node.splitTest.getClass() == NumericAttributeBinaryTest.class &&
                            (argmax(splitDecision.resultingClassDistributions[0]) == argmax(node.getChild(0).getObservedClassDistribution())
                                    ||	argmax(splitDecision.resultingClassDistributions[1]) == argmax(node.getChild(1).getObservedClassDistribution()) )
                    ){
                        // change split but don't destroy the subtrees
                        for (int i = 0; i < splitDecision.numSplits(); i++) {
                            ((EFDTRegressionSplitNode)newSplit).setChild(i, this.getChild(i));
                        }

                    } else {

                        // otherwise, torch the subtree and split on the new best attribute.

                        this.killSubtree(adaEFDTRegression.this);

                        for (int i = 0; i < splitDecision.numSplits(); i++) {

                            double[] j = splitDecision.resultingClassDistributionFromSplit(i);

                            Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));

                            if(splitDecision.splitTest.getClass() == NominalAttributeBinaryTest.class
                                    ||splitDecision.splitTest.getClass() == NominalAttributeMultiwayTest.class){
                                newChild.usedNominalAttributes = new ArrayList<Integer>(node.usedNominalAttributes); //deep copy
                                newChild.usedNominalAttributes.add(splitDecision.splitTest.getAttsTestDependsOn()[0]);
                                // no  nominal attribute should be split on more than once in the path
                            }
                            ((EFDTRegressionSplitNode)newSplit).setChild(i, newChild);
                        }

                        adaEFDTRegression.this.activeLeafNodeCount--;
                        adaEFDTRegression.this.decisionNodeCount++;
                        adaEFDTRegression.this.activeLeafNodeCount += splitDecision.numSplits();

                    }


                    if (parent == null) {
                        ((EFDTRegressionNode)newSplit).setRoot(true);
                        ((EFDTRegressionNode)newSplit).setParent(null);
                        adaEFDTRegression.this.treeRoot = newSplit;
                    } else {
                        ((EFDTRegressionNode)newSplit).setRoot(false);
                        ((EFDTRegressionNode)newSplit).setParent(parent);
                        parent.setChild(parentIndex, newSplit);
                    }
                }
            }









        }

    }

    protected HoeffdingNumericAttributeClassObserver newNumericClassObserver() {
        return new HoeffdingNumericAttributeClassObserver();
    }

    //New for options vote
    public FoundNode[] filterInstanceToLeaves(Instance inst, SplitNode parent, int parentBranch, boolean updateSplitterCounts) {
        List<FoundNode> nodes = new LinkedList<FoundNode>();
        ((EFDTRegressionNode) this.treeRoot).filterInstanceToLeaves(inst, parent, parentBranch, nodes, updateSplitterCounts);
        return nodes.toArray(new FoundNode[nodes.size()]);
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.treeRoot != null) {
            FoundNode[] foundNodes = filterInstanceToLeaves(inst, null, -1, false);
            DoubleVector result = new DoubleVector();
            int predictionPaths = 0;
            for (FoundNode foundNode : foundNodes) {
                if (foundNode.parentBranch != -999) {
                    Node leafNode = foundNode.node;
                    if (leafNode == null) {
                        leafNode = foundNode.parent;
                    }
                    double[] dist = leafNode.getClassVotes(inst, this);
                    //Albert: changed for weights
                    double distSum = Utils.sum(dist);
                    if (distSum > 0.0) {
                        Utils.normalize(dist, distSum);
                    }
                    result.addValues(dist);
                    //predictionPaths++;
                }
            }
            //if (predictionPaths > this.maxPredictionPaths) {
            //	this.maxPredictionPaths++;
            //}
            return result.getArrayRef();
        }
        return new double[0];
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

        ((EFDTRegressionNode)this.treeRoot).learnFromInstance(inst, this, null, -1);
        numInstances++;

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
          /*  for (int i = 0; i < bestSplitSuggestions.length; i++){

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

            }*/
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

    protected LearningNode newLearningNode() {
        EFDTRegressionPerceptron p = new EFDTRegressionPerceptron();
        return new EFDTRegressionLearningNode (new double[0],p);
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

    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, int size)  {
        return new EFDTRegressionSplitNode(splitTest,classObservations,size);
    }
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations)  {
        return new EFDTRegressionSplitNode(splitTest,classObservations);
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
}
