package moa.classifiers.trees;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NominalAttributeBinaryTest;
import moa.classifiers.core.conditionaltests.NominalAttributeMultiwayTest;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.MiscUtils;

import java.util.*;

public class EFRTDD extends EFDTR {
    protected int numInstances = 0;

    protected int alternateTrees;

    protected int prunedAlternateTrees;

    protected int switchedAlternateTrees;
    protected int splitCount=0;
    //region ================ OPTIONS ================

    public interface NewEFAT {
        public void setParent(adaEFDTsplitNode parent);

        public adaEFDTsplitNode getParent();
        public int numberLeaves();
        public void learnFromInstance(Instance inst,EFRTDD  ht, adaEFDTsplitNode parent, int parentBranch);
        public boolean isRoot();
        public void setRoot(boolean isRoot);
        public double getErrorEstimation();

        public double getErrorWidth();

        public boolean isNullError();
        public void killTreeChilds(EFRTDD ht);

        public void filterInstanceToLeaves(Instance inst, SplitNode mygressionparent, int parentBranch, List<FoundNode> foundNodes,
                                           boolean updateSplitterCounts);
    }

    public class adaEFDTsplitNode extends SplitNode implements NewEFAT {
        private static final long serialVersionUID = 1L;
        protected AutoExpandVector<AttributeClassObserver> attributeObservers;
        private adaEFDTsplitNode parent = null;
        protected Random classifierRandom;

        protected Node alternateTree;
        private boolean isRoot;
        protected ADWIN estimationErrorWeight;
        //public boolean isAlternateTree = false;

        public boolean ErrorChange = false;
        public adaEFDTsplitNode(InstanceConditionalTest splitTest, double[] classObservations) {
            super(splitTest, classObservations);
        }
        public adaEFDTsplitNode(InstanceConditionalTest splitTest, double[] classObservations, int size) {
            super(splitTest, classObservations, size);
        }

        @Override
        public void setParent(adaEFDTsplitNode parent) {
            this.parent = parent;
        }

        @Override
        public adaEFDTsplitNode getParent() {
            return this.parent;
        }

        @Override
        public int numberLeaves() {
            int numLeaves = 0;
            for (Node child : this.children) {
                if (child != null) {
                    numLeaves += ((NewEFAT) child).numberLeaves();                }
            }
            return numLeaves;
        }

        @Override
        public void learnFromInstance(Instance inst, EFRTDD ht, adaEFDTsplitNode parent, int parentBranch) {
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
            } // Check condition to replace tree
            else if (this.alternateTree != null && ((NewEFAT) this.alternateTree).isNullError() == false) {
                if (this.getErrorWidth() > 300 && ((NewEFAT) this.alternateTree).getErrorWidth() > 300) {
                    double oldErrorRate = this.getErrorEstimation();
                    double altErrorRate = ((NewEFAT) this.alternateTree).getErrorEstimation();
                    double fDelta = .05;
                    //if (gNumAlts>0k) fDelta=fDelta/gNumAlts;
                    double fN = 1.0 / ((double) ((NewEFAT) this.alternateTree).getErrorWidth()) + 1.0 / ((double) this.getErrorWidth());
                    double Bound = (double) Math.sqrt((double) 2.0 * oldErrorRate * (1.0 - oldErrorRate) * Math.log(2.0 / fDelta) * fN);
                    if (Bound < oldErrorRate - altErrorRate) {
                        // Switch alternate tree
                        ht.activeLeafNodeCount -= this.numberLeaves();
                        ht.activeLeafNodeCount += ((NewEFAT) this.alternateTree).numberLeaves();
                        killTreeChilds(ht);
                        if (parent != null) {
                            parent.setChild(parentBranch, this.alternateTree);
                            //((AdaSplitNode) parent.getChild(parentBranch)).alternateTree = null;
                        } else {
                            // Switch root tree
                            ht.treeRoot = ((adaEFDTsplitNode) ht.treeRoot).alternateTree;
                        }
                        ht.switchedAlternateTrees++;
                    } else if (Bound < altErrorRate - oldErrorRate) {
                        // Erase alternate tree
                        if (this.alternateTree instanceof ActiveLearningNodeForRegression) {
                            this.alternateTree = null;
                            //ht.activeLeafNodeCount--;
                        } else if (this.alternateTree instanceof InactiveLearningNodeForRegression) {
                            this.alternateTree = null;
                            //ht.inactiveLeafNodeCount--;
                        } else {
                            ((adaEFDTsplitNode) this.alternateTree).killTreeChilds(ht);
                        }
                        ht.prunedAlternateTrees++;
                    }
                }
            }
            //}
            //learnFromInstance alternate Tree and Child nodes
            if (this.alternateTree != null) {
                ((NewEFAT) this.alternateTree).learnFromInstance(weightedInst, ht, parent, parentBranch);
            }
            int childBranch = this.instanceChildIndex(inst);
            Node child = this.getChild(childBranch);
            if (child != null) {
                ((NewEFAT) child).learnFromInstance(weightedInst, ht, this, childBranch);
            }
        }

        @Override
        public boolean isRoot() {
            return isRoot;        }

        @Override
        public void setRoot(boolean isRoot) {
            this.isRoot = isRoot;
        }

        @Override
        public double getErrorEstimation() {
            return this.estimationErrorWeight.getEstimation();

        }
        public void killSubtree(EFRTDD ht){
            for (Node child : this.children) {
                if (child != null) {

                    //Recursive delete of SplitNodes
                    if (child instanceof SplitNode) {
                        ((adaEFDTsplitNode) child).killSubtree(ht);
                    }
                    else if (child instanceof ActiveLearningNodeForRegression) {
                        child = null;
                        ht.activeLeafNodeCount--;
                    }
                    else if (child instanceof InactiveLearningNodeForRegression) {
                        child = null;
                        ht.inactiveLeafNodeCount--;
                    }
                    else{

                    }
                }
            }
        }
        public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion, EFRTDD ht) {
            List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
            double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
            if (!ht.noPrePruneOption.isSet()) {
                // add null split as an option
                bestSuggestions.add(new AttributeSplitSuggestion(null,
                        new double[0][], criterion.getMeritOfSplit(
                        preSplitDist, new double[][]{preSplitDist})));
            }
            for (int i = 0; i < this.attributeObservers.size(); i++) {
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs != null) {
                    AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                            preSplitDist, i, ht.binarySplitsOption.isSet());
                    if (bestSuggestion != null) {
                        bestSuggestions.add(bestSuggestion);
                    }
                }
            }
            return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
        }
        protected void reEvaluateBestSplit(adaEFDTsplitNode node, adaEFDTsplitNode parent, int parentIndex){
            node.addToSplitAttempts(1);
            int currentSplit = -1;
            if(this.splitTest != null){
                currentSplit = this.splitTest.getAttsTestDependsOn()[0];
                // given the current implementations in MOA, we're only ever expecting one int to be returned
            } else{ // there is no split, split is null
                currentSplit = -1;
            }
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(EFRTDD.this.splitCriterionOption);
            double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getClassDistributionAtTimeOfCreation()),
                    EFRTDD.this.splitConfidenceOption.getValue(), node.observedClassDistribution.sumOfValues());

            // get best split suggestions
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, EFRTDD.this);
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

            double tieThreshold =EFRTDD.this.tieThresholdOption.getValue();

            // compute the average deltaG
            double deltaG = bestSuggestionAverageMerit - currentAverageMerit;

            if (deltaG > hoeffdingBound
                    || (hoeffdingBound < tieThreshold && deltaG > tieThreshold / 2)) {

                System.err.println(numInstances);

                AttributeSplitSuggestion splitDecision = bestSuggestion;

                // if null split wins
                if(splitDecision.splitTest == null){

                    node.killSubtree(EFRTDD.this);
                    EFDTRLearningNode replacement = (EFDTRLearningNode)newLearningNode();
                    replacement.setVarianceRatiosum(node.getVarianceRatiosum()); // transfer varianceratio history, split to replacement leaf
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

                    ((adaEFDTsplitNode)newSplit).attributeObservers = node.attributeObservers; // copy the attribute observers
                    newSplit.setVarianceRatiosum(node.getVarianceRatiosum());  // transfer varianceratio history, split to replacement split

                    if (node.splitTest == splitDecision.splitTest
                            && node.splitTest.getClass() == NumericAttributeBinaryTest.class &&
                            (argmax(splitDecision.resultingClassDistributions[0]) == argmax(node.getChild(0).getObservedClassDistribution())
                                    ||	argmax(splitDecision.resultingClassDistributions[1]) == argmax(node.getChild(1).getObservedClassDistribution()) )
                    ){
                        // change split but don't destroy the subtrees
                        for (int i = 0; i < splitDecision.numSplits(); i++) {
                            ((adaEFDTsplitNode)newSplit).setChild(i, this.getChild(i));
                        }

                    } else {

                        // otherwise, torch the subtree and split on the new best attribute.

                        this.killSubtree(EFRTDD.this);

                        for (int i = 0; i < splitDecision.numSplits(); i++) {

                            double[] j = splitDecision.resultingClassDistributionFromSplit(i);

                            Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));

                            if(splitDecision.splitTest.getClass() == NominalAttributeBinaryTest.class
                                    ||splitDecision.splitTest.getClass() == NominalAttributeMultiwayTest.class){
                                newChild.usedNominalAttributes = new ArrayList<Integer>(node.usedNominalAttributes); //deep copy
                                newChild.usedNominalAttributes.add(splitDecision.splitTest.getAttsTestDependsOn()[0]);
                                // no  nominal attribute should be split on more than once in the path
                            }
                            ((adaEFDTsplitNode)newSplit).setChild(i, newChild);
                        }

                        EFRTDD.this.activeLeafNodeCount--;
                        EFRTDD.this.decisionNodeCount++;
                        EFRTDD.this.activeLeafNodeCount += splitDecision.numSplits();

                    }


                    if (parent == null) {
                        ((NewEFAT)newSplit).setRoot(true);
                        ((NewEFAT)newSplit).setParent(null);
                        EFRTDD.this.treeRoot = newSplit;
                    } else {
                        ((NewEFAT)newSplit).setRoot(false);
                        ((NewEFAT)newSplit).setParent(parent);
                        parent.setChild(parentIndex, newSplit);
                    }
                }
            }



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

        public void killTreeChilds(EFRTDD ht) {
            for (Node child : this.children) {
                if (child != null) {
                    //Delete alternate tree if it exists
                    if (child instanceof adaEFDTsplitNode && ((adaEFDTsplitNode) child).alternateTree != null) {
                        ((NewEFAT) ((adaEFDTsplitNode) child).alternateTree).killTreeChilds(ht);
                        ht.prunedAlternateTrees++;
                    }
                    //Recursive delete of SplitNodes
                    if (child instanceof adaEFDTsplitNode) {
                        ((NewEFAT) child).killTreeChilds(ht);
                    }
                    if (child instanceof ActiveLearningNodeForRegression) {
                        child = null;
                        ht.activeLeafNodeCount--;
                    } else if (child instanceof InactiveLearningNodeForRegression) {
                        child = null;
                        ht.inactiveLeafNodeCount--;
                    }
                }
            }
        }


        public void filterInstanceToLeaves(Instance inst, SplitNode myparent,
                                           int parentBranch, List<FoundNode> foundNodes,
                                           boolean updateSplitterCounts) {
            if (updateSplitterCounts) {
                this.observedClassDistribution.addToValue(0, inst.weight());
                this.observedClassDistribution.addToValue(1, inst.weight()*inst.classValue());
                this.observedClassDistribution.addToValue(2, inst.weight()*inst.classValue()*inst.classValue());

            }
            int childIndex = instanceChildIndex(inst);
            if (childIndex >= 0) {
                Node child = getChild(childIndex);
                if (child != null) {
                    ((NewEFAT) child).filterInstanceToLeaves(inst, this, childIndex,
                            foundNodes, updateSplitterCounts);
                } else {
                    foundNodes.add(new FoundNode(null, this, childIndex));
                }
            }
            if (this.alternateTree != null) {
                ((NewEFAT) this.alternateTree).filterInstanceToLeaves(inst, this, -999,
                        foundNodes, updateSplitterCounts);
            }
        }

    }
    public static class AdaEFDTLearningNodeForRegression extends PerceptronLearningNode implements NewEFAT {
        private static final long serialVersionUID = 1L;

        protected ADWIN estimationErrorWeight;

        public boolean ErrorChange = false;

        protected int randomSeed = 1;
        protected Random classifierRandom;


        public AdaEFDTLearningNodeForRegression(double[] initialClassObservations, LearningNodePerceptron p) {
            super(initialClassObservations, p);
        }

        @Override
        public void setParent(adaEFDTsplitNode parent) {

        }

        @Override
        public adaEFDTsplitNode getParent() {
            return null;
        }

        @Override
        public int numberLeaves() {
            return 1;
        }

        @Override
        public void learnFromInstance(Instance inst, EFRTDD ht, adaEFDTsplitNode parent, int parentBranch) {
            double trueValue =  inst.classValue();
            //New option vore
            int k = MiscUtils.poisson(1.0, this.classifierRandom);
            Instance weightedInst = (Instance) inst.copy();
            if (k > 0) {
                weightedInst.setWeight(inst.weight() * k);
            }
            //Compute ClassPrediction using filterInstanceToLeaf
            double targetPrediction = 0;
            double absError=0;
            double[] distribution = (this.getClassVotes(inst, ht));
            targetPrediction = distribution[0];

            absError = Math.abs((targetPrediction-trueValue)/ht.examplesSeen);

            if (this.estimationErrorWeight == null) {
                this.estimationErrorWeight = new ADWIN();
            }
            double oldError = this.getErrorEstimation();
            this.ErrorChange = this.estimationErrorWeight.setInput(absError);
            if (this.ErrorChange == true && oldError > this.getErrorEstimation()) {
                this.ErrorChange = false;
            }

            //Update statistics
            learnFromInstance(weightedInst, ht);	//inst

            //Check for Split condition
            double weightSeen = this.getWeightSeen();
            if (weightSeen
                    - this.getWeightSeenAtLastSplitEvaluation() >= ht.gracePeriodOption.getValue()) {
                ht.attemptToSplit(this, parent,
                        parentBranch);
                this.setWeightSeenAtLastSplitEvaluation(weightSeen);
            }
        }

        @Override
        public boolean isRoot() {
            return false;
        }

        @Override
        public void setRoot(boolean isRoot) {

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
            return (this.estimationErrorWeight == null);

        }
        public void killTreeChilds(EFRTDD ht) {
        }

        @Override
        public void filterInstanceToLeaves(Instance inst, SplitNode splitparent, int parentBranch, List<FoundNode> foundNodes, boolean updateSplitterCounts) {
            foundNodes.add(new FoundNode(this, splitparent, parentBranch));

        }

        public double[] getClassVotes(Instance inst, EFRTDD ht) {
            double[] dist;
            boolean predictionOption = ( ht).meanPredictionNodeOption.isSet();
            if (predictionOption == true) { //MeanClass
                dist = this.observedClassDistribution.getArrayCopy();
                double prediction = this.observedClassDistribution.getValue(1)/this.observedClassDistribution.getValue(0);
                //System.out.println(prediction);
                return new double[] {prediction};

            } else { //Perceptron
                return new double[] {this.learningModel.prediction(inst)};

            }


        }

        public void filterInstanceToLeaves(Instance inst, adaEFDTsplitNode splitparent, int parentBranch,
                                           List<FoundNode> foundNodes, boolean updateSplitterCounts) {
            foundNodes.add(new FoundNode(this, splitparent, parentBranch));
        }





    }
    //endregion ================ CLASSES ================

    protected void attemptToSplit(ActiveLearningNodeForRegression node, adaEFDTsplitNode parent, int parentIndex) {
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
                    ((adaEFDTsplitNode)newSplit).attributeObservers = node.attributeObservers; // copy the attribute observers
                    ((adaEFDTsplitNode)newSplit).setVarianceRatiosum(node.getVarianceRatiosum());
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i),new LearningNodePerceptron((LearningNodePerceptron) node.learningModel));
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

    //New for options vote
    public FoundNode[] filterInstanceToLeaves(Instance inst,
                                              SplitNode parent, int parentBranch, boolean updateSplitterCounts) {
        List<FoundNode> nodes = new LinkedList<FoundNode>();
        ((NewEFAT) this.treeRoot).filterInstanceToLeaves(inst, parent, parentBranch, nodes,
                updateSplitterCounts);
        return nodes.toArray(new FoundNode[nodes.size()]);
    }

    public void trainOnInstanceImpl(Instance inst) {
        examplesSeen += inst.weight();
        sumOfValues += inst.weight() * inst.classValue();
        sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            ((NewEFAT) this.treeRoot).setRoot(true);
            this.activeLeafNodeCount = 1;
        }

        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            int aIndex = modelAttIndexToInstanceAttIndex(i, inst);
            sumOfAttrValues.addToValue(i, inst.weight() * inst.value(aIndex));
            sumOfAttrSquares.addToValue(i, inst.weight() * inst.value(aIndex) * inst.value(aIndex));
        }
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;

        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }
        if (leafNode instanceof LearningNode) {
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
        numInstances++;
    }


    protected LearningNode newLearningNode(double[] initialClassObservations, LearningNodePerceptron p) {
        // IDEA: to choose different learning nodes depending on predictionOption
        return new AdaEFDTLearningNodeForRegression(initialClassObservations,p);
    }
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest, double[] classObservations) {
        return new adaEFDTsplitNode(splitTest, classObservations);
    }

    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, int size) {
        return new adaEFDTsplitNode(splitTest, classObservations, size);
    }
    protected LearningNode newLearningNode() {
        LearningNodePerceptron p = new LearningNodePerceptron();
        return newLearningNode(new double[0],p);
    }

    protected void deactivateLearningNode(ActiveLearningNode toDeactivate,
                                          SplitNode parent, int parentBranch) {

    }


}
