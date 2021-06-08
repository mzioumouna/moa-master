package moa.classifiers.trees;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.conditionaltests.NominalAttributeBinaryTest;
import moa.classifiers.core.conditionaltests.NominalAttributeMultiwayTest;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;

import java.util.*;

public class EFDTR extends HoeffdingRegressionTree {
    protected int numInstances = 0;

    protected int splitCount=0;
    //region ================ OPTIONS ================
    public IntOption reEvalPeriodOption = new IntOption(
            "reevaluationPeriod",
            'R',
            "The number of instances an internal node should observe between re-evaluation attempts.",
            2000, 0, Integer.MAX_VALUE);


    public interface EFDTReg {

        public boolean isRoot();
        public void setRoot(boolean isRoot);

        public void learnFromInstance(Instance inst, EFDTR ht, EFDTsplitNodeForRegression parent, int parentBranch);
        public void setParent(EFDTsplitNodeForRegression parent);

        public EFDTsplitNodeForRegression getParent();

    }
    //region ================ CLASSES ================
    public  class EFDTsplitNodeForRegression extends SplitNode implements EFDTReg{

        private boolean isRoot;
        protected AutoExpandVector<AttributeClassObserver> attributeObservers;
        private EFDTsplitNodeForRegression parent = null;


        public EFDTsplitNodeForRegression(InstanceConditionalTest splitTest, double[] classObservations, int size) {
            super(splitTest, classObservations, size);
        }
        public EFDTsplitNodeForRegression(InstanceConditionalTest splitTest, double[] classObservations) {
            super(splitTest, classObservations);
        }
        public boolean isRoot() {
            return isRoot;
        }
        public void setRoot(boolean isRoot) {
            this.isRoot = isRoot;
        }
        public void killSubtree(HoeffdingRegressionTree ht){
            for (Node child : this.children) {
                if (child != null) {

                    //Recursive delete of SplitNodes
                    if (child instanceof SplitNode) {
                        ((EFDTsplitNodeForRegression) child).killSubtree(ht);
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

        public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion, EFDTR ht) {
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
        public void learnFromInstance(Instance inst, EFDTR ht, EFDTsplitNodeForRegression parent, int parentBranch) {
            nodeTime++;
            //// Update node statistics and class distribution
            this.observedClassDistribution.addToValue(0,
                    inst.weight());
            this.observedClassDistribution.addToValue(1,
                    inst.weight() * inst.classValue());
            this.observedClassDistribution.addToValue(2,
                    inst.weight() * inst.classValue() * inst.classValue());


            for (int i = 0; i < inst.numAttributes() - 1; i++) { //update likelihood
                int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                AttributeClassObserver obs = this.attributeObservers.get(i);
                if (obs == null) {
                    obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                    this.attributeObservers.set(i, obs);
                }
                obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
            }

            // check if a better split is available. if so, chop the tree at this point, copying likelihood. predictors for children are from parent likelihood.
            if(ht.numInstances % ht.reEvalPeriodOption.getValue() == 0){
                this.reEvaluateBestSplit(this, parent, parentBranch);
            }

            int childBranch = this.instanceChildIndex(inst);
            Node child = this.getChild(childBranch);

            if (child != null) {
                ((EFDTReg) child).learnFromInstance(inst, ht, this, childBranch);
            }
        }

       protected void reEvaluateBestSplit(EFDTsplitNodeForRegression node, EFDTsplitNodeForRegression parent, int parentIndex){
            node.addToSplitAttempts(1);
            int currentSplit = -1;
            if(this.splitTest != null){
                currentSplit = this.splitTest.getAttsTestDependsOn()[0];
                // given the current implementations in MOA, we're only ever expecting one int to be returned
            } else{ // there is no split, split is null
                currentSplit = -1;
            }
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(EFDTR.this.splitCriterionOption);
            double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getClassDistributionAtTimeOfCreation()),
                    EFDTR.this.splitConfidenceOption.getValue(), node.observedClassDistribution.sumOfValues());

            // get best split suggestions
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, EFDTR.this);
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

            double tieThreshold =EFDTR.this.tieThresholdOption.getValue();

            // compute the average deltaG
            double deltaG = bestSuggestionAverageMerit - currentAverageMerit;

            if (deltaG > hoeffdingBound
                    || (hoeffdingBound < tieThreshold && deltaG > tieThreshold / 2)) {

                System.err.println(numInstances);

                AttributeSplitSuggestion splitDecision = bestSuggestion;

                // if null split wins
                if(splitDecision.splitTest == null){

                    node.killSubtree(EFDTR.this);
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

                    ((EFDTsplitNodeForRegression)newSplit).attributeObservers = node.attributeObservers; // copy the attribute observers
                    newSplit.setVarianceRatiosum(node.getVarianceRatiosum());  // transfer varianceratio history, split to replacement split

                    if (node.splitTest == splitDecision.splitTest
                            && node.splitTest.getClass() == NumericAttributeBinaryTest.class &&
                            (argmax(splitDecision.resultingClassDistributions[0]) == argmax(node.getChild(0).getObservedClassDistribution())
                                    ||	argmax(splitDecision.resultingClassDistributions[1]) == argmax(node.getChild(1).getObservedClassDistribution()) )
                    ){
                        // change split but don't destroy the subtrees
                        for (int i = 0; i < splitDecision.numSplits(); i++) {
                            ((EFDTsplitNodeForRegression)newSplit).setChild(i, this.getChild(i));
                        }

                    } else {

                        // otherwise, torch the subtree and split on the new best attribute.

                        this.killSubtree(EFDTR.this);

                        for (int i = 0; i < splitDecision.numSplits(); i++) {

                            double[] j = splitDecision.resultingClassDistributionFromSplit(i);

                            Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));

                            if(splitDecision.splitTest.getClass() == NominalAttributeBinaryTest.class
                                    ||splitDecision.splitTest.getClass() == NominalAttributeMultiwayTest.class){
                                newChild.usedNominalAttributes = new ArrayList<Integer>(node.usedNominalAttributes); //deep copy
                                newChild.usedNominalAttributes.add(splitDecision.splitTest.getAttsTestDependsOn()[0]);
                                // no  nominal attribute should be split on more than once in the path
                            }
                            ((EFDTsplitNodeForRegression)newSplit).setChild(i, newChild);
                        }

                        EFDTR.this.activeLeafNodeCount--;
                        EFDTR.this.decisionNodeCount++;
                        EFDTR.this.activeLeafNodeCount += splitDecision.numSplits();

                    }


                    if (parent == null) {
                        ((EFDTReg)newSplit).setRoot(true);
                        ((EFDTReg)newSplit).setParent(null);
                        EFDTR.this.treeRoot = newSplit;
                    } else {
                        ((EFDTReg)newSplit).setRoot(false);
                        ((EFDTReg)newSplit).setParent(parent);
                        parent.setChild(parentIndex, newSplit);
                    }
                }
            }



        }
     public EFDTsplitNodeForRegression getParent() {
         return this.parent;
     }

        public void setParent(EFDTsplitNodeForRegression parent) {
            this.parent = parent;
        }


    }
    public  class EFDTRLearningNode extends PerceptronLearningNode implements EFDTReg {
        private boolean isRoot;
        private EFDTsplitNodeForRegression parent = null;
        public EFDTRLearningNode(double[] initialClassObservations, LearningNodePerceptron p) {
            super(initialClassObservations, p);
        }

        public boolean isRoot() {
            return false;
        }

        public void filterInstanceToLeaves(Instance inst, EFDTsplitNodeForRegression splitparent, int parentBranch, List<FoundNode> foundNodes, boolean updateSplitterCounts) {
            foundNodes.add(new FoundNode(this, splitparent, parentBranch));

        }

        public void setRoot(boolean isRoot) {

        }
        public void learnFromInstance(Instance inst, HoeffdingRegressionTree ht) {
            super.learnFromInstance(inst, ht);

        }

        public void learnFromInstance(Instance inst, EFDTR ht, EFDTsplitNodeForRegression parent, int parentBranch) {
            learnFromInstance(inst, ht);

            if (ht.growthAllowed
                    && (this instanceof ActiveLearningNodeForRegression)) {
                ActiveLearningNodeForRegression activeLearningNode = this;
                double weightSeen = activeLearningNode.getWeightSeen();
                if (activeLearningNode.nodeTime % ht.gracePeriodOption.getValue() == 0) {
                    attemptToSplit(activeLearningNode, parent,parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
            }
        }


        public void setParent(EFDTsplitNodeForRegression parent) {
            this.parent = parent;
        }

        public EFDTsplitNodeForRegression getParent() {
            return this.parent;
        }
    }


    //endregion ================ CLASSES ================
    protected void attemptToSplit(ActiveLearningNodeForRegression node, EFDTsplitNodeForRegression parent, int parentIndex) {
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
                    ((EFDTsplitNodeForRegression)newSplit).attributeObservers = node.attributeObservers; // copy the attribute observers
                    ((EFDTsplitNodeForRegression)newSplit).setVarianceRatiosum(node.getVarianceRatiosum());
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
    protected LearningNode newLearningNode(double[] initialClassObservations, LearningNodePerceptron p) {
        // IDEA: to choose different learning nodes depending on predictionOption
        return new EFDTRLearningNode (initialClassObservations,p);
    }
    protected LearningNode newLearningNode() {
        LearningNodePerceptron p = new LearningNodePerceptron();
        return newLearningNode(new double[0],p);
    }

    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, int size) {
        return new EFDTsplitNodeForRegression(splitTest, classObservations, size);
    }
    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations) {
        return new EFDTsplitNodeForRegression(splitTest, classObservations);
    }

    public void trainOnInstanceImpl(Instance inst) {
//Updating the tree statistics
        examplesSeen += inst.weight();
        sumOfValues += inst.weight() * inst.classValue();
        sumOfSquares += inst.weight() * inst.classValue() * inst.classValue();
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            ((EFDTReg) this.treeRoot).setRoot(true);
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


    public void enforceTrackerLimit() {

    }


}
