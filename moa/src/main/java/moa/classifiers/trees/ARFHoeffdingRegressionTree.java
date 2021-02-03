/*
 *    ARFHoeffdingTree.java
 *
 *    @author Heitor Murilo Gomes (heitor_murilo_gomes at yahoo dot com dot br)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

package moa.classifiers.trees;

import com.github.javacliparser.IntOption;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.Utils;
import com.yahoo.labs.samoa.instances.Instance;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Adaptive Random Forest Hoeffding Tree.
 *
 * <p>Adaptive Random Forest Hoeffding Tree. This is the base model for the
 * Adaptive Random Forest ensemble learner
 * (See moa.classifiers.meta.AdaptiveRandomForest.java). This Hoeffding Tree
 * includes a subspace size k parameter, which defines the number of randomly
 * selected features to be considered at each split. </p>
 *
 * <p>See details in:<br> Heitor Murilo Gomes, Albert Bifet, Jesse Read,
 * Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes,
 * Talel Abdessalem. Adaptive random forests for evolving data stream classification.
 * In Machine Learning, DOI: 10.1007/s10994-017-5642-8, Springer, 2017.</p>
 *
 * @author Heitor Murilo Gomes (heitor_murilo_gomes at yahoo dot com dot br)
 * @version $Revision: 1 $
 */
public class ARFHoeffdingRegressionTree extends HoeffdingRegressionTree {

    private static final long serialVersionUID = 1L;

    public IntOption subspaceSizeOption = new IntOption("subspaceSizeSize", 'r',
            "Number of features per subset for each node split. Negative values = #features - k",
            2, Integer.MIN_VALUE, Integer.MAX_VALUE);

    @Override
    public String getPurposeString() {
        return "Adaptive Random Forest Hoeffding Tree for data streams. "
                + "Base learner for AdaptiveRandomForest.";
    }

    public static class RandomLearningNodeForRegression extends ActiveLearningNodeForRegression {

        private static final long serialVersionUID = 1L;

        protected int[] listAttributes;

        protected int numAttributes;


        public RandomLearningNodeForRegression(double[] initialClassObservations, LearningNodePerceptron p, int subspaceSize) {
            super(initialClassObservations, p);
            this.numAttributes = subspaceSize;
        }

        @Override
        public void learnFromInstance(Instance inst, HoeffdingTree ht) {
            this.observedClassDistribution.addToValue(0,
                    inst.weight());
            this.observedClassDistribution.addToValue(1,
                    inst.weight() * inst.classValue());
            this.observedClassDistribution.addToValue(2,
                    inst.weight() * inst.classValue() * inst.classValue());


            if (regressionTree == false) {
                learningModel.updatePerceptron(inst);
            }
            if (this.listAttributes == null) {
                this.listAttributes = new int[this.numAttributes];
                for (int j = 0; j < this.numAttributes; j++) {
                    boolean isUnique = false;
                    while (isUnique == false) {
                        this.listAttributes[j] = ht.classifierRandom.nextInt(inst.numAttributes() - 1);
                        isUnique = true;
                        for (int i = 0; i < j; i++) {
                            if (this.listAttributes[j] == this.listAttributes[i]) {
                                isUnique = false;
                                break;
                            }
                        }
                    }
                }


                for (int j = 0; j < this.numAttributes - 1; j++) {
                    int i = this.listAttributes[j];
                    int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
                    AttributeClassObserver obs = this.attributeObservers.get(i);
                    if (obs == null) {
                        obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                        this.attributeObservers.set(i, obs);
                    }
                    obs.observeAttributeTarget(inst.value(instAttIndex), inst.classValue());
                }
            }
        }
    }

            public static class MeanLearningNode extends RandomLearningNodeForRegression {

                public MeanLearningNode(double[] initialClassObservations, LearningNodePerceptron p,int subspaceSize) {
                    super(initialClassObservations, p,subspaceSize);
                }

                @Override
                public double[] getClassVotes(Instance inst, HoeffdingTree ht) {

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
            public static class PerceptronLearningNode extends RandomLearningNodeForRegression {

                public PerceptronLearningNode(double[] initialClassObservations, LearningNodePerceptron p,int subspaceSize) {
                    super(initialClassObservations, p,subspaceSize);
                }

                @Override
                public double[] getClassVotes(Instance inst, HoeffdingTree ht) {
                    return new double[] {learningModel.prediction(inst)};
                }
            }





    public ARFHoeffdingRegressionTree() {
                this.removePoorAttsOption = null;
            }



    @Override
    protected LearningNode newLearningNode(double[] initialClassObservations,LearningNodePerceptron p) {
        if (meanPredictionNodeOption.isSet())
        {
            regressionTree=true ;
            return new MeanLearningNode(initialClassObservations, p,this.subspaceSizeOption.getValue());
        }
        regressionTree=false ;
        return new PerceptronLearningNode(initialClassObservations, p,this.subspaceSizeOption.getValue());
    }


            @Override
            public boolean isRandomizable () {
                return true;
            }

    @Override
    public void enforceTrackerLimit() {

    }
    @Override
    protected void deactivateLearningNode(ActiveLearningNode toDeactivate,
                                          SplitNode parent, int parentBranch) {

    }





}


