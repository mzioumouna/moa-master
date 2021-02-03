package moa.streams.generators;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import moa.core.FastVector;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.io.Serializable;
import java.util.Random;
import moa.core.InstanceExample;

import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.MiscUtils;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import com.github.javacliparser.IntOption;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

/**
 * Stream generator for a random radial basis function stream.
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class RandomRBFGeneratorForRegression extends RandomRBFGenerator {
    @Override
    protected void generateHeader() {
        FastVector attributes = new FastVector();
        for (int i = 0; i < this.numAttsOption.getValue(); i++) {
            attributes.addElement(new Attribute("att" + (i + 1)));
        }
        FastVector classLabels = new FastVector();
        for (int i = 0; i < this.numClassesOption.getValue(); i++) {
            classLabels.addElement("class" + (i + 1));
        }
        attributes.addElement(new Attribute("class"));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
    }

    protected void generateCentroids() {
        Random modelRand = new Random(this.modelRandomSeedOption.getValue());
        this.centroids = new Centroid[this.numCentroidsOption.getValue()];
        this.centroidWeights = new double[this.centroids.length];
        for (int i = 0; i < this.centroids.length; i++) {
            this.centroids[i] = new Centroid();
            double[] randCentre = new double[this.numAttsOption.getValue()];
            for (int j = 0; j < randCentre.length; j++) {
                randCentre[j] = modelRand.nextDouble();
            }
            this.centroids[i].centre = randCentre;
            this.centroids[i].classLabel = modelRand.nextInt(this.numClassesOption.getValue());
            this.centroids[i].stdDev = modelRand.nextDouble();
            this.centroidWeights[i] = modelRand.nextDouble();
        }
    }




    @Override
    public InstanceExample nextInstance() {
        Centroid centroid = this.centroids[MiscUtils.chooseRandomIndexBasedOnWeights(this.centroidWeights,
                this.instanceRandom)];
        int numAtts = this.numAttsOption.getValue();
        double[] attVals = new double[numAtts + 1];
        for (int i = 0; i < numAtts; i++) {
            attVals[i] = (this.instanceRandom.nextDouble() * 2.0) - 1.0;
        }
        double magnitude = 0.0;
        for (int i = 0; i < numAtts; i++) {
            magnitude += attVals[i] * attVals[i];
        }
        magnitude = Math.sqrt(magnitude);
        double desiredMag = this.instanceRandom.nextGaussian()
                * centroid.stdDev;
        double scale = desiredMag / magnitude;
        for (int i = 0; i < numAtts; i++) {
            attVals[i] = centroid.centre[i] + attVals[i] ;
        }
        Instance inst = new DenseInstance(1.0, attVals);
        inst.setDataset(getHeader());
        inst.setClassValue(centroid.stdDev+scale);
        return new InstanceExample(inst);
    }


}
