package model;

import java.util.Random;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.TechnicalInformation;

/**
 *
 * @author vietan
 */
public class RandomBaseline extends MultiLabelLearnerBase {

    private Random rand;

    public RandomBaseline() {
        this.rand = new Random(1);
    }

    public String getName() {
        return "random";
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
        double[] probabilities = new double[numLabels];
        for (int ii = 0; ii < numLabels; ii++) {
            probabilities[ii] = rand.nextDouble();
        }
        return new MultiLabelOutput(probabilities);
    }

    @Override
    public String globalInfo() {
        return getName();
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        return null;
    }
}
