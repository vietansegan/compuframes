package model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.*;
import util.IOUtils;
import util.MiscUtils;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author vietan
 */
public class MultiLabelEvaluator {

    private InternalEvaluator evaluator;

    public MultiLabelEvaluator() {
        this.evaluator = new InternalEvaluator();
    }

//    public int[] wekaInstance2GibbsArray(SparseInstance instance, MultiLabelLearnerBase a) {
//        ArrayList<Integer> list = new ArrayList<Integer>();
//
//        for (int ii = 0; ii < instance.numValues(); ii++) {
//            int attrIdx = instance.attributeSparse(ii).index();
//            if (attrIdx < a.featureIndices.length) {
//                int count = (int) instance.value(attrIdx);
//                for (int jj = 0; jj < count; jj++) {
//                    list.add(attrIdx);
//                }
//            }
//        }
//        int[] gibbsArr = new int[list.size()];
//        for (int ii = 0; ii < gibbsArr.length; ii++) {
//            gibbsArr[ii] = list.get(ii);
//        }
//        return gibbsArr;
//    }

    public void outputEvaluation(File file, Evaluation eval) {
        try {
            BufferedWriter writer = IOUtils.getBufferedWriter(file);
            writer.write(eval.toString());
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing evaluation");
        }
    }

    public void evaluate(
            MultiLabelLearnerBase model,
            MultiLabelInstances testData,
            File predictionFile,
            File resultFile) {
        try {
            System.out.println(">>> Evaluating ...");
            Evaluation eval = this.evaluator.evaluate(model, testData, predictionFile);
            System.out.println(">>> Outputing to " + resultFile);
            this.outputEvaluation(resultFile, eval);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while evaluating " + model.globalInfo());
        }
    }

    class InternalEvaluator extends Evaluator implements Serializable {

        public Evaluation evaluate(MultiLabelLearner learner,
                MultiLabelInstances data, List<Measure> measures,
                File file)
                throws IllegalArgumentException, Exception {
            checkLearner(learner);
            checkData(data);
            checkMeasures(measures);

            // reset measures
            for (Measure m : measures) {
                m.reset();
            }

            BufferedWriter writer = IOUtils.getBufferedWriter(file);

            int numLabels = data.getNumLabels();
            int[] labelIndices = data.getLabelIndices();
            boolean[] trueLabels;
            Set<Measure> failed = new HashSet<Measure>();
            Instances testData = data.getDataSet();
            int numInstances = testData.numInstances();
            for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
                Instance instance = testData.instance(instanceIndex);
                if (data.hasMissingLabels(instance)) {
                    continue;
                }
                Instance labelsMissing = (Instance) instance.copy();
                labelsMissing.setDataset(instance.dataset());
                for (int i = 0; i < data.getNumLabels(); i++) {
                    labelsMissing.setMissing(data.getLabelIndices()[i]);
                }
                MultiLabelOutput output = learner.makePrediction(labelsMissing);
                trueLabels = getTrueLabels(instance, numLabels, labelIndices);
                Iterator<Measure> it = measures.iterator();
                while (it.hasNext()) {
                    Measure m = it.next();
                    if (!failed.contains(m)) {
                        try {
                            m.update(output, trueLabels);
                        } catch (Exception ex) {
                            failed.add(m);
                        }
                    }
                }

                writer.write(MiscUtils.arrayToString(output.getConfidences()) + "\n");
            }

            writer.close();

            return new Evaluation(measures, data);
        }

        public Evaluation evaluate(MultiLabelLearner learner,
                MultiLabelInstances data,
                File file) throws IllegalArgumentException, Exception {
            checkLearner(learner);
            checkData(data);
            List<Measure> measures = prepareMeasures(learner, data);
            return evaluate(learner, data, measures, file);
        }

        private List<Measure> prepareMeasures(MultiLabelLearner learner, MultiLabelInstances data) {
            List<Measure> measures = new ArrayList<Measure>();

            MultiLabelOutput prediction;
            try {
                MultiLabelLearner copyOfLearner = learner.makeCopy();
                prediction = copyOfLearner.makePrediction(data.getDataSet().instance(0));
                // add bipartition-based measures if applicable
                if (prediction.hasBipartition()) {
                    // add example-based measures
                    measures.add(new HammingLoss());
                    measures.add(new SubsetAccuracy());
                    measures.add(new ExampleBasedPrecision());
                    measures.add(new ExampleBasedRecall());
                    measures.add(new ExampleBasedFMeasure());
                    measures.add(new ExampleBasedAccuracy());
                    measures.add(new ExampleBasedSpecificity());
                    // add label-based measures
                    int numOfLabels = data.getNumLabels();
                    measures.add(new MicroPrecision(numOfLabels));
                    measures.add(new MicroRecall(numOfLabels));
                    measures.add(new MicroFMeasure(numOfLabels));
                    measures.add(new MicroSpecificity(numOfLabels));
                    measures.add(new MacroPrecision(numOfLabels));
                    measures.add(new MacroRecall(numOfLabels));
                    measures.add(new MacroFMeasure(numOfLabels));
                    measures.add(new MacroSpecificity(numOfLabels));
                }
                // add ranking-based measures if applicable
                if (prediction.hasRanking()) {
                    // add ranking based measures
                    measures.add(new AveragePrecision());
                    measures.add(new Coverage());
                    measures.add(new OneError());
                    measures.add(new IsError());
                    measures.add(new ErrorSetSize());
                    measures.add(new RankingLoss());
                }
                // add confidence measures if applicable
                if (prediction.hasConfidences()) {
                    int numOfLabels = data.getNumLabels();
                    measures.add(new MeanAveragePrecision(numOfLabels));
                    measures.add(new GeometricMeanAveragePrecision(numOfLabels));
                    measures.add(new MeanAverageInterpolatedPrecision(numOfLabels, 10));
                    measures.add(new GeometricMeanAverageInterpolatedPrecision(numOfLabels, 10));
                    measures.add(new MicroAUC(numOfLabels));
                    measures.add(new MacroAUC(numOfLabels));
                }
                // add hierarchical measures if applicable
                if (data.getLabelsMetaData().isHierarchy()) {
                    measures.add(new HierarchicalLoss(data));
                }
            } catch (Exception ex) {
                Logger.getLogger(Evaluator.class.getName()).log(Level.SEVERE, null, ex);
            }

            return measures;
        }

        private boolean[] getTrueLabels(Instance instance, int numLabels,
                int[] labelIndices) {

            boolean[] trueLabels = new boolean[numLabels];
            for (int counter = 0; counter < numLabels; counter++) {
                int classIdx = labelIndices[counter];
                String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
                trueLabels[counter] = classValue.equals("1");
            }

            return trueLabels;
        }

        private void checkLearner(MultiLabelLearner learner) {
            if (learner == null) {
                throw new IllegalArgumentException("Learner to be evaluated is null.");
            }
        }

        private void checkData(MultiLabelInstances data) {
            if (data == null) {
                throw new IllegalArgumentException("Evaluation data object is null.");
            }
        }

        private void checkMeasures(List<Measure> measures) {
            if (measures == null) {
                throw new IllegalArgumentException("List of evaluation measures to compute is null.");
            }
        }
    }
}
