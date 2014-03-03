package experiment;

import core.AbstractExperiment;
import core.crossvalidation.Fold;
import data.CompuframesDataset;
import data.CorpusProcessor;
import data.LabelTextDataset;
import data.TextDataset;
import java.io.File;
import model.MultiLabelEvaluator;
import model.RandomBaseline;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import util.CLIUtils;
import util.IOUtils;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;

/**
 *
 * @author vietan
 */
public class MultilabelExperiment extends AbstractExperiment<CompuframesDataset> {

    public static final String PREDICTION_FILE = "predictions.txt";
    public static final String RESULT_FILE = "results.txt";
    protected static CorpusProcessor corpProc;
    private String datasetName;
    private String datasetFolder;
    private MultiLabelEvaluator evaluator;

    @Override
    public void setup() {
        if (verbose) {
            logln("Setting up ...");
        }
        datasetName = CLIUtils.getStringArgument(cmd, "dataset", "compuframes");
        datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "data");
        data = new CompuframesDataset(datasetName, datasetFolder);

        evaluator = new MultiLabelEvaluator();
    }

    @Override
    public void preprocess() throws Exception {
        if (verbose) {
            logln("Preprocessing ...");
        }
        String jsonFile = CLIUtils.getStringArgument(cmd, "json-file",
                "rounds_4-9_singly_coded.json");
        String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder",
                "format");
        corpProc = TextDataset.createCorpusProcessor();
        data.setCorpusProcessor(corpProc);
        data.loadCorpus(new File(jsonFile));
        data.format(new File(datasetFolder, formatFolder));
    }

    public void createDocumentCrossValidation() throws Exception {
        if (verbose) {
            logln("Creating document-level cross-validation ...");
        }

        corpProc = TextDataset.createCorpusProcessor();
        String formatFile = CLIUtils.getStringArgument(cmd, "format-file", datasetName);
        String jsonFile = CLIUtils.getStringArgument(cmd, "json-file",
                "rounds_4-9_singly_coded.json");

        int numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
        double trToDevRatio = CLIUtils.getDoubleArgument(cmd, "tr2dev-ratio", 0.8);
        String cvFolder = cmd.getOptionValue("cv-folder");
        IOUtils.createFolder(cvFolder);

        data.setCorpusProcessor(corpProc);
        data.setFormatFilename(formatFile);
        data.loadCorpus(new File(jsonFile));
        data.createCrossValidation(cvFolder, numFolds, trToDevRatio);
    }

    @Override
    public void run() throws Exception {
        if (verbose) {
            logln("Running ...");
        }

        int numFolds = CLIUtils.getIntegerArgument(cmd, "num-folds", 5);
        String cvFolder = cmd.getOptionValue("cv-folder");

        int foldIndex = -1;
        if (cmd.hasOption("fold")) {
            foldIndex = Integer.parseInt(cmd.getOptionValue("fold"));
        }

        for (int ff = 0; ff < numFolds; ff++) {
            if (foldIndex != -1 && ff != foldIndex) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + ff);
            }

            Fold fold = new Fold(ff, cvFolder);
            File foldFolder = new File(new File(experimentPath, datasetName), fold.getFoldName());
            data.loadCrossValidation(cvFolder, ff);
            LabelTextDataset trainData = data.getTrainData();
            LabelTextDataset devData = data.getDevelopmentData();
            LabelTextDataset testData = data.getTestData();

            if (verbose) {
                System.out.println("Fold " + fold.getFoldName());
                System.out.println("--- # labels: " + data.getLabelVocab().size());
                System.out.println("--- training: " + trainData.getWords().length
                        + "\t" + data.getMulanTrainData().getNumInstances());
                System.out.println("--- development: " + devData.getWords().length
                        + "\t" + data.getMulanDevelopmentData().getNumInstances());
                System.out.println("--- test: " + testData.getWords().length
                        + "\t" + data.getMulanTestData().getNumInstances());
                System.out.println();
            }

            String model = CLIUtils.getStringArgument(cmd, "model", "random");
            if (model.equals("random")) {
                runRandomBaseline(foldFolder);
            } else if (model.equals("br")) {
                runBinaryRelevance(foldFolder);
            } else {
                throw new RuntimeException("Model " + model + " not supported");
            }
        }
    }

    private void runRandomBaseline(File outputFolder) throws Exception {
        if (verbose) {
            logln("--- Run random baseline ...");
        }
        RandomBaseline model = new RandomBaseline();
        model.build(data.getMulanTrainData());

        if (verbose) {
            logln("--- Evaluating model on test data ...");
        }
        File modelFolder = new File(outputFolder, model.getName());
        IOUtils.createFolder(modelFolder);
        evaluator.evaluate(model, data.getMulanTestData(),
                new File(modelFolder, PREDICTION_FILE),
                new File(modelFolder, RESULT_FILE));
    }

    private void runBinaryRelevance(File outputFolder) throws Exception {
        Classifier classifer;
        String classifierName = CLIUtils.getStringArgument(cmd, "classifier", "c45");
        if (classifierName.equals("svm")) {
            classifer = new SMO();
        } else if (classifierName.equals("c45")) {
            classifer = new J48();
        } else {
            throw new RuntimeException("Classifier " + classifierName
                    + " is not supported");
        }

        if (verbose) {
            logln("--- Running Binary Relevance with " + classifierName);
        }
        BinaryRelevance model = new BinaryRelevance(classifer);
        model.setDebug(true);
        model.build(data.getMulanTrainData());

        File modelFolder = new File(outputFolder, model.globalInfo());
        IOUtils.createFolder(modelFolder);

        evaluator.evaluate(model, data.getMulanTestData(),
                new File(modelFolder, PREDICTION_FILE),
                new File(modelFolder, RESULT_FILE));
    }

    @Override
    public void evaluate() {
        if (verbose) {
            logln("Evaluating ...");
        }
    }

    public static void main(String[] args) {
        try {
            // create the command line parser
            parser = new BasicParser();

            // create the Options
            options = new Options();

            // directories
            addOption("json-file", "Path to the raw JSON file");
            addOption("dataset", "Folder storing processed data");
            addOption("data-folder", "Folder storing processed data");
            addOption("format-folder", "Format folder");
            addOption("model-folder", "Model folder");

            addOption("run-mode", "Run mode");
            addOption("model", "Model");

            // mode parameters
            addGreekParametersOptions();

            // processing options
            addCorpusProcessorOptions();

            // cross validation
            addCrossValidationOptions();

            // sampling
            addSamplingOptions();

            options.addOption("train", false, "train");
            options.addOption("dev", false, "development");
            options.addOption("test", false, "test");
            options.addOption("parallel", false, "parallel");

            options.addOption("paramOpt", false, "Optimizing parameters");
            options.addOption("v", false, "verbose");
            options.addOption("d", false, "debug");
            options.addOption("help", false, "Help");

            cmd = parser.parse(options, args);

            if (cmd.hasOption("help")) {
                CLIUtils.printHelp(getHelpString(MultilabelExperiment.class
                        .getName()), options);
                return;
            }

            verbose = cmd.hasOption("v");
            debug = cmd.hasOption("d");

            MultilabelExperiment expt = new MultilabelExperiment();
            expt.setup();
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "preprocess");
            if (runMode.equals("preprocess")) {
                expt.preprocess();
            } else if (runMode.equals("create-cv")) {
                expt.createDocumentCrossValidation();
            } else if (runMode.equals("run")) {
                expt.run();
            } else {
                throw new RuntimeException("Run mode " + runMode + " is not supported");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
