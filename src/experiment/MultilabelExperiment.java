package experiment;

import core.AbstractExperiment;
import core.AbstractSampler;
import core.AbstractSampler.InitialState;
import core.crossvalidation.Fold;
import data.CompuframesDataset;
import data.CompuframesDataset.DocType;
import data.CorpusProcessor;
import data.LabelTextDataset;
import data.TextDataset;
import java.io.BufferedWriter;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.Options;
import sampler.labeled.LabeledLDA;
import sampler.labeled.TFIDF;
import sampler.supervised.classification.SLDA;
import util.CLIUtils;
import util.IOUtils;
import util.PredictionUtils;
import util.evaluation.Measurement;
import util.evaluation.MultilabelClassificationEvaluation;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;

/**
 *
 * @author vietan
 */
public class MultilabelExperiment extends AbstractExperiment<CompuframesDataset> {

    protected static CorpusProcessor corpProc;
    protected String datasetName;
    protected String datasetFolder;
    protected DocType docType;
//    protected MultiLabelEvaluator evaluator;
    protected LabelTextDataset trainData;
    protected LabelTextDataset devData;
    protected LabelTextDataset testData;
    protected int numTopWords = 15;

    @Override
    public void setup() {
        if (verbose) {
            logln("Setting up ...");
        }
        datasetName = CLIUtils.getStringArgument(cmd, "dataset", "compuframes");
        datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder", "data");
        data = new CompuframesDataset(datasetName, datasetFolder);
        experimentPath = CLIUtils.getStringArgument(cmd, "expt-folder", "experiment");
//        evaluator = new MultiLabelEvaluator();
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
        String dType = CLIUtils.getStringArgument(cmd, "doc-type", "all");
        if (dType.equals("all")) {
            docType = DocType.ALL;
        } else if (dType.equals("immigration")) {
            docType = DocType.IMMIGRATION;
        } else if (dType.equals("tobacco")) {
            docType = DocType.TOBACCO;
        } else {
            throw new RuntimeException("Document type " + dType + " not supported");
        }
        corpProc = TextDataset.createCorpusProcessor();
        data.setCorpusProcessor(corpProc);
        data.setDocumentType(docType);
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

        String dType = CLIUtils.getStringArgument(cmd, "doc-type", "all");
        if (dType.equals("all")) {
            docType = DocType.ALL;
        } else if (dType.equals("immigration")) {
            docType = DocType.IMMIGRATION;
        } else if (dType.equals("tobacco")) {
            docType = DocType.TOBACCO;
        } else {
            throw new RuntimeException("Document type " + dType + " not supported");
        }

        data.setCorpusProcessor(corpProc);
        data.setDocumentType(docType);
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

        ArrayList<Integer> runningFolds = new ArrayList<Integer>();
        if (cmd.hasOption("fold")) {
            String foldList = cmd.getOptionValue("fold");
            for (String f : foldList.split(",")) {
                runningFolds.add(Integer.parseInt(f));
            }
        }

        for (int ff = 0; ff < numFolds; ff++) {
            if (!runningFolds.isEmpty() && !runningFolds.contains(ff)) {
                continue;
            }
            if (verbose) {
                System.out.println("\nRunning fold " + ff);
            }

            Fold fold = new Fold(ff, cvFolder);
            File foldFolder = new File(new File(experimentPath, datasetName), fold.getFoldName());
            data.loadCrossValidation(cvFolder, ff);
            trainData = data.getTrainData();
            devData = data.getDevelopmentData();
            testData = data.getTestData();

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
            } else if (model.equals("tfidf-nn")) {
                runTFIDF_NNs(foldFolder);
            } else if (model.equals("labeled-lda")) {
                runLabeledLDA(foldFolder);
            } else if (model.equals("slda")) {
                runSLDA(foldFolder);
            } else if (model.equals("br")) {
                runBinaryRelevance(foldFolder);
            } else if (model.equals("none")) {
                logln("Doing nothing :D");
            } else {
                throw new RuntimeException("Model " + model + " not supported");
            }
            evaluate();
        }
    }

    private void runRandomBaseline(File outputFolder) throws Exception {
        if (verbose) {
            logln("--- Run random baseline ...");
        }
        Random rand = new Random(1);
        File randomFolder = new File(outputFolder, "Random-Baseline");
        File teResultFolder = new File(randomFolder, TEST_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(teResultFolder);
        double[][] finalPredictions = new double[testData.getWords().length][testData.getLabelVocab().size()];
        for (int dd = 0; dd < finalPredictions.length; dd++) {
            for (int ii = 0; ii < finalPredictions[dd].length; ii++) {
                finalPredictions[dd][ii] = rand.nextDouble();
            }
        }
        outputResults(new File(teResultFolder, RESULT_FILE), testData.getLabels(), finalPredictions);
    }

    private void runTFIDF_NNs(File outputFolder) throws Exception {
        if (verbose) {
            logln("--- Run TFIDF nearest neighbors ...");
        }

        TFIDF model = new TFIDF(trainData.getWords(),
                trainData.getLabels(),
                trainData.getLabelVocab().size(),
                trainData.getWordVocab().size());
        File modelFolder = new File(outputFolder, model.getName());
        IOUtils.createFolder(modelFolder);

        if (cmd.hasOption("train")) {
            model.learn();
            model.outputPredictor(new File(modelFolder, "model.zip"));
        }

        if (cmd.hasOption("test")) {
            File teResultFolder = new File(modelFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            double[][] predictions = new double[testData.getWords().length][testData.getLabels().length];
            for (int dd = 0; dd < predictions.length; dd++) {
                predictions[dd] = model.predict(testData.getWords()[dd]);
            }
            outputResults(new File(teResultFolder, RESULT_FILE), testData.getLabels(), predictions);
        }
    }

    private void runLabeledLDA(File outputFolder) throws Exception {
        if (verbose) {
            logln("--- Run Labeled LDA ...");
        }

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        boolean paramOpt = cmd.hasOption("paramOpt");

        LabeledLDA sampler = new LabeledLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(trainData.getWordVocab());
        sampler.setLabelVocab(trainData.getLabelVocab());

        int V = data.getWordVocab().size();
        int K = data.getLabelVocab().size();
        sampler.configure(outputFolder.getAbsolutePath(),
                V, K, alpha, beta, InitialState.RANDOM, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.train(trainData.getWords(), trainData.getLabels());
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            if (trainData.getTopicCoherence() != null) {
                sampler.outputTopicCoherence(new File(samplerFolder, TopicCoherenceFile),
                        trainData.getTopicCoherence());
            }
        }

        if (cmd.hasOption("test")) {
            File teIterPredFolder = new File(sampler.getSamplerFolderPath(),
                    TEST_PREFIX + AbstractSampler.IterPredictionFolder);
            IOUtils.createFolder(teIterPredFolder);
            if (cmd.hasOption("parallel")) {
                LabeledLDA.parallelTest(testData.getWords(),
                        teIterPredFolder, sampler);
            } else {
                sampler.test(testData.getWords(), teIterPredFolder);
            }
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            double[][] finalPredictions = PredictionUtils.evaluateClassifications(teIterPredFolder,
                    new File(teResultFolder, "iter-" + RESULT_FILE),
                    testData.getLabels());
            outputResults(new File(teResultFolder, RESULT_FILE), testData.getLabels(), finalPredictions);
        }
    }

    private void runSLDA(File outputFolder) throws Exception {
        if (verbose) {
            logln("--- Run Supervised LDA for classification ...");
        }

        int burnIn = CLIUtils.getIntegerArgument(cmd, "burnIn", 250);
        int maxIters = CLIUtils.getIntegerArgument(cmd, "maxIter", 500);
        int sampleLag = CLIUtils.getIntegerArgument(cmd, "sampleLag", 25);
        int repInterval = CLIUtils.getIntegerArgument(cmd, "report", 1);

        double alpha = CLIUtils.getDoubleArgument(cmd, "alpha", 0.1);
        double beta = CLIUtils.getDoubleArgument(cmd, "beta", 0.1);
        double sigma = CLIUtils.getDoubleArgument(cmd, "sigma", 0.5);
        boolean paramOpt = cmd.hasOption("paramOpt");

        SLDA sampler = new SLDA();
        sampler.setVerbose(verbose);
        sampler.setDebug(debug);
        sampler.setLog(true);
        sampler.setReport(true);
        sampler.setWordVocab(trainData.getWordVocab());
        sampler.setLabelVocab(trainData.getLabelVocab());

        int V = data.getWordVocab().size();
        int L = data.getLabelVocab().size();
        int K = CLIUtils.getIntegerArgument(cmd, "K", 50);
        sampler.configure(outputFolder.getAbsolutePath(),
                V, K, L, alpha, beta, sigma, InitialState.RANDOM, paramOpt,
                burnIn, maxIters, sampleLag, repInterval);
        File samplerFolder = new File(sampler.getSamplerFolderPath());
        IOUtils.createFolder(samplerFolder);

        if (cmd.hasOption("train")) {
            sampler.train(trainData.getWords(), trainData.getLabels());
            sampler.initialize();
            sampler.iterate();
            sampler.outputTopicTopWords(new File(samplerFolder, TopWordFile), numTopWords);
            if (trainData.getTopicCoherence() != null) {
                sampler.outputTopicCoherence(new File(samplerFolder, TopicCoherenceFile),
                        trainData.getTopicCoherence());
            }
        }

        if (cmd.hasOption("test")) {
            File teIterPredFolder = new File(sampler.getSamplerFolderPath(),
                    TEST_PREFIX + AbstractSampler.IterPredictionFolder);
            IOUtils.createFolder(teIterPredFolder);
            if (cmd.hasOption("parallel")) {
                SLDA.parallelTest(testData.getWords(),
                        teIterPredFolder, sampler);
            } else {
                sampler.test(testData.getWords(), teIterPredFolder);
            }
            File teResultFolder = new File(samplerFolder, TEST_PREFIX + RESULT_FOLDER);
            IOUtils.createFolder(teResultFolder);
            double[][] finalPredictions = PredictionUtils.evaluateClassifications(teIterPredFolder,
                    new File(teResultFolder, "iter-" + RESULT_FILE),
                    testData.getLabels());
            outputResults(new File(teResultFolder, RESULT_FILE), testData.getLabels(), finalPredictions);
        }
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

        File modelFolder = new File(outputFolder, "br-" + classifierName);
        IOUtils.createFolder(modelFolder);

        // test
        File teResultFolder = new File(modelFolder, TEST_PREFIX + RESULT_FOLDER);
        IOUtils.createFolder(teResultFolder);

        double[][] finalPredictions = new double[data.getMulanTestData().getNumInstances()][];
        for (int d = 0; d < data.getMulanTestData().getNumInstances(); d++) {
            MultiLabelOutput output = model.makePrediction(data.getMulanTestData().getDataSet().instance(d));
            finalPredictions[d] = output.getConfidences();
        }
        outputResults(new File(teResultFolder, RESULT_FILE), testData.getLabels(), finalPredictions);
    }

    @Override
    public void evaluate() throws Exception {
        if (verbose) {
            logln("Evaluating ...");
        }
        File resultFolder = new File(experimentPath, datasetName);
        int numFolds = Integer.parseInt(cmd.getOptionValue("num-folds"));

        if (verbose) {
            logln("Summarizing " + resultFolder);
        }
        evaluate(resultFolder.getAbsolutePath(),
                null, numFolds,
                TEST_PREFIX,
                RESULT_FILE);
    }

    private void outputResults(File outputFile, int[][] trueLabels, double[][] predictions) {
        if (verbose) {
            logln("Outputing results to " + outputFile);
        }
        try {
            MultilabelClassificationEvaluation eval = new MultilabelClassificationEvaluation(trueLabels, predictions);
            eval.computeMeasurements();
            BufferedWriter writer = IOUtils.getBufferedWriter(outputFile);
            for (Measurement m : eval.getMeasurements()) {
                writer.write(m.getName() + "\t" + m.getValue() + "\n");
            }
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while outputing results to "
                    + outputFile);
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
            addOption("expt-folder", "Experiment folder");

            addOption("run-mode", "Run mode");
            addOption("model", "Model");

            // mode parameters
            addGreekParametersOptions();

            // processing options
            addCorpusProcessorOptions();
            addOption("doc-type", "Document type");

            // cross validation
            addCrossValidationOptions();

            // sampling
            addSamplingOptions();

            // baselines
            addOption("classifier", "Weka classifier");

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
            String runMode = CLIUtils.getStringArgument(cmd, "run-mode", "run");
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
