package data;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import core.crossvalidation.CrossValidation;
import core.crossvalidation.Fold;
import core.crossvalidation.Instance;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import mulan.data.MultiLabelInstances;
import util.IOUtils;

/**
 *
 * @author vietan
 */
public class CompuframesDataset extends LabelTextDataset {

    public static final String sentInfoExt = ".sentinfo";
    // raw input from JSON file
    private HashMap<String, Document> documents;
    // internal
    private String[][] sentences;
    private double[][][] sentenceAnnotations;
    private int[][][] sentenceLabels;
    
    // cross validation
    private LabelTextDataset trainData;
    private LabelTextDataset devData;
    private LabelTextDataset testData;
    private MultiLabelInstances mulanTrainData;
    private MultiLabelInstances mulanDevData;
    private MultiLabelInstances mulanTestData;

    public CompuframesDataset(String name, String folder) {
        super(name, folder);
        this.documents = new HashMap<String, Document>();
    }

    public CompuframesDataset(String name, String folder,
            CorpusProcessor corpProc) {
        super(name, folder, corpProc);
        this.documents = new HashMap<String, Document>();
    }

    public void loadCorpus(File jsonFile) {
        if (verbose) {
            logln("Loading corpus from " + jsonFile);
        }

        try {
            StringBuilder str = new StringBuilder();
            BufferedReader reader = IOUtils.getBufferedReader(jsonFile);
            String line;
            while ((line = reader.readLine()) != null) {
                str.append(line).append("\n");
            }
            reader.close();

            Gson gson = new Gson();
            Type mapType = new TypeToken<Map<String, Document>>() {
            }.getType();
            Map<String, Document> map = (Map<String, Document>) gson.fromJson(str.toString(), mapType);

            for (String key : map.keySet()) {
                Document doc = map.get(key);
                doc.parseAnnotatedSentences();
                this.documents.put(key, doc);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading corpus from "
                    + jsonFile);
        }

        if (verbose) {
            logln("--- Loaded " + this.documents.size() + " documents");
        }

        int D = documents.size();

        // document-level data
        this.docIdList = new ArrayList<String>();
        this.textList = new ArrayList<String>();
        this.labelList = new ArrayList<ArrayList<String>>();

        // sentence-level data
        this.sentences = new String[D][];
        this.sentenceAnnotations = new double[D][][];

        int dd = 0;
        for (String docId : documents.keySet()) {
            this.docIdList.add(docId);

            Document doc = documents.get(docId);
            int numSents = doc.getAnnotatedSentences().size();

            StringBuilder docText = new StringBuilder();
            ArrayList<String> docLabels = new ArrayList<String>();

            this.sentences[dd] = new String[numSents];
            this.sentenceAnnotations[dd] = new double[numSents][];

            for (int ss = 0; ss < numSents; ss++) {
                Sentence sent = doc.getAnnotatedSentences().get(ss);
                this.sentences[dd][ss] = sent.getText();
                docText.append(" ").append(sent.getText());

                List<Annotation> sentAnnts = sent.getAnnotations();
                if (sentAnnts == null || sentAnnts.isEmpty()) {
                    this.sentenceAnnotations[dd][ss] = new double[0];
                } else {
                    this.sentenceAnnotations[dd][ss] = new double[sentAnnts.size()];
                    for (int ii = 0; ii < sentAnnts.size(); ii++) {
                        this.sentenceAnnotations[dd][ss][ii] = sentAnnts.get(ii).getFrame();

                        int frameLabel = (int) sentAnnts.get(ii).getFrame();
                        docLabels.add(Integer.toString(frameLabel));
                    }
                }
            }

            this.textList.add(docText.toString());
            this.labelList.add(docLabels);
            dd++;
        }
    }

    @Override
    public void format(String outputFolder) throws Exception {
        if (verbose) {
            logln("Formatting ...");
        }
        IOUtils.createFolder(outputFolder);

        formatLabels(outputFolder);

        String[] rawTexts = textList.toArray(new String[textList.size()]);
        corpProc.setRawTexts(rawTexts);
        corpProc.process(sentences);

        outputWordVocab(outputFolder);
        outputTextData(outputFolder);
        outputDocumentInfo(outputFolder);
        outputSentTextData(outputFolder);

        // provide I/O for JSON files
    }

    @Override
    protected void outputSentTextData(String outputFolder) throws Exception {
        File sentTextFile = new File(outputFolder, formatFilename + numSentDataExt);
        File sentInfoFile = new File(outputFolder, formatFilename + sentInfoExt);
        if (verbose) {
            logln("--- Outputing sentence data ... " + outputFolder);
        }

        int[][][] numSents = corpProc.getNumericSentences();
        String[][] rawSents = corpProc.getRawSentences();
        BufferedWriter rawSentWriter = IOUtils.getBufferedWriter(sentTextFile + ".raw");
        BufferedWriter sentWriter = IOUtils.getBufferedWriter(sentTextFile);
        BufferedWriter sentInfoWriter = IOUtils.getBufferedWriter(sentInfoFile);
        for (int d : this.processedDocIndices) {
            StringBuilder docStr = new StringBuilder();
            StringBuilder docInfo = new StringBuilder();
            ArrayList<String> docRawSents = new ArrayList<String>();

            for (int s = 0; s < numSents[d].length; s++) {
                HashMap<Integer, Integer> sentTypeCounts = new HashMap<Integer, Integer>();
                for (int w = 0; w < numSents[d][s].length; w++) {
                    Integer count = sentTypeCounts.get(numSents[d][s][w]);
                    if (count == null) {
                        sentTypeCounts.put(numSents[d][s][w], 1);
                    } else {
                        sentTypeCounts.put(numSents[d][s][w], count + 1);
                    }
                }

                if (sentTypeCounts.size() > 0) {
                    // store numeric sentence
                    StringBuilder str = new StringBuilder();
                    for (int type : sentTypeCounts.keySet()) {
                        str.append(type).append(":").append(sentTypeCounts.get(type)).append(" ");
                    }
                    docStr.append(str.toString().trim()).append("\t");

                    // store raw sentence
                    docRawSents.add(rawSents[d][s]);

                    // sentence label
                    str = new StringBuilder();
                    for (int ll = 0; ll < this.sentenceAnnotations[d][s].length; ll++) {
                        int frameLabel = (int) sentenceAnnotations[d][s][ll];
                        str.append(frameLabel).append(" ");
                    }
                    docInfo.append(str.toString().trim()).append("\t");
                }
            }
            // write numeric sentence
            sentWriter.write(docStr.toString().trim() + "\n");

            // write raw sentence
            rawSentWriter.write(docRawSents.size() + "\n");
            for (String docRawSent : docRawSents) {
                rawSentWriter.write(docRawSent.trim().replaceAll("\n", " ") + "\n");
            }

            // write sentence labels
            sentInfoWriter.write(docInfo.toString().trim() + "\n");
        }
        sentWriter.close();
        rawSentWriter.close();
        sentInfoWriter.close();
    }

    @Override
    public void loadFormattedData(String fFolder) {
        try {
            super.loadFormattedData(fFolder);
            this.inputSentenceInfo(new File(fFolder, formatFilename + sentInfoExt));
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading formatted data "
                    + "from " + fFolder);
        }
    }

    protected void inputSentenceInfo(File sentInfoFile) throws Exception {
        this.sentenceLabels = new int[docIds.length][][];
        BufferedReader reader = IOUtils.getBufferedReader(sentInfoFile);
        String line;
        String[] sline;
        int d = 0;
        while ((line = reader.readLine()) != null) {
            if (line.isEmpty()) {
                d++;
                continue;
            }

            sline = line.split("\t");
            this.sentenceLabels[d] = new int[sline.length][];
            for (int s = 0; s < sline.length; s++) {
                if (sline[s].isEmpty()) {
                    continue;
                }
                String[] ssline = sline[s].split(" ");
                this.sentenceLabels[d][s] = new int[ssline.length];
                for (int ii = 0; ii < ssline.length; ii++) {
                    this.sentenceLabels[d][s][ii] = Integer.parseInt(ssline[ii]);
                }
            }
        }
        reader.close();
    }
    
    @Override
    public void createCrossValidation(String cvFolder, int numFolds,
            double trToDevRatio) throws Exception {
        ArrayList<Instance<String>> instanceList = new ArrayList<Instance<String>>();
        ArrayList<Integer> groupIdList = new ArrayList<Integer>();
        for (int d = 0; d < this.docIdList.size(); d++) {
            instanceList.add(new Instance<String>(docIdList.get(d)));
            groupIdList.add(0); // random, no stratified
        }

        String cvName = "";
        CrossValidation<String, Instance<String>> cv =
                new CrossValidation<String, Instance<String>>(
                cvFolder,
                cvName,
                instanceList);

        cv.stratify(groupIdList, numFolds, trToDevRatio);
        cv.outputFolds();

        for (Fold<String, Instance<String>> fold : cv.getFolds()) {
            String foldFolder = fold.getFoldFolderPath();
            // processor
            CorpusProcessor cp = new CorpusProcessor(corpProc);

            // training data
            trainData = new LabelTextDataset(fold.getFoldName(),
                    cv.getFolderPath(), cp);
            trainData.setFormatFilename(fold.getFoldName() + Fold.TrainingExt);
            ArrayList<String> trDocIds = new ArrayList<String>();
            ArrayList<String> trDocTexts = new ArrayList<String>();
            ArrayList<ArrayList<String>> trLabelList = new ArrayList<ArrayList<String>>();
            for (int ii = 0; ii < fold.getNumTrainingInstances(); ii++) {
                int idx = fold.getTrainingInstances().get(ii);
                trDocIds.add(this.docIdList.get(idx));
                trDocTexts.add(this.textList.get(idx));
                trLabelList.add(this.labelList.get(idx));
            }
            trainData.setTextData(trDocIds, trDocTexts);
            trainData.setLabelList(trLabelList);
            trainData.format(foldFolder);
            trainData.loadFormattedData(foldFolder);
            trainData.outputArffFile(new File(foldFolder,
                    trainData.getFormatFilename() + arffExt));
            setLabelVocab(trainData.getLabelVocab());
            outputLabelVocabXML(foldFolder);

            // development data: process using vocab from training
            devData = new LabelTextDataset(fold.getFoldName(),
                    cv.getFolderPath(), cp);
            devData.setFormatFilename(fold.getFoldName() + Fold.DevelopExt);
            ArrayList<String> deDocIds = new ArrayList<String>();
            ArrayList<String> deDocTexts = new ArrayList<String>();
            ArrayList<ArrayList<String>> deLabelList = new ArrayList<ArrayList<String>>();
            for (int ii = 0; ii < fold.getNumDevelopmentInstances(); ii++) {
                int idx = fold.getDevelopmentInstances().get(ii);
                deDocIds.add(this.docIdList.get(idx));
                deDocTexts.add(this.textList.get(idx));
                deLabelList.add(this.labelList.get(idx));
            }
            devData.setTextData(deDocIds, deDocTexts);
            devData.setLabelVocab(trainData.getLabelVocab());
            devData.setLabelList(deLabelList);
            devData.format(foldFolder);
            devData.loadFormattedData(foldFolder);
            devData.outputArffFile(new File(foldFolder,
                    devData.getFormatFilename() + arffExt));

            // test data: process using vocab from training
            testData = new LabelTextDataset(fold.getFoldName(),
                    cv.getFolderPath(), cp);
            testData.setFormatFilename(fold.getFoldName() + Fold.TestExt);
            ArrayList<String> teDocIds = new ArrayList<String>();
            ArrayList<String> teDocTexts = new ArrayList<String>();
            ArrayList<ArrayList<String>> teLabelList = new ArrayList<ArrayList<String>>();
            for (int ii = 0; ii < fold.getNumTestingInstances(); ii++) {
                int idx = fold.getTestingInstances().get(ii);
                teDocIds.add(this.docIdList.get(idx));
                teDocTexts.add(this.textList.get(idx));
                teLabelList.add(this.labelList.get(idx));
            }
            testData.setTextData(teDocIds, teDocTexts);
            testData.setLabelVocab(trainData.getLabelVocab());
            testData.setLabelList(teLabelList);
            testData.format(foldFolder);
            testData.loadFormattedData(foldFolder);
            testData.outputArffFile(new File(foldFolder,
                    testData.getFormatFilename() + arffExt));
        }
    }
    
    public void outputLabelVocabXML(String outputFolder) throws Exception {
        // format labels in Mulan's XML
        File labelVocXml = new File(outputFolder, formatFilename + xmlExt);
        if (verbose) {
            logln("--- Outputing XML label vocab ... " + labelVocXml);

        }
        BufferedWriter writer = IOUtils.getBufferedWriter(labelVocXml);
        writer.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
        writer.write("<labels xmlns=\"http://mulan.sourceforge.net/labels\">\n");
        StringBuilder str = new StringBuilder();
        for (String label : this.labelVocab) {
            str.append("<label name=\"").append("label_").append(label).append("\"></label>\n");
        }
        writer.write(str.toString());
        writer.write("</labels>\n");
        writer.close();
    }
    
    public void loadCrossValidation(String cvFolder, int foldIdx) {
        try {
            Fold fold = new Fold(foldIdx, cvFolder);
            String foldFolderpath = fold.getFoldFolderPath();
            LabelTextDataset[] foldData = LabelTextDataset.loadCrossValidationFold(fold);
            trainData = foldData[Fold.TRAIN];
            devData = foldData[Fold.DEV];
            testData = foldData[Fold.TEST];

            mulanTrainData = new MultiLabelInstances(
                    new File(foldFolderpath, fold.getFoldName() + Fold.TrainingExt + arffExt).getAbsolutePath(),
                    new File(foldFolderpath, formatFilename + xmlExt).getAbsolutePath());
            mulanDevData = new MultiLabelInstances(
                    new File(foldFolderpath, fold.getFoldName() + Fold.DevelopExt + arffExt).getAbsolutePath(),
                    new File(foldFolderpath, formatFilename + xmlExt).getAbsolutePath());
            mulanTestData = new MultiLabelInstances(
                    new File(foldFolderpath, fold.getFoldName() + Fold.TestExt + arffExt).getAbsolutePath(),
                    new File(foldFolderpath, formatFilename + xmlExt).getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Exception while loading cross validation"
                    + " fold " + foldIdx + " from " + cvFolder);
        }
        labelVocab = trainData.labelVocab;
        wordVocab = trainData.wordVocab;
    }
    
    public LabelTextDataset getTrainData() {
        return this.trainData;
    }

    public LabelTextDataset getDevelopmentData() {
        return this.devData;
    }

    public LabelTextDataset getTestData() {
        return this.testData;
    }

    public MultiLabelInstances getMulanTrainData() {
        return mulanTrainData;
    }

    public MultiLabelInstances getMulanDevelopmentData() {
        return mulanDevData;
    }

    public MultiLabelInstances getMulanTestData() {
        return mulanTestData;
    }


//    public static void main(String[] args) {
//
//        try {
//            // create the command line parser
//            parser = new BasicParser();
//
//            // create the Options
//            options = new Options();
//
//            addOption("json-file", "Path to the raw JSON file");
//
//            options.addOption("help", false, "Help");
//            options.addOption("v", false, "verbose");
//            options.addOption("d", false, "debug");
//            cmd = parser.parse(options, args);
//
//            if (cmd.hasOption("help")) {
//                CLIUtils.printHelp(getHelpString(CompuframesDataset.class
//                        .getName()), options);
//                return;
//            }
//
//            verbose = cmd.hasOption("v");
//            debug = cmd.hasOption("d");
//
//            verbose = true;
//            debug = true;
//            String jsonFile = CLIUtils.getStringArgument(cmd, "json-file",
//                    "rounds_4-9_singly_coded.json");
//            String datasetFolder = CLIUtils.getStringArgument(cmd, "data-folder",
//                    "data");
//            String formatFolder = CLIUtils.getStringArgument(cmd, "format-folder", "format");
//            CompuframesDataset data = new CompuframesDataset("compuframes", datasetFolder);
//            data.loadCorpus(new File(jsonFile));
//            data.format(new File(datasetFolder, formatFolder));
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException();
//        }
//    }
}