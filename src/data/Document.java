package data;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 *
 * @author vietan
 */
public class Document {

    private HashMap<String, Integer> irrelevant;
    private JsonArray sentences;
    private List<String> xml_file;
    private String filename;
    private List<Sentence> annotatedSentences;

    public HashMap<String, Integer> getIrrelevant() {
        return irrelevant;
    }

    public void setIrrelevant(HashMap<String, Integer> irrelevant) {
        this.irrelevant = irrelevant;
    }

    public List<Sentence> getAnnotatedSentences() {
        return this.annotatedSentences;
    }

    public void parseAnnotatedSentences() {
        Gson gson = new Gson();
        this.annotatedSentences = new ArrayList<Sentence>();
        for (int ii = 0; ii < sentences.size(); ii++) {
            JsonArray sentence = sentences.get(ii).getAsJsonArray();
            String text = gson.fromJson(sentence.get(0), String.class);
            JsonArray annotations = sentence.get(1).getAsJsonArray();

            List<Annotation> annts = new ArrayList<Annotation>();
            for (int jj = 0; jj < annotations.size(); jj++) {
                JsonArray annotation = annotations.get(jj).getAsJsonArray();
                String annotator = gson.fromJson(annotation.get(0), String.class);
                double frame = gson.fromJson(annotation.get(1), Double.class);
                int round = gson.fromJson(annotation.get(2), Integer.class);
                Annotation annt = new Annotation(annotator, frame, round);
                annts.add(annt);
            }

            Sentence sent = new Sentence(text, annts);
            this.annotatedSentences.add(sent);
        }
    }

    public JsonArray getSentences() {
        return sentences;
    }

    public void setSentences(JsonArray sentences) {
        this.sentences = sentences;
    }

    public List<String> getXml_file() {
        return xml_file;
    }

    public void setXml_file(List<String> xml_file) {
        this.xml_file = xml_file;
    }

    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    @Override
    public String toString() {
        return irrelevant
                + "\n" + annotatedSentences.size()
                + "\n" + xml_file
                + "\n" + filename;
    }
}
