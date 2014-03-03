package data;

import java.util.List;

/**
 *
 * @author vietan
 */
public class Sentence {

    private String text;
    private List<Annotation> annotations;

    public Sentence(String text, List<Annotation> annts) {
        this.text = text;
        this.annotations = annts;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public List<Annotation> getAnnotations() {
        return annotations;
    }

    public void setAnnotations(List<Annotation> annotations) {
        this.annotations = annotations;
    }

    @Override
    public String toString() {
        return text + "\n" + annotations.toString();
    }
}
