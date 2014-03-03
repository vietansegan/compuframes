package data;

/**
 *
 * @author vietan
 */
public class Annotation {

    private String annotator;
    private double frame;
    private int round;

    public Annotation(String annotator, double frame, int round) {
        this.annotator = annotator;
        this.frame = frame;
        this.round = round;
    }

    public String getAnnotator() {
        return annotator;
    }

    public void setAnnotator(String annotator) {
        this.annotator = annotator;
    }

    public double getFrame() {
        return frame;
    }

    public void setFrame(double annotation) {
        this.frame = annotation;
    }

    public int getRound() {
        return round;
    }

    public void setRound(int round) {
        this.round = round;
    }
    
    @Override
    public String toString() {
        return annotator + ", " + frame + ", " + round;
    }
}
