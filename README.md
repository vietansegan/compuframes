compuframes
===========

## Required libraries/external files in `lib` folder
- commons-cli-1.2.jar
- gson-2.2.4.jar
- segan.jar
- commons-compress-1.4.1.jar
- mulan.jar
- stopwords.txt
- commons-math3-3.1-SNAPSHOT.jar
- opennlp-maxent-3.0.1-incubating.jar
- weka-3.7.6.jar
- en-sent.bin
- opennlp-tools-1.5.1-incubating.jar
- en-token.bin
- opennlp-uima-1.5.1-incubating.jar

## Compile
- To compile: `ant compile`
- To build jar file: `ant jar`
- To make a clean build: `ant clean-build`
  
## Process data
### To perform preprocessing of the text data on the whole dataset

```
java -Xmx10000M -Xms10000M -cp 'dist/compuframes.jar:lib/*' experiment.MultilabelExperiment --json-file <json-file> -v -d --data-folder <processed-data-folder> --format-folder <format-folder> --run-mode preprocess -s -l --u 5 --b 10 --bs 5 --V 10000 --doc-type <document-type>
```
- `<json-file>`: path to the JSON file
- `<processed-data-folder>`: path to the folder containing processed data
- `<format-folder>`: subfolder of `<processed-data-folder>` to store one specific instance of processed data (e.g., with a specific set of preprocessing parameter of the same dataset)
- `<document-type>`: the type of documents to consider. This can be either "tobacco" or "immigration". If this is not set, all documents are considered. 
- `-s`: whether stopwords are removed (stopwords are stored in `lib/stopwords.txt`)
- `-l`: whether stemming is performed
- `--u`: the minimum raw count of unigrams
- `--b`: the minimum raw count of bigrams
- `--bs`: the minimum chi-square score for a bigram to be considered
- `--V`: the maximum vocab size
For other arguments, use `-help`.

Working example:
```
java -Xmx10000M -Xms10000M -cp 'dist/compuframes.jar:lib/*' experiment.MultilabelExperiment --json-file data/rounds_4-9_singly_coded.json -v -d --data-folder data --format-folder format --run-mode preprocess -s -l --u 5 --b 10 --bs 5 -s -l --V 10000
```

### To create cross-validated data

```
java -Xmx10000M -Xms10000M -cp 'dist/compuframes.jar:lib/*' experiment.MultilabelExperiment --json-file <json-file> -v -d --run-mode create-cv -s -l --u 5 --b 10 --bs 5 --V 10000 --num-folds <number-of-folds> --tr2dev-ratio <train-to-dev-ratio> --cv-folder <cross-validation-folder>
```
- `<number-of-folds>`: number of cross validation folds
- `<train-to-dev-ratio>`: the ratio between the number of training instances and the number of development instances
- `<cross-validation-folder>`: folder to store the cross validated data

This will creates `<number-of-folds>` sets of train/dev/test. It will perform preprocessing (like in section 1 above) on the training set and use the vocabulary extracted from the training data to perform preprocessing on the development and test sets.

Working example:
```
java -Xmx10000M -Xms10000M -cp 'dist/compuframes.jar:lib/*' experiment.MultilabelExperiment --json-file data/rounds_4-9_singly_coded.json -v -d --run-mode create-cv -s -l --u 5 --b 10 --bs 5 -s -l --V 10000 --num-folds 5 --tr2dev-ratio 0.8 --cv-folder data/cv
```
    
## Run document-level multi-label model
```
java -Xmx10000M -Xms10000M -cp 'dist/compuframes.jar:lib/*:../segan/dist/lib/*:../segan/dist/*' experiment.MultilabelExperiment -v -d --dataset compuframes --run-mode run --num-folds <number-of-folds> --cv-folder <cross-validation-folder> --model <model-name> --expt-folder <experiment-folder> --fold <fold-number>
```
- `<model-name>`: the name of a supported model
- `<experiment-folder>`: folder to store the learned model and prediction results
- `<fold-number>`: the fold to be run. If this is not specified, all folds in the cross-validation folder will be run.

Working example: running random baseline
```
java -Xmx10000M -Xms10000M -cp 'dist/compuframes.jar:lib/*:../segan/dist/lib/*:../segan/dist/*' experiment.MultilabelExperiment -v -d --dataset compuframes --run-mode run --num-folds 5 --cv-folder data/cv --model random --expt-folder experiments --fold 0
```
