# fasttext
Text/Sentence Classification Python Module

# Installation:

    pip install fasttext
  
# Training Data Set Format:

Example:  __label__moviereview The film's two major strengths come down to the two most important ingredients - cast and story.

Explanation:

Prefix: __label__

Label_name: moviereview

Sentence: The film's two major strengths come down to the two most important ingredients - cast and story.

Note: There should be space between Label_name and Sentence and there should be one Sentence/Line followed by it's Label_name.
  
# Training the Data Set:

    classifier = fasttext.supervised('train_data.txt', 'model_name', label_prefix ='__label__')

# Testing the Data Set:

    test_result = classifier.test('test_data.txt')

    print('Precision', test_result.precision)

    print('Recall', test_result.recall)

    print('No. of examples:', test_result.nexamples)

# Loading the model:

    model = fasttext.load_model('model_directory_path',label_prefix='__label__')

# Prediction:

    Sentence = "provide any sentence for testing"

    predict = (model.predict(sentence))

# Prediction with probability:

    predict_prob = model.predict_proba(sentence, k=2)

Note: k value indicates no. of label predictions.




