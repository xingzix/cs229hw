import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    m = message.split()
    lower_m = [item.lower() for item in m]
    return lower_m
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    dictionary_all = {}  # a dictionary with keys as word and values as number of messages
    for m in messages:
        words = list(set(get_words(m)))
        for w in words:
            if w in dictionary_all.keys():
                dictionary_all[w] += 1
            else:
                dictionary_all[w] = 1
    dictionary_fre = {}  # a dictionary with keys as word that occurs in at least 5 messages and values as index
    i = 0
    for ele in dictionary_all.keys():
        if dictionary_all[ele] >= 5:
            dictionary_fre[ele] = i
            i += 1
    inv_dic = {v: k for k, v in dictionary_fre.items()}

    return inv_dic
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    transform_matrix = np.zeros((len(messages), len(word_dictionary)))

    for i in range(0, len(messages)):
        words = get_words(messages[i])
        for w in words:
            if w in word_dictionary.values():
                j = [key for (key, value) in word_dictionary.items() if value == w]  # find the right j
                transform_matrix[i, j] += 1
        i += 1
    return transform_matrix
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # *** START CODE HERE ***
    V = matrix.shape[1]
    matrix0 = matrix[labels == 0, :]
    matrix1 = matrix[labels == 1, :]
    d_i_0 = matrix0.sum(axis=1)   # length of i'th non spam message
    d_i_1 = matrix1.sum(axis=1)   # length of i'th spam message
    phi_y_0 = (1 + matrix0.sum(axis=0)) / (V + np.sum(d_i_0))
    phi_y_1 = (1 + matrix1.sum(axis=0)) / (V + np.sum(d_i_1))
    phi_y = matrix1.shape[0] / matrix.shape[0]
    state = [phi_y_0, phi_y_1, phi_y]
    return state
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predictions from the model
    """
    # *** START CODE HERE ***
    phi_y_0 = model[0]
    phi_y_1 = model[1]
    phi_y1 = model[2]   # fraction of spam messages
    phi_y0 = 1 - phi_y1  # fraction of non spam messages
    log_phi_y_0 = np.sum(np.log(phi_y_0) * matrix, axis=1)
    log_phi_y_1 = np.sum(np.log(phi_y_1) * matrix, axis=1)

    pred = np.zeros((matrix.shape[0]))
    ratio = np.exp(log_phi_y_0 + np.log(phi_y0) - log_phi_y_1 - np.log(phi_y1))
    probs = 1 / (1 + ratio)
    pred[probs > 0.5] = 1
    pred[probs < 0.5] = 0
    return pred
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_y_0 = model[0]
    phi_y_1 = model[1]
    metric = np.log(phi_y_1/phi_y_0)
    order = np.flip(np.argsort(metric))[0:5]
    top_five = []
    for o in order:
        top_five.append(dictionary[o])
    return top_five
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    outputs = {}
    i = 0
    for radius in radius_to_consider:
        output = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(output == val_labels)
        outputs[radius] = accuracy
    best_radius = max(outputs, key=outputs.get)
    return best_radius
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
