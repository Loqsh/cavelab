import tensorflow as tf
import math
import numpy as np

def soft_cross_entropy(prediction, label, n_classes=2):
    flat_labels = construct_label(label, n_classes=n_classes)
    flat_logits = tf.reshape(tensor=prediction, shape=(-1, n_classes))

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                              labels=flat_labels)
    cross_entropy_sum = tf.reduce_sum(cross_entropies)
    return cross_entropy_sum, flat_labels

def construct_label(label, n_classes=2):
    if n_classes==2:
        label = tf.to_float(tf.not_equal(label, 0))

    classes = []
    for i in range(n_classes):
        classes.append(tf.to_float(tf.equal(label, i)))

    combined_mask = tf.stack(classes, axis = 3)

    flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, n_classes))
    return flat_labels

def precision_and_recall(label, pred, threshold = 0.5):

    pred = pred > threshold
    label = label > 0.5

    tp = np.mean(np.multiply(label, pred))
    fp = np.mean(np.multiply(1-label, pred))
    fn = np.mean(np.multiply(label, 1-pred))
    tn = np.mean(np.multiply(1-label, 1-pred))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    if math.isnan(precision):
        precision = 0
    if math.isnan(recall):
        recall = 0

    return precision, recall

#Under Development
def roi_curve(label, pred):
    samples = 100
    prec, rec = np.zeros((samples)), np.zeros((samples))

    for i in range(samples):
        prec[i], rec[i] = precision_and_recall(label, pred, threshol=i/float(samples))
