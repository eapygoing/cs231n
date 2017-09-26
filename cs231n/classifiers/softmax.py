import numpy as np
from random import shuffle
#from past.builtins import range


def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    f = np.dot(X[i], W)
    f -= np.max(f)  # soft(x + c) = soft(x)
    f_exp = np.exp(f)
    f_exp_sum = np.sum(f_exp)
    score_right_class = f_exp[y[i]] / f_exp_sum
    loss -= np.log(score_right_class)

    for c in range(num_classes):
      score_c = f_exp[c] / f_exp_sum
      if c == y[i]:
        dW[:, c] += (score_c - 1) * X[i]
      else:
        dW[:, c] += score_c * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  f = np.dot(X, W)
  f_max = np.max(f, axis=1, keepdims=True)
  f -= f_max
  f_exp = np.exp(f)
  f_exp_sum = np.sum(f_exp, axis=1, keepdims=True)
  scores = f_exp / f_exp_sum
  loss += np.sum(-np.log(scores[np.arange(num_train), y])) / num_train
  loss += 0.5 * reg * np.sum(W * W)

  correct_class_index = np.zeros((num_train, num_classes))
  correct_class_index[np.arange(num_train), y] = 1
  dW += np.dot(X.T, scores - correct_class_index)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

