import numpy as np
import matplotlib.pyplot as plt
import warnings

from src import Models

# Set style for plots as latex style
plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=18)
plt.rc('lines', markersize=10)

warnings.filterwarnings('ignore')


def accuracy(y, y_pred):
    if len(y) != len(y_pred):
        raise ValueError('y and y_pred must have the same length')
    return np.sum(y == y_pred) / len(y)


def precision(y, y_pred):
    if np.sum(y_pred) == 0:
        return 0
    tp = np.sum(y * y_pred)
    fp = np.sum((1 - y) * y_pred)
    return tp / (tp + fp)


def recall(y, y_pred):
    if len(y) != len(y_pred):
        raise ValueError("y and y_pred must have the same length")
    if len(y) == 0:
        raise ValueError("y and y_pred must have at least 1 element")
    tp = np.sum(y * y_pred)
    fn = np.sum(y * (1 - y_pred))
    return tp / (tp + fn)


def f1_score(y, y_pred):
    p = precision(y, y_pred)
    r = recall(y, y_pred)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


def classification_summary(y, y_pred):
    from sklearn.metrics import classification_report
    print('Classification report:')
    print(classification_report(y, y_pred))


def roc_curve(y, y_pred, save=True, save_to=None):
    if len(y) != len(y_pred):
        raise ValueError("y and y_pred must have the same length")
    if not all([i in [0, 1] for i in y]):
        raise ValueError("y must only contain 0s and 1s")
    if not all([i in [0, 1] for i in y_pred]):
        raise ValueError("y_pred must only contain 0s and 1s")
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='black', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save:
        if save_to is None:
            raise ValueError("save_to must be specified")
        plt.savefig(save_to)
    plt.show()


def precision_recall_curve(y, y_pred, save=True, save_to=None):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y, y_pred)
    average_precision = average_precision_score(y, y_pred)
    plt.figure(figsize=(8, 8))
    plt.step(recall, precision, color='black', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    if save:
        if save_to is None:
            raise ValueError("save_to must be specified")
        plt.savefig(save_to)
    plt.show()


def confusion_matrix(y, y_pred, save=True, save_to=None):
    if len(y) != len(y_pred):
        raise ValueError("y and y_pred must have the same length")
    if len(y) == 0:
        raise ValueError("y and y_pred must have at least 1 element")
    if not all([i in [0, 1] for i in y]):
        raise ValueError("y must only contain 0s and 1s")
    if not all([i in [0, 1] for i in y_pred]):
        raise ValueError("y_pred must only contain 0s and 1s")
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    tp = np.sum((y == 1) & (y_pred == 1))
    cm = np.array([[tn, fp], [fn, tp]])
    print('Confusion matrix:')
    print(cm)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.matshow(cm, cmap=plt.cm.Greys, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    plt.tick_params(labelsize=15)
    if save:
        if save_to is None:
            raise ValueError("save_to must be specified")
        plt.savefig(save_to)
    plt.show()


def loss_curve(losses, save=True, save_to=None):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(losses, label='Loss', linewidth=1, color='black')
        plt.title('Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.tick_params(labelsize=15)
        ax.legend(loc='upper right')
        if save:
            if save_to is None:
                raise ValueError("save_to must be specified")
            plt.savefig(save_to)
        plt.show()
    except Exception as e:
        print("Error: ", e)


def accuracy_curve(accuracies, save=True, save_to=None):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(accuracies, label='Accuracy', linewidth=1, color='black')
        plt.title('Accuracy')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.tick_params(labelsize=15)
        ax.legend(loc='lower right')
        if save:
            if save_to is None:
                raise ValueError("save_to must be specified")
            plt.savefig(save_to)
        plt.show()
    except Exception as e:
        print("Error: ", e)


def learning_curve_lr(X_train, y_train, X_test, y_test, learning_rate, max_iter, save=True, save_to=None):
    train_score = []
    cross_val_score = []
    size_set = 10
    for i in range(1, size_set + 1):
        # Get the training data
        X_train_ = X_train[:int(i * X_train.shape[0] / size_set)]
        y_train_ = y_train[:int(i * y_train.shape[0] / size_set)]
        # Train the model
        model = Models.LogisticRegression(learning_rate=learning_rate, max_iter=max_iter)
        model.fit(X_train_, y_train_)
        # Get the training score
        train_score.append(accuracy(y_train_, model.predict(X_train_)))
        # Get the cross validation score
        cross_val_score.append(accuracy(y_test, model.predict(X_test)))
    # Plot the learning curve
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title('Learning curve of Logistic Regression model')
    plt.plot(train_score, label='Training score', linewidth=1, color='blue', marker='v', markersize=10)
    plt.fill_between(range(len(train_score)), np.array(train_score) - np.std(train_score),
                     np.array(train_score) + np.std(train_score), alpha=0.1, color='blue')
    plt.plot(cross_val_score, label='Cross validation score', linewidth=1, color='green', marker='o', linestyle='--',
             markersize=10)
    plt.fill_between(range(len(cross_val_score)), np.array(cross_val_score) - np.std(cross_val_score),
                     np.array(cross_val_score) + np.std(cross_val_score), alpha=0.1, color='green')
    plt.xlabel('Percentage of training set')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.05)
    plt.yticks(np.arange(0.5, 1.05, 0.1))
    plt.xticks(range(len(train_score)), [str(int(i * 100 / size_set)) + '%' for i in range(1, size_set + 1)])
    plt.legend()
    if save:
        if save_to is None:
            raise ValueError("save_to must be specified")
        plt.savefig(save_to)
    plt.grid()
    plt.show()


def learning_curve_lda(X_train, y_train, X_test, y_test, save=True, save_to=None):
    train_score = []
    cross_val_score = []
    size_set = 10
    for i in range(1, size_set + 1):
        # Get the training data
        X_train_ = X_train[:int(i * X_train.shape[0] / size_set)]
        y_train_ = y_train[:int(i * y_train.shape[0] / size_set)]
        # Train the model
        model = Models.LDA()
        model.fit(X_train_, y_train_)
        # Get the training score
        train_score.append(accuracy(y_train_, model.predict(X_train_)))
        # Get the cross validation score
        cross_val_score.append(accuracy(y_test, model.predict(X_test)))
    # Plot the learning curve
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title('Learning curve of Linear Discriminant Analysis model')
    plt.plot(train_score, label='Training score', linewidth=1, color='blue', marker='v', markersize=10)
    plt.fill_between(range(len(train_score)), np.array(train_score) - np.std(train_score),
                     np.array(train_score) + np.std(train_score), alpha=0.1, color='blue')
    plt.plot(cross_val_score, label='Cross validation score', linewidth=1, color='green', marker='o', linestyle='--',
             markersize=10)
    plt.fill_between(range(len(cross_val_score)), np.array(cross_val_score) - np.std(cross_val_score),
                     np.array(cross_val_score) + np.std(cross_val_score), alpha=0.1, color='green')
    plt.xlabel('Percentage of training set')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.05)
    plt.yticks(np.arange(0.5, 1.05, 0.1))
    plt.xticks(range(len(train_score)), [str(int(i * 100 / size_set)) + '%' for i in range(1, size_set + 1)])
    plt.legend()
    plt.grid()
    if save:
        if save_to is None:
            raise ValueError("save_to must be specified")
        plt.savefig(save_to)
    plt.show()


def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    return auc(fpr, tpr)
