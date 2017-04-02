import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Number of samples from each class to use in train dataset
nSamplesPerClass = 10
# Array of plots that show the image of samples
figs = []


def getData():
    """ Read the file sayi.dat
        Every sample is in vector format(1 x 64)
        N is the number of samples
        Return X, samples in matrix format (N, 64) as a numpy.array
        Return Y, labels in vector format (N, 1)
    """
    Y = []
    X = []
    for line in open('sayi.dat'):
        row = line.split(',')
        y = int(row[-1])
        x = map(int, row[:-1])
        Y.append(y)
        X.append(x)
    return np.array(X), np.array(Y)


def splitData(X, Y, ndigit):
    """ Split data in to train and test, and samples and labels
        Get nSamplesPerClass samples from each class and combine them in trainX
        The other samples are send into testX
        trainY and test Y are constructed in the same way.
        Return trainX, as numpy.array, (nSamplesPerClass * 10, 64)
        Return trainY, as numpy.array, (nSamplesPerClass * 10, 1)
        Return testX, as numpy.array, (N - (nSamplesPerClass * 10), 64)
        Return testY, as numpy.array, (N - (nSamplesPerClass * 10), 1)
    """
    trainX = np.zeros((0, X.shape[1]))
    trainY = np.array([])
    testX = np.zeros((0, X.shape[1]))
    testY = np.array([])
    for i in xrange(ndigit):
        x = X[Y == i]
        y = Y[Y == i]
        trainX = np.concatenate((trainX, x[0:nSamplesPerClass]))
        trainY = np.concatenate((trainY, y[0:nSamplesPerClass]))
        testX = np.concatenate((testX, x[nSamplesPerClass:]))
        testY = np.concatenate((testY, y[nSamplesPerClass:]))
    return trainX, trainY, testX, testY


def getEigenVectors(X_new, elambda):
    """ Gets Adjusted samples data (n, 64)
        Where n is the number of samples in the X_new
        Calculate covariance of X_new
        Calculate eigenvalues and eigenvectors of covariance of X_new
        Divide eigenvalues by sum of eigenvalues, and sort them
        Get eigenvalues to achieve elamda percent of all eigenvalues
        Return corresponding eigenvectors to the selected eigenvalues
    """
    X_cov = X_new.T.dot(X_new)
    eigenvalues, eigenvectors = np.linalg.eig(X_cov)
    eigen_sorted_ind = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues / eigenvalues.sum()
    eigenvalues_sum = 0
    eigenvectors_selected = eigenvectors
    for i in xrange(len(eigen_sorted_ind)):
        eigenvalues_sum += eigenvalues[eigen_sorted_ind[i]]
        if eigenvalues_sum >= elambda:
            eigenvectors_selected = eigenvectors[:, eigen_sorted_ind[0:i]]
            break
    return eigenvectors_selected


def getNearestSampleIndex(test, trainX):
    """ For a given test sample test,
        Calculate the euclidean distance between test and all samples in trainX
        return the index of the sample, which is nearest to the test, in trainX
    """
    dist_matrix = test - trainX
    dist_square = dist_matrix ** 2
    dist_sums = dist_square.sum(axis=1)
    distance_vector = np.sqrt(dist_sums)
    return (distance_vector).argmin()


def showImage(title, actual, prediction, test, nearest, test_eig, nearest_eig):
    """ Draw the figures of the test sample: test,
        the nearest sample to the test: nearest,
        the recovered test sample from eigen space: test_eig,
        the recovered nearest sample: nearest_eig
    """
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(test, cmap='Greys_r')
    a.set_title('Sample Actual: %d' % actual)
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(nearest, cmap='Greys_r')
    a.set_title('Nearest Prediction: %d' % prediction)
    a = fig.add_subplot(2, 2, 3)
    plt.imshow(test_eig, cmap='Greys_r')
    a.set_title('Sample Recovered')
    a = fig.add_subplot(2, 2, 4)
    plt.imshow(nearest_eig, cmap='Greys_r')
    a.set_title('Nearest Recovered')
    figs.append(fig)


def displaySamples(labels, predictions, samples, nearest, samples_eigen,
                   nearest_eigen, eigenvectors, mean, elambda, predict):
    """ Gets the lists of the test sample and the nearest sample to test
        for each test and nearest pair call showImage function
    """
    for i in xrange(len(labels)):
        actual = labels[i]
        prediction = predictions[i]
        img = samples[i].reshape(8, 8)
        img_nearest = nearest[i].reshape(8, 8)
        img_eig = (samples_eigen[i].dot(eigenvectors.T) + mean).reshape(8, 8)
        img_nearest_eig = (nearest_eigen[i].dot(
            eigenvectors.T) + mean).reshape(8, 8)
        title = "%s Prediction, Lambda: %.2f, # of Eigenvectors : %d" % (
            predict, elambda, eigenvectors.shape[1])
        showImage(title, actual, prediction, img, img_nearest,
                  img_eig, img_nearest_eig)


def test(ndigit, elambda, showSamples, showConfusion):
    """ The main function, where training and testing performed
        ndigit is the number of digits that will be included (2 - 10)
        elambda is the value for selecting eigen vectors (0.4, 0.6, 0.8)
        showSamples is a boolean for drawing sample images
        showConfusion is a boolean for printing the confusion matrices
    """
    Data, Label = getData()
    trainX, trainY, testX, testY = splitData(Data, Label, ndigit)
    trainX_mean = np.mean(trainX, axis=0)
    trainX_new = trainX - trainX_mean
    eigenvectors = getEigenVectors(trainX_new, elambda)
    trainX_eigen = trainX_new.dot(eigenvectors)
    testX_new = testX - trainX_mean
    testX_eigen = testX_new.dot(eigenvectors)
    testO = []
    if showSamples:
        correct_samples = []
        correct_samples_nearest = []
        correct_samples_eigen = []
        correct_samples_nearest_eigen = []
        correct_samples_labels = []
        correct_samples_predictions = []
        wrong_samples = []
        wrong_samples_nearest = []
        wrong_samples_eigen = []
        wrong_samples_nearest_eigen = []
        wrong_samples_labels = []
        wrong_samples_predictions = []
    if showConfusion:
        conf = np.zeros((ndigit, ndigit))
    for i in xrange(testX_eigen.shape[0]):
        t = testX_eigen[i]
        j = getNearestSampleIndex(t, trainX_eigen)
        p = int(trainY[j])
        y = int(testY[i])
        if showConfusion:
            conf[p, y] += 1
        if showSamples:
            if p == y:
                if len(correct_samples) < y + 1:
                    correct_samples.append(testX[i])
                    correct_samples_nearest.append(trainX[j])
                    correct_samples_eigen.append(testX_eigen[i])
                    correct_samples_nearest_eigen.append(trainX_eigen[j])
                    correct_samples_labels.append(y)
                    correct_samples_predictions.append(p)
            else:
                if len(wrong_samples) < y + 1:
                    wrong_samples.append(testX[i])
                    wrong_samples_nearest.append(trainX[j])
                    wrong_samples_eigen.append(testX_eigen[i])
                    wrong_samples_nearest_eigen.append(trainX_eigen[j])
                    wrong_samples_labels.append(y)
                    wrong_samples_predictions.append(p)
        testO.append(p)
    testO = np.array(testO)
    train0 = []
    for i in xrange(trainX_eigen.shape[0]):
        t = trainX_eigen[i]
        j = getNearestSampleIndex(t, trainX_eigen)
        min_class = trainY[j]
        train0.append(min_class)
    train0 = np.array(train0)
    print "for digits = %d lambda = %.2f train = %.6f test = %.6f " % (
        ndigit, elambda, (train0 == trainY).mean(), (testO == testY).mean())
    if showConfusion:
        print conf
    if showSamples:
        displaySamples(correct_samples_labels, correct_samples_predictions,
                       correct_samples, correct_samples_nearest,
                       correct_samples_eigen, correct_samples_nearest_eigen,
                       eigenvectors, trainX_mean, elambda, 'Correct')
        displaySamples(wrong_samples_labels, wrong_samples_predictions,
                       wrong_samples, wrong_samples_nearest,
                       wrong_samples_eigen, wrong_samples_nearest_eigen,
                       eigenvectors, trainX_mean, elambda, 'Wrong')


def test_default(ndigit):
    """ This function is for training and testing
        WITHOUT CONVERTING TO THE EIGEN SPACE
        ndigit is the number of digits that will be included (2 - 10)
    """
    Data, Label = getData()
    trainX, trainY, testX, testY = splitData(Data, Label, ndigit)
    trainX_mean = np.mean(trainX, axis=0)
    trainX_new = trainX - trainX_mean
    trainX_eigen = trainX_new
    testX_new = testX - trainX_mean
    testX_eigen = testX_new
    testO = []
    for i in xrange(testX_eigen.shape[0]):
        t = testX_eigen[i]
        j = getNearestSampleIndex(t, trainX_eigen)
        min_class = trainY[j]
        testO.append(min_class)
    testO = np.array(testO)
    train0 = []
    for i in xrange(trainX_eigen.shape[0]):
        t = testX_eigen[i]
        j = getNearestSampleIndex(t, trainX_eigen)
        min_class = trainY[j]
        train0.append(min_class)
    train0 = np.array(train0)
    print "for digits = %d default       train = %.6f test = %.6f " % (
        ndigit, (train0 == trainY).mean(), (testO == testY).mean())


def pltmulti(filename):
    """ Stores samples images to a file given in the filename parameter
    """
    pp = PdfPages(filename)

    for fig in figs:
        pp.savefig(fig)
    pp.close()
    for fig in figs:
        fig.clear()
    plt.close()


def main(showSamples=True, showConfusion=True):
    """
    ndigit is the number of digits that will be included (2 - 10)
    elambda is the value for selecting eigen vectors (0.4, 0.6, 0.8)
    showSamples is a boolean for drawing sample images
    showConfusion is a boolean for printing the confusion matrices
    graphs.pdf is the file name for storing the images of the samples.
    """
    ndigit = 10
    elambda = [0.4, 0.6, 0.8]
    for i in elambda:
        test(ndigit, i, showSamples, showConfusion)
    if showSamples:
        pltmulti('graphs.pdf')


if __name__ == "__main__":
    main()
