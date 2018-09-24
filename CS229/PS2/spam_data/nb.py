import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    
    tokens = fd.readline().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    N = float(matrix.shape[1])
    ###################
    
    spam = [i for i,label in enumerate(category) if label==1]
    notspam = [i for i,label in enumerate(category) if label==0]

    spam_prior = np.sum(category[spam])/float(category.shape[0])
    notspam_prior = 1.0 - spam_prior
    
    spam_freq = np.sum(matrix[spam], axis=0) + 1
    notspam_freq = np.sum(matrix[notspam], axis=0) + 1

    spam_words = np.sum(np.sum(matrix[spam], axis=1))
    notspam_words = np.sum(np.sum(matrix[notspam], axis=1))

    spam_like = spam_freq * 1. / (spam_words + N)
    notspam_like = notspam_freq * 1. / (notspam_words + N)
    
    state = np.vstack([np.hstack([spam_like, spam_prior]), np.hstack([notspam_like, notspam_prior])])
    
    state = np.log(state.T)
  
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    post=matrix.dot(state[:-1,:])+state[-1,:]
    output = post[:,0]>post[:,1]
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
