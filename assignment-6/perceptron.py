# ---------------------------------------------#
# -------| Written By: Sibt ul Hussain |-------#
# ---------------------------------------------#

# A Perceptron algorithm with regularized weights...

from classifier import *
#Note: Here the bias term is considered as the last added feature

# Note: Here the bias term is being considered as the last added feature
class Perceptron(Classifier):
    ''' Implements the Perceptron inherited from Classifier For Classification... '''

    def __init__(self, lembda=0):
        """
            lembda= Regularization parameter...
        """

        Classifier.__init__(self, lembda)

        pass

    def hypothesis(self, X, theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        # Your Code here
        return np.dot(X,theta)
    def cost_function(self, X, Y, theta):
        '''
            Computes the Cost function for given input data (X) and labels (Y).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
                
            Return:
                Returns the cost of hypothesis with input parameters 
        '''

        
        # Your Code here

        temp=-Y*self.hypothesis(X,theta)
        temp[temp<0]=0;
        t=theta
        t[-1]=0
        return np.mean(temp)+(self.lembda*0.5)*np.sum(np.square(t))


    def derivative_cost_function(self, X, Y, theta):
        '''
            Computes the derivates of Cost function w.r.t input parameters (thetas)  
            for given input and labels.

            Input:
            ------
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix of inputs
                theata: must  d X 1-dimensional vector for representing vectors
                Y: Must be n X 1-dimensional label vector
            Returns:
            ------
                partial_thetas: a d X 1-dimensional vector of partial derivatives of cost function w.r.t parameters..
        '''
        
        # Your Code here
        temp=-Y*X
        index=Y*self.hypothesis(X,theta)
        ind=np.argwhere(index[:,0]>=0)
        #print "Check 0: ",ind
        temp[ind[:,0]]=0
        #print "Check 1: ", np.shape(temp)
        temp=np.mean(temp,axis=0)
        #print "Check 2: ", np.shape(temp)
        temp=temp.reshape(theta.shape[0],1)
        #print "Check 3: ", np.shape(temp)
        t=np.copy(theta)
        t[-1]=0
        #print "Check 4: ", np.shape(temp+self.lembda*t)
        return temp+self.lembda*t

    def train(self, X, Y, optimizer):
        ''' Train classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            optimizer: an object of Optimizer class, used to find
                       find best set of parameters by calling its
                       gradient descent method...
            Returns:
            -----------
            Nothing
            '''

        nexamples, nfeatures = X.shape
        # Your Code here
        self.theta = optimizer.gradient_descent(X,Y,self.cost_function,self.derivative_cost_function)
    def predict(self, X):
        """
        Test the trained perceptron classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
           
            Returns:
            -----------
            pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """

        # Your Code here
        return self.hypothesis(X,self.theta)