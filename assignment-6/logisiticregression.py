#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#

# A Logistic Regression algorithm with regularized weights...

from classifier import *

#Note: Here the bias term is considered as the last added feature 

class LogisticRegression(Classifier):
    ''' Implements the LogisticRegression For Classification... '''
    def __init__(self, lembda=0.001):        
        """
            lembda= Regularization parameter...            
        """
        Classifier.__init__(self,lembda)                
        
        pass
    def sigmoid(self,z):
        """
            Compute the sigmoid function 
            Input:
                z can be a scalar or a matrix
            Returns:
                sigmoid of the input variable z
        """

        # Your Code here
        z=np.asarray(z,dtype=float)
        return 1 / (1 + np.exp(- z))
    
    
    def hypothesis(self, X,theta):
        '''
            Computes the hypothesis for over given input examples (X) and parameters (thetas).

            Input:
                X: can be either a single n X d-dimensional vector or n X d dimensional matrix
                theta: Must be a d-dimensional vector
            Return:
                The computed hypothesis
        '''
        
        # Your Code here
        return self.sigmoid(np.dot(X, theta))


        
    def cost_function(self, X,Y, theta):
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
        #m = np.shape(X)[0]
        #p_1 = self.hypothesis(X, theta)
        #log_l = (-Y)*np.log(p_1) - (1-Y)*np.log(1-p_1) 
        #without_reg = 1/float(m)*np.sum(log_l)
        #reg = self.lembda/2.0 * np.sum(np.power(theta,2))
        #return without_reg + reg
        
        p_1 = self.hypothesis(X, theta)
        log_l = (-Y)*np.log(p_1) - (1-Y)*np.log(1-p_1) 

        return log_l.mean()

    def derivative_cost_function(self,X,Y,theta):
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
        
        # Your Code here6
        #m = np.shape(X)[0]
        #without_reg = 1/float(m) * np.sum((self.hypothesis(X,theta)-Y)*X)
        #reg = self.lembda * theta
        #return without_reg + reg
        
        res = ((self.hypothesis(X, theta) - Y) * X)
        dthetas = np.mean(res, axis=0)
        return dthetas

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
        
        # Your Code here 
        # Use optimizer here
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
        
        num_test = X.shape[0]
        
        
        # Your Code here
        return self.hypothesis(X,self.theta)
