#---------------------------------------------#
#-------| Written By: Sibt ul Hussain |-------#
#---------------------------------------------#


#---------------Instructions------------------#

# You will be writing a super class named WeakLearner
# and then will be implmenting its sub classes
# RandomWeakLearner and LinearWeakLearner. Remember
# all the overridded functions in Python are by default
# virtual functions and every child classes inherits all the
# properties and attributes of parent class.

# Your task is to  override the train and evaluate functions
# of superclass WeakLearner in each of its base classes.
# For this purpose you might have to write the auxiliary functions as well.

#--------------------------------------------------#

# Now, go and look for the missing code sections and fill them.
#-------------------------------------------#


import numpy as np
import scipy.stats as stats
from numpy import inf
import math
from random import randint
from collections import defaultdict, Counter
class WeakLearner: # A simple weaklearner you used in Decision Trees...
    """ A Super class to implement different forms of weak learners...


    """
    def __init__(self):
        """
        Input:


        """
        #print "   "
        pass

    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#
        return score, Xlidx,Xridx

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#

    def evaluate_numerical_attribute(self, feat, Y):
        '''
            Evaluates the numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            feat: a contiuous feature
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''

        classes=np.unique(Y)
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # Same code as you written in DT assignment...

        #---------End of Your Code-------------------------#

        return split,mingain,Xlidx,Xridx

class RandomWeakLearner(WeakLearner):  # Axis Aligned weak learner....
    """ An Inherited class to implement Axis-Aligned weak learner using
        a random set of features from the given set of features...

    """

    def __init__(self,nsplits=+np.inf, nrandfeat=None):
        """
        Input:
            nsplits = How many nsplits to use for each random feature, (if +inf, check all possible splits)
            nrandfeat = number of random features to test for each node (if None, nrandfeat= sqrt(nfeatures) )
        """
        WeakLearner.__init__(self) # calling base class constructor...
        self.nsplits=nsplits
        self.nrandfeat=nrandfeat
        #pass
    def train(self, X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible split points for
            possible feature selection

            Input:
            ---------
            X: a [m x d]  features matrix
            Y: a [m x 1] labels matrix

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        if(not self.nrandfeat):
            self.nrandfeat=int(np.round(np.sqrt(nfeatures)))

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        feat = X[self.nrandfeat]
        classes,class_count = np.unique(Y,return_counts = True)
        nclasses=len(classes)
        sidx=np.argsort(feat)
        f=feat[sidx] # sorted features
        sY=Y[sidx] # sorted features class labels...

        midpoints = []
        for idx  in range(len(f[:-1])):
            if f[idx] != f[idx+1]:
                mid_value = float(f[idx]+f[idx+1])/2
                midpoints.append(mid_value)
        PD = np.inf
        score = 0
        v = 0

        for mid_ in midpoints:
            mask = f < mid_
            Dy = sY[mask]
            Dn = sY[~mask]
            len_Dy = len(Dy)
            len_Dn = len(Dn)
            prob_dy = len_Dy/float(len_Dy + len_Dn)
            prob_dn = len_Dn/float(len_Dy + len_Dn)
            Dy_c,counts_dy_c = np.unique(Dy,return_counts=True)
            PDY = 0
            for _c_dy in counts_dy_c:
                prob_dy_c = _c_dy/float(len_Dy)
                PDY += prob_dy_c * math.log(prob_dy_c,2)
            PDY *= -(prob_dy)
            Dn_c,counts_dn_c = np.unique(Dn,return_counts=True)
            PDN = 0
            for _c_dn in counts_dn_c:
                prob_dn_c = _c_dn/float(len_Dn)
                PDN += prob_dn_c * math.log(prob_dn_c,2)
            PDN *= -(prob_dn)
            if  (PDY+PDN) < PD:
                PD = PDY+PDN
                v = mid_
                score = PD
        Xlidx = np.where(f < v)
        Xridx = np.where(f > v)
        self.split_point = v
        return score, Xlidx, Xridx

        #---------End of Your Code-------------------------#
        #return minscore, bXl,bXr

    def findBestRandomSplit(self,feat,Y):
        """

            Find the best random split by randomly sampling "nsplits"
            splits from the feature range...

            Input:
            ----------
            feat: [n X 1] nexamples with a single feature
            Y: [n X 1] label vector...

        """
        frange=np.max(feat)-np.min(feat)


        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#

        #---------End of Your Code-------------------------#
        return splitvalue, minscore, Xlidx, Xridx

    def calculateEntropy(self, Y, mship):
        """
            calculates the split entropy using Y and mship (logical array) telling which
            child the examples are being split into...

            Input:
            ---------
                Y: a label array
                mship: (logical array) telling which child the examples are being split into, whether
                        each example is assigned to left split or the right one..
            Returns:
            ---------
                entropy: split entropy of the split
        """

        lexam=Y[mship]
        rexam=Y[np.logical_not(mship)]

        pleft= len(lexam) / float(len(Y))
        pright= 1-pleft

        pl= stats.itemfreq(lexam)[:,1] / float(len(lexam)) + np.spacing(1)
        pr= stats.itemfreq(rexam)[:,1] / float(len(rexam)) + np.spacing(1)

        hl= -np.sum(pl*np.log2(pl))
        hr= -np.sum(pr*np.log2(pr))

        sentropy = pleft * hl + pright * hr

        return sentropy


    def evaluate(self, X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # print self.split_point,self.nrandfeat,X[0][1]
        if self.split_point > X[0][self.nrandfeat]:
            return True
        else:
            return False
        #---------End of Your Code-------------------------#




# build a classifier ax+by+c=0
class LinearWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D line based weak learner using
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...

        """
        RandomWeakLearner.__init__(self,nsplits)

        #pass

    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible

            Input:
            ---------
            X: a [m x d] data matrix ...
            Y: labels

            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node

        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        # random_splits=[]
        # for x in range(self.nsplits):
        #     random_splits.append(randint(0,2000))
        dy = 0
        dn = 0
        t_bXl = []
        t_bXr = []
        minscore = np.inf
        split = 0
        current_score = np.inf
        for random_split in range(10):
            a = np.random.uniform(-1,1)
            b = np.random.uniform(-1,1)
            c = np.random.uniform(-1,1)
            for idx,values_x in enumerate(X):
                if values_x[0]*a + values_x[1]*b + c > 0:
                    dy+=1
                    t_bXl.append(idx)
                else:
                    dn+=1
                    t_bXr.append(idx)
            if dy != 0 and dn != 0:
                current_score = -1*((dy/float(dy+dn))*math.log((dy/float(dy+dn)),2) + (dn/float(dy+dn))*math.log((dn/float(dy+dn)),2))
            if current_score < minscore:
                minscore = current_score
                bXl = t_bXl
                bXr = t_bXr
                self.a = a
                self.b = b
                self.c = c
            t_bXl = []
            t_bXr = []

        #---------End of Your Code-------------------------#
        return minscore, bXl, bXr

        
    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """
        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        if X[0][0]*self.a + X[0] [1]*self.b +self.c > 0:
            return True
        else:
            return False
        #---------End of Your Code-------------------------#


#build a classifier a*x^2+b*y^2+c*x*y+ d*x+e*y+f
class ConicWeakLearner(RandomWeakLearner):  # A 2-dimensional linear weak learner....
    """ An Inherited class to implement 2D Conic based weak learner using 
        a random set of features from the given set of features...


    """
    def __init__(self, nsplits=10):
        """
        Input:
            nsplits = How many splits to use for each choosen line set of parameters...
            
        """
        self.nsplits = nsplits
        RandomWeakLearner.__init__(self,nsplits)
        
        pass

    
    def train(self,X, Y):
        '''
            Trains a weak learner from all numerical attribute for all possible 
            
            Input:
            ---------
            X: a [m x d] training matrix...
            Y: labels
            
            Returns:
            ----------
            v: splitting threshold
            score: splitting score
            Xlidx: Index of examples belonging to left child node
            Xridx: Index of examples belonging to right child node
            
        '''
        nexamples,nfeatures=X.shape

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        dy = 0
        dn = 0
        t_bXl = []
        t_bXr = []
        minscore = np.inf
        split = 0
        current_score = np.inf
        for random_split in range(self.nsplits):
            a = np.random.uniform(-1,1)
            b = np.random.uniform(-1,1)
            c = np.random.uniform(-1,1)
            d = np.random.uniform(-1,1)
            e = np.random.uniform(-1,1)
            f = np.random.uniform(-1,1)
            for idx,values_x in enumerate(X):
                #  a*x^2+b*y^2+c*x*y+ d*x+e*y+f
                if (values_x[0]**2)*a + (values_x[1]**2)*b + c*values_x[0]*values_x[1] + d*values_x[0] + e*values_x[1] + f > 0:
                    dy+=1
                    t_bXl.append(idx)
                else:
                    dn+=1
                    t_bXr.append(idx)
            if dy != 0 and dn != 0:
                current_score = -1*((dy/float(dy+dn))*math.log((dy/float(dy+dn)),2) + (dn/float(dy+dn))*math.log((dn/float(dy+dn)),2))
            if current_score < minscore:
                minscore = current_score
                bXl = t_bXl
                bXr = t_bXr
                self.a = a
                self.b = b
                self.c = c
                self.d = d
                self.e = e
                self.f = f
            t_bXl = []
            t_bXr = []
        #---------End of Your Code-------------------------#
        return minscore, bXl, bXr

    def evaluate(self,X):
        """
        Evalute the trained weak learner  on the given example...
        """

        #-----------------------TODO-----------------------#
        #--------Write Your Code Here ---------------------#
        if (X[0][0]**2)*self.a + (X[0][1]**2)*self.b + self.c*X[0][0]*X[0][1] + self.d*X[0][0] + self.e*X[0][1] + self.f > 0:
            return True
        else:
            return False

        #---------End of Your Code-------------------------#