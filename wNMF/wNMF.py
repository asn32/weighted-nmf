'''
Non-negative matrix factorization (NMF) using the "Weighted-NMF" algorithim (wNMF).
See "Weighted Nonnegative Matrix Factorization and Face Feature Extraction", Blondel, Ho and Dooren 2007

NMF decomposes a matrix X into two matrices U,V with a shared internal dimension, representing a reduced-dimension
latent space.

X = UV

Columns of U are the basis vectors for this latent space, and columns of V contain the set of coeffcients required
to represent each sample in A as a linear combination of the basis vectors in U. 

Weighted NMF: 
    Blondel, Ho and Dooren introduce a weight matrix W that pre-weights the importance of each feature (row) in
    each sample (column) of the data matrix X, such that W ⊗ X = UV, where ⊗ is the Hadamard product of W and X. 
    To determine U and V, given W and X, the authors develop a variation of the Multiplicative Update algorithim 
    proposed by (Lee, 1999) and (Lee, 2001) to minimize the Kubllback-Leibler divergence, or, 
    alternatively the Frobenius Norm. Variants of algorithims to solve the weighted-NMF problem by minimizing both
    KL-divergence and the Frobenius Norm are provided. See, reference.
    
'''
import numpy as np

class wNMF:
    '''
    Params
    ----------
    X : numpy.ndarray or coercible array-like object, float64
        A data matrix to be factorized, with dimensions (n_features, n_samples). 
        NOTE this is different from the SKLearn NMF function, which expects X to be (n_samples,n_features)
    
    W : numpy.ndarray or coercible array-like object, float64
        A weight matrix of same dimension as X, which weights each entry in X. Generally expected
        to be values ranging from 0 to 1, but can contain any non-negative entries.
    
    n_components : int
        The rank of the decomposition of X, alternatively the reduced dimension of the factorization.
        
    init : str --> ("random" , None) default "random"
        The initialization strategy for matrices U and V. Defaults to "random" if no value is provided
    
    beta_loss : string --> ("frobenius", "kullback-leibler") default "frobenius"
        The error to be minimized between W ⊗ X, and UV, using the approrite multiplicative update variant. 
        
    max_iter : int
        The maximum number of minimization iterations to perform before stopping.
    
    tol : float, default 1e-4
        If the relative error is found to change less than this amount after 20 iterations, or alternativley increase
        then minimization is completed.
        
    random_state : int default 12345
        Specifies a seed value to initilaize the numpy random number generator. Defaults to 12345
    
    rescale : bool, default False
        Controls whether to normalize the resulting U matrix columns such that each basis vector can be interpreted
        as a categorical probability distribution over the features in X. Useful for Signature Extraction, but invalidates
        the coefficients in V.
    
    track_error : bool, default False
        Controls whether to track the error of each wNMF fitting run, and store the result as a vector of length max_iter.
        One vector is generated per run and tracks the performace of that fitting run over time. By default this is false
        as it can slow down the overall fitting, and is primarily useful for diagnostics
    
    verbose : integer --> (0, 1) default 1
        The level of verbosity. If 1 is provided, reports n_features, n_samples, and n_components, as well as the current
        error every 100 iterations. 
        
   
    Returns
    -------
    
    U : numpy.ndarray, shape (n_features, n_components)
        The basis matrix for the reduced dimension latent space. Columns of U are basis vectors that can be 
        added with different weights to yield a sample from X (columns).

    V : numpy.ndarray, shape (n_components, n_samples)
        The coefficent matrix for the reduced dimension latent space. Columns of V are the reduced representation of
        each sample in X, decomposed into a linear combination of the basis vectors in U. Samples in X can be 'reconstructed'
        by multiplying a column of U by V. 
    
    reconstruction_error : float
        The reconstruction error between X, and W ⊗ UV, using the approriate error function specified in beta_loss
    
    n_iter : int
        The number of iterations at which the minimization terminated, maximal value is max_iter.
    

    This information can be accessed from the following variables, to mimic the SKlearn API
    
        U : self.components_
        |   The matrix U from the best run, with dimensions (n_features, n_components)
        |
        | : self.components_all_
        |    A tuple of length n_runs, with each entry containing a matrix U from a single run.
          
          
        V : self.coefficents_
        |    The matrix V from the best  run, with dimensions (n_features, n_components)
        |   
        | : self.coefficients_all_
        |    A tuple of length n_runs, with each entry containing a matrix V from a single run.
          
        reconstruction_error : self.reconstruction_err_
                        |       The reconstruction error from the best run, a float.
                        |
                        |    : self.reconstruction_err_all_
                        |         A tuple of length n_runs, with each entry containing a the reconstruction error from a single run
        
        n_iter : self.n_iter_
           |      The number of iterations at which the minimization terminated for the best fitting run
           |      
           |   : self.n_iter_all_
           |      A tuple of length n_runs, with each entry containing the number of iterations at which minimization terminated for a single run
        
        
    But can also be accessed more directly using what you would expect the variables to be named
        
        U : self.U
        |   The matrix U from the best run, with dimensions (n_features, n_components)
        |
        | : self.U_all
        |    A tuple of length n_runs, with each entry containing a matrix U from a single run.
          
          
        V : self.V
        |    The matrix V from the best  run, with dimensions (n_features, n_components)
        |   
        | : self.V_all
        |    A tuple of length n_runs, with each entry containing a matrix V from a single run.
          
        reconstruction_error : self.err
                        |       The reconstruction error from the best run, a float.
                        |
                        |    : self.err_all
                        |         A tuple of length n_runs, with each entry containing a the reconstruction error from a single run
        
        n_iter : self.n_iter
           |      The number of iterations at which the minimization terminated for the best fitting run
           |      
           |   : self.n_iter_all
           |      A tuple of length n_runs, with each entry containing the number of iterations at which minimization terminated for a single run
        
        
    Methods
    -------
    A set of methods that reflect the SKlearn model API (fit, fit_transform) are implemented. 
    
    fit(X,W,n_run,...): 
        description: Fits an NMF model for the data X, and the weight matrix W
        requires: X,W,n_run
        returns: self - the wNMF object with access to all the return variables listed above
    
    fit_transfrom(X,W,n_run,...):
        description: Fits an NMF model for the data X, weight matrix W, and returns the coefficient matrix V.
        requires: X,W,n_run
        returns: self.coefficents_  - specifically the best version of V (lowest self.err) identified in n_run's
        
    The other two methods, transform, and inverse transform do not make sense in the context of wNMF, as the
    NMF model is fit with a specific weight matrix, and transforming the data with another weight matrix would not
    be applicable. Hence, these methods are not implemented at the moment 
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> W = np.array([0.8,0.4],[0.1,0.1],[1,1],[0.7,0.3],[0.9,1],[0.01,0.04])
    >>> from weighted-nmf import wNMF
    >>> model = wNMF(n_components=3).fit(X,W,n_run=1)
    >>> V = model.V
    >>> V = model.coefficents_
    >>> U = model.U
    >>> U = model.components_
    >>> iterations_to_convergence = model.n_iters_
    >>> final_error = model.reconst_err_
    >>> ## Accessing all matrices in n_run runs
    >>> V_all = model.V_all
    >>> V_all = model.coefficients_all_
    >>> U_all = model.U_all
    >>> U_all = model.components_all_
    
    References
    ----------
    Blondel, Vincent & Ho, Ngoc-Diep & Van Dooren, Paul. (2007). 
    Weighted Nonnegative Matrix Factorization and Face Feature Extraction. 
    Image and Vision Computing - IVC. 
    
    '''

    def __init__(self,n_components,init='random',beta_loss='frobenius',
                 max_iter=1000,tol=1e-4,random_state=12345,
                 rescale=False,track_error=False,verbose=1):
        
        ## init variables
        self.n_components=n_components
        self.init=init
        self.beta_loss=beta_loss
        self.max_iter=max_iter
        self.tol=tol
        self.random_state=random_state
        self.rescale=rescale
        self.track_error = track_error
        self.verbose=verbose
        
        ## Return Variables
        self.X= None
        
        ## Components / U
        self.components_=None
        self.U = None
        self.components_all_=tuple()
        self.U_all = tuple()
        
        ## Coefficents / V
        self.coefficients_=None
        self.V = None
        self.coefficients_all_=tuple()
        self.V_all = tuple()
        
        ## Reconstruction error / reconst_err_
        self.reconstruction_err_=None
        self.err=None
        self.reconstruction_err_all_=tuple()
        self.err_all = tuple()
        
        self.err_stored=list()
        
        ## n_iters
        self.n_iter_=None
        self.n_iter_all_=tuple()
        self.n_iter=None
        self.n_iter_all=tuple()
        
        ## run check
        self._check_init()
    
    def __repr__(self):
        return f"wNMF Model with {self.n_components} Components"
        
    def _check_init(self):
        '''
        Function to check the values supplied during wNMF initialization for various run parameters. 
        
        Parameters checked : expected values
            n_components  : int, greater than 0
                    init  : string, 'random' | no other initialization strategies allowed at present
                beta_loss : string, ('kullback-leibler','frobenius')
                 max_iter : int, greater than 0
                      tol : float, greater than 0
             random_state : int greater than or equal to zero
                  rescale : boolean
              track_error : boolean
                  verbose : int, (0, 1)
        
        '''
        ## check n_components is int > 0
        if not isinstance(self.n_components,int) or self.n_components <=0:
            raise ValueError(f"Number of components must be a positive integer greater than zero; got '{self.n_components}', of type {type(self.n_components)}")
            
        ## check init is random
        if self.init != 'random':
            raise ValueError(f"Only random initialization is supported; got '{self.init}' of type {type(self.init)}")
        
        ## check beta_loss is frobenius or kullback-leiblier
        if self.beta_loss not in ['kullback-leibler','frobenius']:
            raise ValueError(f"Selected loss must be either 'frobenius' or 'kullback-leibler'; got '{self.beta_loss}'")
        
        ## check max_iter is int > 0
        if not isinstance(self.max_iter,int) or self.max_iter <=0:
            raise ValueError(f"Number of iterations must be a positive integer greater than zero; got '{self.max_iter}', of type {type(self.max_iter)}")
        
        ## check tol is numeric > 0
        if not isinstance(self.tol,float) or self.tol <=0:
            raise ValueError(f"Error convergence criteria must be a positive float greater than zero; got '{self.tol}', of type {type(self.tol)}")
        
        ## check random_state is int > 0
        if not isinstance(self.random_state,int) or self.random_state <0:
            raise ValueError(f"Random state seed must be a positive integer, or zero; got '{self.random_state}', of type {type(self.random_state)}")
        
        ## check rescale is boolean
        if not isinstance(self.rescale,bool):
            raise ValueError(f"rescale must be a boolean; got '{self.rescale}', of type {type(self.rescale)}")
        
        ## check track_error is boolean
        if not isinstance(self.track_error,bool):
            raise ValueError(f"rescale must be a boolean; got '{self.track_error}', of type {type(self.track_error)}")
        
        ## check verbose is int 
        if self.verbose !=0 and self.verbose !=1:
            raise ValueError(f"Verbosity is specified with an it, 0 or 1; got '{self.verbose}', of type {type(self.verbose)}")
        
        
    def fit(self,X,W,n_run=1):
        '''
        Function to fit a wNMF model to X, given weight matrix W. The fitting procedure utilizes a modified
        multiplicative update algoithim (see reference), and is repeated n_run times. It is recommended to repeat
        the fitting procedure multiple times (at least 100) and take the best solution (with the lowest error), or
        alternatively to cluster multiple runs together. 
        
        The algorithim is roughly as follows:
        1) Initialize matrices U (n_features,n_components) and V(n_components,n_samples) with random entries
            scaled approximately to the mean of X divded by n_components
        2) For each iteration, successively update U, then V using the aformentioned multiplicative update steps
        3) Terminate the iterations of the number exceeds max_iter, or if error does not change within tol
        4) Repeat 1-3 n_run times and select the best run, but store all runs. 
        
        Params
        -------
        X : numpy.ndarray or coercible array-like object
            A data matrix to be factorized, with dimensions (n_features, n_samples). 
            NOTE this is differnt from the SKLearn API, which expects X to be (n_samples,n_features)

        W : numpy.ndarray or coercible array-like object
            A weight matrix of same dimension as X, which weights each entry in X. Generally expected
            to be values ranging from 0 to 1, but can contain any non-negative entries.
            
        n_run : int
            The number of times to repeat the wNMF fitting process on the data matrix X and weight matrix W,
            where each attempt utilizes a unique random initialization. The best solution is then selected and
            returned.
        
        Returns:
        -------
        self, with added variables ------
        
        SKLearn response API variables:
            self.components_, 
            self.coefficents_, 
            self.n_iters_, 
            self.reconst_err_
        
        Normal variables:
            self.U
            self.V
            self.n_iters
            self.err
            
        And lists containing all values for all runs
            self.components_all_  / self.U_all
            self.coefficents_all_ / self.V_all
            self.n_iters_all_     / self.n_iters_all
            self.reconst_err_all_ / self.err_all
            
        And the error tracker, if enabled
            self.error_tracker
        
        '''
        
        ## Set the minimal value (that masks 0's) to be the smallest
        ## step size for the data-type in matrix X.
        self.epsmin = np.finfo(type(X[0,0])).eps

        
        ## Try to coerce X and W to numpy arrays
        X = self.coerce(X)
        W = self.coerce(W)
        
        ## Check X and W are suitable for NMF
        self._check_x_w(X,W)
        
        ## If passes, initialize random number generator using random_state
        rng = self.init_random_generator()
    
        ## Extract relevant information from X
        n_features, n_samples = X.shape
        mean = np.mean(X)
        
        ## Initialize result storage
        result = list()
        
        ## Begin Runs...
        for r in range(0,n_run):
    
            if self.verbose==1:
                print(f"Beginning Run {r+1}...")

            ## Generate random initializatoins of U,V using random number generator
            if self.verbose==1:
                print("|--- Initializing U,V")
            U,V = self.initialize_u_v(rng,n_features,n_samples,mean)
           
            ## Factorize X into U,V given W
            if self.verbose==1:
                print("|--- Running wNMF")
                
            if self.beta_loss == 'frobenius':
                factorized = self.weighted_euclidean(X,U,V,W)
                
            elif self.beta_loss == 'kullback-leibler':
                factorized = self.weighted_kullback_leibler(X,U,V,W)
            
            ## Rescale the columns of U (basis vectors) if needed
            if self.rescale==True:
                if self.verbose==1:
                    print("|--- Rescaling U basis vectors")
                    
                factorized[0] = factorized[0]/np.sum(factorized[0],0)
            
            ## append the result and store it
            result.append(factorized)
            
            if self.verbose==1:
                print("|--- Completed")
        
        ## transform the result from a list of tuples to a set of lists, each with multiple individual entries
        result = list(zip(*result))
    
        ## Implementing the SKLearn model response API
        self.U_all=result[0]
        self.V_all=result[1]
        self.n_iter_all=result[2]
        self.err_all=result[3]
        
        ## if tracking errors, set variable to store tracked errors
        if self.track_error:
            self.error_tracker = result[4]
        
        ## setting up lists
        self.components_all_= self.U_all
        self.coefficients_all_ = self.V_all
        self.n_iter_all_ = self.n_iter_all
        self.reconstruction_err_all_ = self.err_all
        
        
        ## finding best result
        best_result = np.argmin(self.err_all)
        
        ## Index out the best result, and set variables
        self.U = self.U_all[best_result]
        self.components_=self.U
        
        self.V = self.V_all[best_result]
        self.coefficients_=self.V
        
        self.n_iter=self.n_iter_all[best_result]
        self.n_iter_=self.n_iter
        
        self.err = self.err_all[best_result]
        self.reconstruction_err_ = self.err
        
        ## return entire wNMF object
        return self
       
    
    def fit_transfrom(self,X,W,n_run=1):
        '''
        Implements the fit_transform functionality from the SKlearn model API. Fits an NMF model to the
        data matrix X, and weight matrix W. Determines the best solution U,V over n_run's. The data-matrix
        is then "transformed" into its latent space coefficents given by the matrix V, or coefficents_. 
        
        Params:
        ------
        X : numpy.ndarray or coercible array-like object
            A data matrix to be factorized, with dimensions (n_features, n_samples). 
            NOTE this is differnt from the SKLearn API, which expects X to be (n_samples,n_features)

        W : numpy.ndarray or coercible array-like object
            A weight matrix of same dimension as X, which weights each entry in X. Generally expected
            to be values ranging from 0 to 1, but can contain any non-negative entries.
            
        n_run : int
            The number of times to repeat the wNMF fitting process on the data matrix X and weight matrix W,
            where each attempt utilizes a unique random initialization. The best solution is then selected and
            returned.
        
        
        Returns:
        ------
        f.coefficents : numpy.ndarray
            The best fit matrix V, or coefficients_ in SKlearn API language 
        
        '''
        f = self.fit(X,W,n_run=1)
        
        return f.coefficients_
            
    def weighted_euclidean(self,A,U,V,W):
        '''
        Function to perform minimization of the Frobenius / Euclidean norm in the
        weighted-NMF case. 
        
        Params:
        -------
        A : numpy.ndarray, values > 0, (n_features, n_samples)
            Data matrix to be factorized, referred to as X in the main code body, referred to as A here to make it easier to
            read the update steps because the authors Blondel, Ho, Ngoc-Diep and Dooren use A.
            
        U : numpy.ndarray, values > 0, (n_features,n_components)
            U matrix, randomly initialized entries.
            
        V : numpy.ndarray, values > 0 (n_components, n_samples)
            V matrix, randomly initialized entries.
            
        W : numpy.ndarray, values > 0 (n_features, n_samples)
            Weight matrix, weighting importance of each feature in each sample, for all samples in X
            
        
        Returns:
        ------
        U : numpy.ndarray, values > 0, (n_features,n_components)
            Optimized version of the U-matrix
        
        V : numpy.ndarray, values > 0 (n_components, n_samples)
            Optimized version of the V-matrix
            
        i : int
            The iteration at which the minimization procedure terminated
        
        err : float
            The final error between the reconstruction UV and the actual values of W ⊗ X
        
        err_stored : numpy.ndarray 
            A numpy vector containing the estimated reconstruction error at each minimization step
            if self.track_error is True, otherwise an empty array of zeroes. 
        
        '''
        epsmin = self.epsmin
        err_stored = np.zeros(self.max_iter)
        ## Begin iterations until max_iter
        for i in range(0,self.max_iter):
            ## Every 10 iterations conver zeroes to epsmin to prevent divide by zero error
            if i % 10 == 0:
                V[V==0]=epsmin
                U[U==0]=epsmin                    
            
            ## If enabled, track errors using the Euclidean Norm loss function
            if self.track_error:
                err_stored[i] = self.calculate_reconstruction_error(A,U,V,W)

            ## update V
            V = V*( (U.T@(W*A))/((U.T@(W*(U@V)))) )
            ## update U
            U = U*( ((W*A)@V.T)/(((W*(U@V))@V.T)) )

        ## Calculate final reconstruction error
        err = self.calculate_reconstruction_error(A,U,V,W)
        return U,V,i,err,err_stored                
                    
    def weighted_kullback_leibler(self,A,U,V,W):
        '''
        Function to perform minimization of the Kullback-Leibler divergence in the
        weighted-NMF case. 
        
        Params:
        -------
        A : numpy.ndarray, values > 0, (n_features, n_samples)
            Data matrix to be factorized, referred to as X in the main code body, referred to as A here to make it easier to
            read the update steps because the authors Blondel, Ho, Ngoc-Diep and Dooren use A.
            
        U : numpy.ndarray, values > 0, (n_features,n_components)
            U matrix, randomly initialized entries.
            
        V : numpy.ndarray, values > 0 (n_components, n_samples)
            V matrix, randomly initialized entries.
            
        W : numpy.ndarray, values > 0 (n_features, n_samples)
            Weight matrix, weighting importance of each feature in each sample, for all samples in X
            
        
        Returns:
        ------
        U : numpy.ndarray, values > 0, (n_features,n_components)
            Optimized version of the U-matrix
        
        V : numpy.ndarray, values > 0 (n_components, n_samples)
            Optimized version of the V-matrix
            
        i : int
            The iteration at which the minimization procedure terminated
        
        err : float
            The final error between the reconstruction UV and the actual values of W ⊗ X
        
        err_stored : numpy.ndarray 
            A numpy vector containing the estimated reconstruction error at each minimization step
            if self.track_error is True, otherwise an empty array of zeroes. 
        
        '''
        epsmin = self.epsmin
        err_stored = np.zeros(self.max_iter) 
        ## Begin iterations until max_iter
        for i in range(0,self.max_iter):
            ## Every 10 iterations conver zeroes to epsmin to prevent divide by zero error
            if i % 10 == 0:
                V[V==0]=epsmin
                U[U==0]=epsmin
            
            ## If enabled, track errors using KL-divergence loss function
            if self.track_error:
                err_stored[i] = self.calculate_reconstruction_error(A,U,V,W)
            
            ## Update V
            V = ((V)/(U.T@W))*(U.T@((W*A)/(U@V)))
            ## Update U
            U =((U)/(W@V.T))*(((W*A)/(U@V))@V.T)
        
        ## Calculate final reconstruction error
        err = self.calculate_reconstruction_error(A,U,V,W)
        return U,V,i,err,err_stored
    
    def coerce(self,matrix):
        '''
        Function to coerce a matrix like object to a numpy.ndarray or return the array 
        if it is already a numpy array. Used for converting X, W to suitable matrices. 
        Throws an error from numpy if the object provided is not coercible. No guarantees 
        are provided on what the coerced result looks like. Zeroes are also replaced with
        epsmin to prevent potential underflow.
        
        Params:
        -------
        matrix : a numpy.ndarray or any object that can be coerced to an array by numpy.ndarray
            An object that is or can be coerced to a numpy.ndarray
        
        Returns: 
        -------
        matrix : numpy.ndarray
            A coerced verision of the provided matrix
        
        '''
        
        ## test if object is a numpy.ndarray / ndarray
        if str(type(matrix))!="<class 'numpy.ndarray'>":
            matrix = np.array(matrix)
        
        ## Convert 0 entries to epsmin to prevent underflow
        matrix[matrix==0]=self.epsmin
        return matrix
            
            
    def _check_x_w(self,X,W):
        '''
        Function to check the whether supplied X and W are suitable for NMF
        
           Conditions checked : expected values
            X.shape, W.shape  : shapes / dimensions should be equal
                entries in X  : greater than or equal to 0, no NaNs
                entries in W  : greater than or equal to 0, no NaNs
        X.shape, n_components : n_components < n_samples in X

        '''
        ## check X and W are the same shape
        if X.shape != W.shape:
            raise ValueError("Dimensions of X and weight matrix W must be the same")
                    
        ## check if entries of X and W are greater than 0
        if np.all(X>=0) == False:
            raise ValueError("Entries of X must be positive or zero")
                    
        if np.all(W>=0) == False:
            raise ValueError("Entries of W must be positive or zero")
                    
        ## Check for Nans / and halt if there are any
        if np.any(np.isnan(X)):
            raise ValueError("Entries of X must not contain NaN / NA, or missing entries")
                    
        if np.any(np.isnan(W)):
            raise ValueError("Entries of W must not contain NaN / NA, or missing entries")

        ## check to ensure n_components < n_samples 
        if  X.shape[1] < self.n_components: 
            raise ValueError("Number of components cannot be greater than the number of samples (columns) in X")
            
    
    def init_random_generator(self):
        '''
        Function to initialize a numpy random number generator 
        
        Params:
        -------
        seed : random_seed, int, greater than 0
            A random seed to initialize the random number generator. Default is 12345
            
        Returns:
        -------
        rng : numpy.random.RandomState
            A numpy random number generator 
        '''
        ## initialize the numpy random generator with random seed
        rng = np.random.RandomState(self.random_state)
        return rng
    
    def initialize_u_v(self,random_number_generator,n_features,n_samples,mean):
        '''
        Function to randomly initialize U and V. U and V are initialized randomly but scaled to the mean
        of X divided by n_components. 
        
        Params:
        -------
        random_number_generator : numpyp.random.RandomState
            An initialized numpy random number generator with a set seed. 
            
        n_features : int 
            The number of features in X, or rows of X
            
        n_samples : int
            The number of samples in X, or columns of X
        
        mean : float
            Estimated mean over the entire data-set X, used for scaling initilization to approximately 
            similar range
        
        Returns:
        -------
        U : numpy.ndarray
            The matrix U, with randomly initialized entries
            
        V : numpy.ndarray
            The matrix V, with randomly initialized entries
            
        '''
        ## estimate density by partitioning mean across components
        est = np.sqrt(mean/self.n_components)
        
        ## generate entries of U/V using randn, scale by est
        U = est*random_number_generator.randn(n_features,self.n_components)
        np.abs(U,U) ## mutate in-place absolute value
        
        V = est*random_number_generator.randn(self.n_components,n_samples)
        np.abs(V,V) ## mutate in-place absolute value
        
        ## set all zeroes (if there are any) to epsmin
        V[V==0]=self.epsmin
        U[U==0]=self.epsmin                    
                    
        return U,V

    def calculate_reconstruction_error(self,X,U,V,W):
        '''
        Function to calculate the reconstruction error of U,V to X, given W. The function to estimate the
        error is based on the selected loss function, beta_loss
        
        Params:
        ------
        A : numpy.ndarray, values > 0, (n_features, n_samples)
            Data matrix to be factorized / compared to
            
        U : numpy.ndarray, values > 0, (n_features,n_components)
            U matrix
            
        V : numpy.ndarray, values > 0 (n_components, n_samples)
            V matrix
            
        W : numpy.ndarray, values > 0 (n_features, n_samples)
            Weight matrix, weighting importance of each feature in each sample, for all samples in X
            
        Returns:
        ------
        err : the estimated error using the selected loss function
        
        '''
        
        ## Replace zeroes with epsmin to prevent divide by zero / log(0) errors
        V[V==0]=self.epsmin
        U[U==0]=self.epsmin  
        
        ## select loss function and calculate error
        if self.beta_loss=='frobenius':
            rec = X-U@V
            err = 0.5*np.sum(W*rec*rec)
            
        elif self.beta_loss=='kullback-leibler':
            rec = U@V
            err = np.sum(W*(X*np.log(X/rec)-X+rec))
            
        return err
