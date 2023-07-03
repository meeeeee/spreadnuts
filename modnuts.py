"""
This program uses the NUTS implementation found at https://github.com/mfouesneau/NUTS largely unmodified except where changes were made to the trajectory-point sampling method and u-turn condition check
It also contains functions that generate mixtures of distributions randomly for use in testing NUTS and the modified NUTS algorithm presented here
"""

import numpy as np
from numpy import identity, log, exp, sqrt
from helpers import progress_range

import pdb

#np.random.seed(0)

__all__ = ['nuts6']

def leapfrog(theta, r, grad, epsilon, f):
    """ Perfom a leapfrog jump in the Hamiltonian space
    INPUTS
    ------
    theta: ndarray[float, ndim=1]
        initial parameter position

    r: ndarray[float, ndim=1]
        initial momentum

    grad: float
        initial gradient value

    epsilon: float
        step size

    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)

    OUTPUTS
    -------
    thetaprime: ndarray[float, ndim=1]
        new parameter position
    rprime: ndarray[float, ndim=1]
        new momentum
    gradprime: float
        new gradient
    logpprime: float
        new lnp
    """
    # make half step in r
    rprime = r + 0.5 * epsilon * grad
    # make new step in theta
    thetaprime = theta + epsilon * rprime
    #compute new gradient
    logpprime, gradprime = f(thetaprime)
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime, rprime, gradprime, logpprime


def find_reasonable_epsilon(theta0, grad0, logp0, f):
    """ Heuristic for choosing an initial value of epsilon """
    epsilon = 1.
    r0 = np.random.normal(0., 1., len(theta0))

    # Figure out what direction we should be moving epsilon.
    _, rprime, gradprime, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
    # brutal! This trick make sure the step is not huge leading to infinite
    # values of the likelihood. This could also help to make sure theta stays
    # within the prior domain (if any)
    k = 1.
    while np.isinf(logpprime) or np.isinf(gradprime).any():
        k *= 0.5
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon * k, f)

    epsilon = 0.5 * k * epsilon

    # acceptprob = np.exp(logpprime - logp0 - 0.5 * (np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
    # a = 2. * float((acceptprob > 0.5)) - 1.
    logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))
    a = 1. if logacceptprob > np.log(0.5) else -1.
    # Keep moving epsilon in that direction until acceptprob crosses 0.5.
    # while ( (acceptprob ** a) > (2. ** (-a))):
    while a * logacceptprob > -a * np.log(2):
        epsilon = epsilon * (2. ** a)
        _, rprime, _, logpprime = leapfrog(theta0, r0, grad0, epsilon, f)
        # acceptprob = np.exp(logpprime - logp0 - 0.5 * ( np.dot(rprime, rprime.T) - np.dot(r0, r0.T)))
        logacceptprob = logpprime-logp0-0.5*(np.dot(rprime, rprime)-np.dot(r0,r0))

    print("find_reasonable_epsilon=", epsilon)

    return epsilon


def stop_criterion(thetaminus, thetaplus, rminus, rplus):
    """ Compute the stop condition in the main loop
    dot(dtheta, rminus) >= 0 & dot(dtheta, rplus >= 0)

    INPUTS
    ------
    thetaminus, thetaplus: ndarray[float, ndim=1]
        under and above position
    rminus, rplus: ndarray[float, ndim=1]
        under and above momentum

    OUTPUTS
    -------
    criterion: bool
        return if the condition is valid
    """
    dtheta = thetaplus - thetaminus
    return (np.dot(dtheta, rminus.T) >= 0) & (np.dot(dtheta, rplus.T) >= 0)


def build_tree(theta, r, grad, logu, v, j, epsilon, f, joint0):
    """The main recursion."""
    if (j == 0):
        # Base case: Take a single leapfrog step in the direction v.
        thetaprime, rprime, gradprime, logpprime = leapfrog(theta, r, grad, v * epsilon, f)
        joint = logpprime - 0.5 * np.dot(rprime, rprime.T)
        # Is the new point in the slice?
        nprime = int(logu < joint)
        # Is the simulation wildly inaccurate?
        sprime = int((logu - 1000.) < joint)
        # Set the return values---minus=plus for all things here, since the
        # "tree" is of depth 0.
        thetaminus = thetaprime[:]
        thetaplus = thetaprime[:]
        rminus = rprime[:]
        rplus = rprime[:]
        gradminus = gradprime[:]
        gradplus = gradprime[:]
        # Compute the acceptance probability.
        alphaprime = min(1., np.exp(joint - joint0))
        #alphaprime = min(1., np.exp(logpprime - 0.5 * np.dot(rprime, rprime.T) - joint0))
        nalphaprime = 1
    else:
        # Recursion: Implicitly build the height j-1 left and right subtrees.
        thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime = build_tree(theta, r, grad, logu, v, j - 1, epsilon, f, joint0)
        # No need to keep going if the stopping criteria were met in the first subtree.
        if (sprime == 1):
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaminus, rminus, gradminus, logu, v, j - 1, epsilon, f, joint0)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2 = build_tree(thetaplus, rplus, gradplus, logu, v, j - 1, epsilon, f, joint0)
            # Choose which subtree to propagate a sample up from.
            if (np.random.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                thetaprime = thetaprime2[:]
                gradprime = gradprime2[:]
                logpprime = logpprime2
            # Update the number of valid points.
            nprime = int(nprime) + int(nprime2)
            # Update the stopping criterion.
            sprime = int(sprime and sprime2 and stop_criterion(thetaminus, thetaplus, rminus, rplus))
            # Update the acceptance probability statistics.
            alphaprime = alphaprime + alphaprime2
            nalphaprime = nalphaprime + nalphaprime2

    return thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime

def nuts6(f, M, Madapt, theta0, delta=0.6, progress=False):
    """
    Implements the No-U-Turn Sampler (NUTS) algorithm 6 from from the NUTS
    paper (Hoffman & Gelman, 2011).
    Runs Madapt steps of burn-in, during which it adapts the step size
    parameter epsilon, then starts generating samples to return.
    Note the initial step size is tricky and not exactly the one from the
    initial paper.  In fact the initial step size could be given by the user in
    order to avoid potential problems
    INPUTS
    ------
    epsilon: float
        step size
        see nuts8 if you want to avoid tuning this parameter
    f: callable
        it should return the log probability and gradient evaluated at theta
        logp, grad = f(theta)
    M: int
        number of samples to generate.
    Madapt: int
        the number of steps of burn-in/how long to run the dual averaging
        algorithm to fit the step size epsilon.
    theta0: ndarray[float, ndim=1]
        initial guess of the parameters.
    KEYWORDS
    --------
    delta: float
        targeted acceptance fraction
    progress: bool
        whether to show progress (requires tqdm module for full functionality)
    OUTPUTS
    -------
    samples: ndarray[float, ndim=2]
    M x D matrix of samples generated by NUTS.
    note: samples[0, :] = theta0
    """

    if len(np.shape(theta0)) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    D = len(theta0)
    samples = np.empty((M + Madapt, D), dtype=float)
    lnprob = np.empty(M + Madapt, dtype=float)

    logp, grad = f(theta0)
    samples[0, :] = theta0
    lnprob[0] = logp

    # Choose a reasonable first epsilon by a simple heuristic.
    epsilon = find_reasonable_epsilon(theta0, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = 1
    Hbar = 0

    for m in progress_range(1, M + Madapt, progress=progress):
        # Resample momenta.
        r0 = np.random.normal(0, 1, D)

        #joint lnp of theta and momentum r
        joint = logp - 0.5 * np.dot(r0, r0.T)

        # Resample u ~ uniform([0, exp(joint)]).
        # Equivalent to (log(u) - joint) ~ exponential(1).
        logu = float(joint - np.random.exponential(1, size=1))

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        lnprob[m] = lnprob[m - 1]

        # initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = r0[:]
        rplus = r0[:]
        gradminus = grad[:]
        gradplus = grad[:]

        j = 0  # initial heigth j = 0
        n = 1  # Initially the only valid point is the initial point.
        s = 1  # Main loop: will keep going until s == 0.

        while (s == 1):
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (np.random.uniform() < _tmp):
                samples[m, :] = thetaprime[:]
                lnprob[m] = logpprime
                logp = logpprime
                grad = gradprime[:]
            if m%1000 == 0:
                print(m)
            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Increment depth.
            j += 1

        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = exp(mu - sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
        else:
            epsilon = epsilonbar
    samples = samples[Madapt:, :]
    lnprob = lnprob[Madapt:]
    return samples, lnprob, epsilon

def nuts7(f, M, Madapt, theta0, delta=0.6, progress=False):
    """
    implements the modified NUTS algorithm described in SpreadNUTS with the same inputs and outputs as the nuts6 function
    """
    
    if len(np.shape(theta0)) > 1:
        raise ValueError('theta0 is expected to be a 1-D array')

    D = len(theta0)
    samples = np.empty((M + Madapt, D), dtype=float)
    sapmles = np.empty((M + Madapt, D), dtype=float)
    lnprob = np.empty(M + Madapt, dtype=float)
    ln_prob = np.empty(M + Madapt, dtype=float)

    logp, grad = f(theta0)
    samples[0, :] = theta0
    lnprob[0] = logp

    # Choose a reasonable first epsilon by a simple heuristic.
    epsilon = find_reasonable_epsilon(theta0, grad, logp, f)

    # Parameters to the dual averaging algorithm.
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = log(10. * epsilon)

    # Initialize dual averaging algorithm.
    epsilonbar = 1
    Hbar = 0

    for m in progress_range(1, M + Madapt, progress=progress):
        # Resample momenta; generates a trajectory and chooses a point from this trajectory
        r0 = np.random.normal(0, 1, D)

        #joint lnp of theta and momentum r
        joint = logp - 0.5 * np.dot(r0, r0.T)

        # Resample u ~ uniform([0, exp(joint)]).
        # Equivalent to (log(u) - joint) ~ exponential(1).
        logu = float(joint - np.random.exponential(1, size=1))

        # if all fails, the next sample will be the previous one
        samples[m, :] = samples[m - 1, :]
        lnprob[m] = lnprob[m - 1]

        # initialize the tree
        thetaminus = samples[m - 1, :]
        thetaplus = samples[m - 1, :]
        rminus = r0[:]
        rplus = r0[:]
        gradminus = grad[:]
        gradplus = grad[:]

        j = 0  # initial height j = 0
        n = 1  # Initially the only valid point is the initial point.
        s = 1  # Main loop: will keep going until s == 0.
        
        thetap_arr = [samples[m-1,:]]
        logp_arr = [lnprob[m-1]]
        cdf = [0.0] # cdf array

        while (s == 1):
            # Choose a direction. -1 = backwards, 1 = forwards.
            v = int(2 * (np.random.uniform() < 0.5) - 1)

            # Double the size of the tree.
            if (v == -1):
                thetaminus, rminus, gradminus, _, _, _, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint)
            else:
                _, _, _, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha = build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint)

            # Use Metropolis-Hastings to decide whether or not to move to a
            # point from the half-tree we just generated.

            #pdb.set_trace()
            _tmp = min(1, float(nprime) / float(n))
            if (sprime == 1) and (np.random.uniform() < _tmp):
                samples[m, :] = thetaprime[:]
                lnprob[m] = logpprime
                logp = logpprime
                grad = gradprime[:]

                thetap_arr.append(thetaprime[:])
                if m%1000 == 0:
                    print(m)#print(thetap_arr)
                logp_arr.append(logpprime)
                cdf.append(cdf[-1])
                sm = np.sum(np.square(thetaprime[:] - samples[:m-1,:]), axis = -1)
                cdf[-1] += np.amin(sm) if len(sm) != 0 else 0
            # make thetaprime array and choose by distro over this array
            # Update number of valid points we've seen.
            n += nprime
            # Decide if it's time to stop.
            s = sprime and stop_criterion(thetaminus, thetaplus, rminus, rplus)
            # Increment depth.
            j += 1

        # Do adaptation of epsilon if we're still doing burn-in.
        eta = 1. / float(m + t0)
        Hbar = (1. - eta) * Hbar + eta * (delta - alpha / float(nalpha))
        if (m <= Madapt):
            epsilon = exp(mu - sqrt(m) / gamma * Hbar)
            eta = m ** -kappa
            epsilonbar = exp((1. - eta) * log(epsilonbar) + eta * log(epsilon))
        else:
            epsilon = epsilonbar

        # select samples[m,:] from array using discrete probability measure weighted by minimum squared distance to previous samples
        cdf = np.array(cdf)
        pt = np.random.uniform()*cdf[-1]
        index = np.searchsorted(cdf, pt)
        sapmles[m, :] = thetap_arr[index][:]
        ln_prob[m] = logp_arr[index]


    samples = sapmles[Madapt:, :]
    lnprob = ln_prob[Madapt:]
    return samples, lnprob, epsilon


def test_nuts6():
    """ Example usage of nuts6: sampling a 2d highly correlated Gaussian distribution """

    class Counter:
        def __init__(self, c=0):
            self.c = c

    c = Counter()
  
    def logsum(logs):
        addend = np.exp(logsum(logs[1:])-logs[0]) if len(logs) > 1 else 0
        return logs[0] + addend - (addend**2)/2

    def logratios(logs):
        #logs.sort(reverse = True)
        lst = []
        sumlog = logsum(logs)
        for log in logs:
            lst.append(np.exp(log-sumlog))
        return lst, sumlog


    def mixture(p, means, covs, D):
        """
        returns function that generates log-likelihood and its gradient for Gaussian mixture defined by p, means, cov
        """
        def mixtureinfo(theta):
            c.c += 1
            invs = []
            logps = []
            grad = np.zeros((D))
            for cov in covs:
                invs.append(np.linalg.inv(cov))
            
            for i in range(len(p)):
                logps.append(np.log(p[i]) + 0.5*(np.exp(np.linalg.det(invs[i])) - np.square(2*np.pi) - np.dot((means[i]-theta).T,np.dot(invs[i],(means[i]-theta)))))

            logps.sort(reverse = True) 
            ratios, logps = logratios(logps)

            for i in range(len(p)):
                grad += np.dot(invs[i],(means[i]-theta))*ratios[i]
            
            grad[grad == -np.inf] = 0

            return logps, grad

        return mixtureinfo

    def emp_pdf(samples, resolution, D): # resolution is the number of divisions of interval from -20 to 20
        rnge = []
        binalloc = []
        for _ in range(D):
            rnge.append([-20,20])
            binalloc.append(resolution)
        pewp = np.histogramdd(samples, bins = binalloc, range = rnge)[0]/len(samples)
        print(pewp.size)
        return pewp

    def mTV(goodpdf, pdfhat):
        return np.sum(goodpdf*np.abs(goodpdf - pdfhat))
    
    def genMixture():
        D = int(np.ceil(np.random.uniform(0,3, 1))) # dimension
        k = int(np.ceil(np.random.uniform(0,5, 1))) # number of gaussians in mixture
        print("d,k: ", D, k)

        theta0 = np.random.normal(0,1,D)
        
        means, covs = [], []
        for _ in range(k):
            mean = np.random.uniform(-20,20, (D))
            mat = np.random.rand(D, D)
            mat += D*np.identity(D)
            mat = mat@mat.T
            means.append(mean)
            covs.append(mat)

        p = np.random.rand(k)
        p /= np.sum(p)
        
        return (D, theta0, means, covs, p, mixture(p, means, covs, D))

    def sampleMixture(means, covs, p, D, M):
        samples = np.random.multivariate_normal(means[0], covs[0], size=int(np.ceil(M*p[0])))
        for i in range(1, len(means)):
            samples = np.concatenate((samples, np.random.multivariate_normal(means[i], covs[i], size=int(np.ceil(M*p[i])))), axis = 0)
        
        return samples

    def evalmod(M = 10000, Madapt = 500, delta = 0.2, resolution = 400):
        D, theta0, means, covs, p, func = genMixture()

        samples6, lnprob6, epsilon6 = nuts6(func, M, Madapt, theta0, delta)
        samples7, lnprob7, epsilon7 = nuts7(func, M, Madapt, theta0, delta)

        goodpdf = emp_pdf(sampleMixture(means, covs, p, D, M), resolution, D)
        goodpdf2 = emp_pdf(sampleMixture(means, covs, p, D, M), resolution, D)
        pdfhat6 = emp_pdf(samples6, resolution, D)
        pdfhat7 = emp_pdf(samples7, resolution, D)

        return mTV(goodpdf, pdfhat6), mTV(goodpdf, pdfhat7), mTV(goodpdf, goodpdf2), D, len(means)

    
    pdfhat6, pdfhat7, goodpdf2, d, samples, ks = [], [], [], [], 100, []
    for _ in range(samples):
        print("it" + str(_))
        p6, p7, gp2, dd, k = evalmod()
        print(p6, p7, gp2, dd)
        pdfhat6.append(p6)
        pdfhat7.append(p7)
        goodpdf2.append(gp2)
        d.append(dd)
        ks.append(k)
    
    pewp = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]

    for i in range(samples):
        pewp[d[i]-1] = [pewp[d[i]-1][0] + pdfhat6[i-1], pewp[d[i]-1][1] + pdfhat7[i], pewp[d[i]-1][2] + goodpdf2[i], pewp[d[i]-1][3]+1]

    for i in range(len(pewp)):
        if pewp[i][-1] != 0:
            print(str(i+1) + ": ", pewp[i][0]/pewp[i][-1], pewp[i][1]/pewp[i][-1], pewp[i][2]/pewp[i][-1])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import pylab as plt
    plt.subplot(1,3,1)
    plt.plot([i+1 for i in range(4)], [np.log2(pewp[i][0]/pewp[i][-1]) if pewp[i][-1] != 0 else 0 for i in range(4)], 'r-')
    plt.plot([i+1 for i in range(4)], [np.log2(pewp[i][1]/pewp[i][-1]) if pewp[i][-1] != 0 else 0 for i in range(4)], 'g-')
    plt.plot([i+1 for i in range(4)], [np.log2(pewp[i][2]/pewp[i][-1]) if pewp[i][-1] != 0 else 0 for i in range(4)], 'b-')
    plt.ylabel("discretized log total variation metric (red is old NUTS, green is new, blue is Gaussian sampling)")
    plt.xlabel("dimension")

    plt.subplot(1,3,2)
    plt.plot([i+1 for i in range(4)], [np.log2(pewp[i][0]/pewp[i][1]) if pewp[i][-1] != 0 else 1 for i in range(4)], 'r-')
    plt.plot([i+1 for i in range(4)], [np.log2(pewp[i][0]/pewp[i][2]) if pewp[i][-1] != 0 else 1 for i in range(4)], 'g-')
    plt.plot([i+1 for i in range(4)], [np.log2(pewp[i][1]/pewp[i][2]) if pewp[i][-1] != 0 else 1 for i in range(4)], 'b-')
    plt.ylabel("discretized TV log ratio (red is old NUTS over new, green is old over Gaussian sampling, blue is new over Gaussian sampling)")
    plt.xlabel("dimension")

    plt.show()

    pdb.set_trace()

    f = open("data.txt", "a")
    f.write(str(pdfhat6))
    f.write("\n")
    f.write(str(pdfhat7))
    f.write("\n")
    f.write(str(goodpdf2))
    f.write("\n")
    f.write(str(d))
    f.write("\n")
    f.write(str(ks))
    f.write("\n")
    f.close()

    pdb.set_trace()

    """
    M = 10000
    Madapt = 500
    delta = 0.2
    D = 2
    theta0 = np.asarray([0,0])
    

    mean1 = np.asarray([5,5])
    mean2 = -mean1[:]

    cov = np.asarray([[1, 0],
                      [0, 1]])

    func = mixture([0.5,0.5], [mean1, mean2], [cov, cov], D)
    np.seterr(divide = 'ignore') 
    print('Running HMC with dual averaging and trajectory length %0.2f...' % delta)
    samples, lnprob, epsilon = nuts7(func, M, Madapt, theta0, delta)
    #print("hist: ", emp_pdf(samples, 20, D))
    print('Done. Final epsilon = %f.' % epsilon)
    print('(M+Madapt) / Functions called: %f' % ((M+Madapt)/float(c.c)))

    print("old samples: ", len(samples))
    samples = samples[1::10, :]
    print("new samples: ", len(samples))
    print('Percentiles')
    print (np.percentile(samples, [16, 50, 84], axis=0))
    print('Mean')
    print (np.mean(samples, axis=0))
    print('Stddev')
    print (np.std(samples, axis=0))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import pylab as plt
    temp = np.random.multivariate_normal(mean1, cov, size=500)
    plt.plot(temp[:, 0], temp[:, 1], '.')

    temp = np.random.multivariate_normal(mean2, cov, size=500)
    plt.plot(temp[:, 0], temp[:, 1], '.')

    plt.plot(samples[:, 0], samples[:, 1], 'r+')

    #plt.subplot(1,3,2)
    #plt.hist(samples[:,0], bins=50)
    #plt.xlabel("x-samples")

    #plt.subplot(1,3,3)
    #plt.hist(samples[:,1], bins=50)
    #plt.xlabel("y-samples")
    #[-0.88384795 -0.84102893], [2.56551073 2.56593785]
    plt.show()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        import pylab as plt

    pewp = [[0.0025412157242617013, 0.002891545708021004, 0.0016351386758860834], [0.00752409570758503, 0.00018464036105510757, 8.27768411590583e-05], [0.0010841441146584936, 0.00011283743864821408, 8.25246078070308e-05]]
    # 1:     --- pdfhat6, pdfhat7, empdf1
    # 2:    
    # 3:    
    plt.plot([i+1 for i in range(len(pewp[0]))], [np.log2(pewp[i][0]/pewp[i][1]) for i in range(len(pewp))], 'r-')
    plt.plot([i+1 for i in range(len(pewp[0]))], [np.log2(pewp[i][0]/pewp[i][2]) for i in range(len(pewp))], 'g-')
    plt.plot([i+1 for i in range(len(pewp[0]))], [np.log2(pewp[i][1]/pewp[i][2]) for i in range(len(pewp))], 'b-')
    plt.ylabel("log discretized TV ratio")
    plt.xlabel("dimension")

    plt.show()
    """

if __name__ == "__main__":
    test_nuts6()
