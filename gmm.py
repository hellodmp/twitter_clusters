#Expectation Maximisation Alrgorithm
import random as rnd
import math
from numpy import *
from matplotlib import pyplot as mp

#below is added for kmeans-------------------- ****************************************************************************
def distance(tupa, tupb):
    ''' Calculate the distance between two points '''
    return math.sqrt(math.pow((tupa[0]-tupb[0]), 2) + math.pow((tupa[1]-tupb[1]), 2)) #euclidian distance

def assign(dp, means):
    ''' Take a data point(tuple), return the closest of the means '''
    dists = [distance(dp, m) for m in means]
    return dists.index(min(dists)) #return index centroid with smallest distance from dp

def updatemeans(dataset, assignments, k): #presumes 2 dimensions
    '''calclate the Barycentre of the clusters in the assignments array'''
    newcentroids = [[0,0]]*k #presumes two dimensions
    for i in range(k):
        nincluster = assignments.count(i)
        if nincluster > 0: #Note: if no observations are assigned to centroid, centroid does not move
            prior = 1 / nincluster ; sigma = [0,0] #presumes 2 dimensions
            for x in range(len(assignments)):
                if assignments[x] == i:
                    point = dataset[x]
                    sigma[0] += point[0] ; sigma[1] += point[1]
            newcentroids[i] = [sigma[0]*prior, sigma[1]*prior] #arithmetic mean for 2 dimensions
    return newcentroids

def squareerrorfunct(dataset, assignments, centroids, k):
    sqe = 0
    for i in range(k):
        for x in range(len(assignments)):
            if assignments[x] == i:
                sqe += math.pow(distance(dataset[x], centroids[i]),2)
    return sqe



def initialisekcentroids(k, dataset): #currently chooses k random centroids from data
    '''Pick k initial centroids with Forgy method'''
    centroids = [(0,0)] * k  # initialise centroids as zero
    for i in range(k): ###initialise means from random datapoints using the FOrgy method
        centroids[i] = dataset[random.randint(0,len(dataset)-1)]
    return centroids





#implement k-means:

def kMeansCluster(k, dataset):
    '''Take a dataset (each entry[:-1] is an observation, entry[-1] is the class) and cluster into k clusters'''
    centroids = initialisekcentroids(k, dataset) #this uses the Forgy method
    oldassignments, newassignments = [], [1]
    i = 0
    while i <= 50 and newassignments != oldassignments: #this repeats until clusters dont change, could repeat until centroids dont move
        oldassignments = newassignments
        newassignments = []
        for obs in dataset: #assign each datapoint to a cluster
            newassignments.append(assign(obs , centroids))
        #print(newassignments)
        centroids = updatemeans(dataset,newassignments,k) #update the new centroid assignments based on the distance metrics over all cluster members
        #print("Square error:", squareerrorfunct(dataset, newassignments, centroids, k))
        i += 1
    return centroids, newassignments, i





#---------------- change this part of GMM to make function GMM2

def GMM2(data):
    '''GMM on dataset from ex2, return info for each class k=n'''
    ''' This functionaility is implemented in GMM itself with 3rd argument'''
    theta = []
    means, assignments, iterations = kMeansCluster(K, data)
    for k in range(K):
        theta.append([array(means[k]).reshape((2,1)) , matrix(cov([data[i] for i, x in enumerate(assignments) if x==k], rowvar=0)), assignments.count(k)/len(assignments)]) #means, covarmatrices, priors
    print(theta)
    #Iteration
    pass # et cetera
#above was added for kmeans ---------- ***************************************************************************************


def plot_data(Gamma,theta):
    #Calculating p(k|x)*p(x) = p(x|k)*p(k) and using that to compare the p(k|x) assuming p(x) is constant
    klass_prob=[]
    for point in range(len(Gamma)):
        maxklassprob=0
        maxklass=0
        colors=[]
        points_per_klass=[]
        for i in range(len(theta)):
            points_per_klass.append(0)
        for klass in  range(len(theta)):
            p_of_k_by_x=Gamma[point][klass]*theta[klass][2]
            #p_of_k_by_x=theta[klass][2]
            if ( p_of_k_by_x >= maxklassprob ):
               maxklass=klass
               maxklassprob=p_of_k_by_x
        points_per_klass[maxklass] += 1
        klass_prob.append([data[point],maxklass])

    #Define k no. of colors
    for color in range(len(theta)):
        colors.append(random.rand(3,1))

    # Plotting 
    for point in klass_prob:
        mp.scatter(point[0][0],point[0][1],c=colors[point[1]])

    mp.show()
     

def GMM(data,K,km_init=False):
    '''GMM on dataset from ex2, return info for each class k=n'''
    theta = []
    # Initialization of theta
    if ( km_init ):
      means, assignments, iterations = kMeansCluster(K, data)
      for k in range(K):
        theta.append([array(means[k]).reshape((2,1)) , matrix(cov([data[i] for i, x in enumerate(assignments) if x==k], rowvar=0)), assignments.count(k)/len(assignments)]) #means, covarmatrices, priors                 
    else:
      for k in range(K): #initialise for each cluster
        prior = 1/K
        theta.append(Initialise(data) + [prior] )
    #Iterations
    precision=0.01
    maxiters=11
    i=0
    converging=False
    old_mu_dists=[ 0 for  every in range(K) ]
    while( i<= maxiters and converging==False ):
        Gamma=Expectation(data, theta)
        newtheta=Maximisation(Gamma,theta)
        # Check the distances between previous mu and new mu and decide on converging if
        mu_distances=[]
        #Calculate distances betweeen mu from old theta and mu from new theta
        for klass in range(len(theta)):
            mu_distances.append(linalg.norm(theta[klass][0] - newtheta[klass][0]))
       # Criterion to stop the iterations - check the maximum movement of mus in consequtive theta's ,if the maximum movement is less
       # than precision value and iterations are atleast half of the max iterations , then confirm it as converging.
        if ( max(subtract(array(mu_distances),array(old_mu_dists))) <= precision  and i >= maxiters//2 ):
           converging=True
        #print ( "maximum mu distance - ", max(subtract(array(mu_distances),array(old_mu_dists))) )
        old_mu_dists=mu_distances
        theta=newtheta

        print ( "K - ", K, ",finished iteration - ",i,", converging -",converging)
        i += 1
    print ( "Mu s for K = ",K,"---",[ newtheta[every][0] for every in range(K) ] )
    plot_data(Gamma,newtheta)

    return


def Initialise(data):
    #for class 
    rans = rnd.sample(range(0,N),5)
    dps = [data[x] for x in rans]
    cov = matrix(identity(2))
    mean = [sum(dps[x][0] for x in range(5))/len(rans), sum(dps[x][1] for x in range(5))/len(rans) ]
    return [array(mean).reshape((2,1)), cov]

def Expectation(data, theta):
    posterior=[]
    for point in data:
        arrpoint=array(point).reshape((2,1))
        den = sum( norm_pdf_multivariate(arrpoint ,theta[i][0], theta[i][1]) * theta[i][2] for i in range(K) )
        xposterior=[]
        for k in range(K):
            num = norm_pdf_multivariate(arrpoint ,theta[k][0], theta[k][1]) * theta[k][2]
            xposterior.append( num/den )
        posterior.append(xposterior)
    return(posterior)


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I        


        result = math.pow(math.e, -0.5 * (x_mu.T * inv * x_mu))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")        
            
    
    return

def Maximisation(Gamma,theta):
    newtheta=[]
    for k in range(K):
        normalizer=1/sum(Gamma,axis=0)[k]
        x_newmu = normalizer * sum ( [ data[i][0]*Gamma[i][k] for i in range(N) ] )
        y_newmu = normalizer * sum( [ data[i][1]*Gamma[i][k] for i in range(N) ] )
        new_mu=matrix([[x_newmu],[y_newmu]])
        sumofmatr=matrix([[0,0],[0,0]])
        
        for i in range(N):
            sumofmatr = sumofmatr + Gamma[i][k]* dot(matrix(array(data[i]).reshape((2,1)))-new_mu,transpose(matrix(array(data[i]).reshape((2,1)))-new_mu))


        new_covar = normalizer * sumofmatr 

        denom_newprior=sum([  theta[k][2]*Gamma[i][k] for k in range(K) ] ) 
        newprior= (1/N)*sum ( [  theta[k][2]*Gamma[i][k] / sum([ theta[l][2]*Gamma[i][l] for l in range(K) ] )  for i in range(N) ] )
        
        newtheta.append([new_mu,new_covar,newprior])

    return newtheta

data = [[float(line.split('\t')[0]),float(line.split('\t')[1])] for line in open("R15.txt", "r").readlines()]

mp.plot( [ data[i][0] for i in range(len(data)) ], [ data[i][1] for i in range(len(data)) ] ,'bo') 
mp.show()

N=len(data)

# For k=4 and k=15 , normal GMM with random theta initialization

K=4
GMM(data,K)

K=15
GMM(data,K)


##   BONUS   ### 
K=4
GMM(data,K,True)  # To call with km-initialisation

K=15
GMM(data,K,True)  # To call with km-initialisation
