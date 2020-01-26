import autograd.numpy as np
import autograd.scipy.signal as sig
import autograd.grad as grad


HowManyCells = 50
values = np.zeros((HowManyCells))
values[HowManyCells//2] = 10
def doPDE(values, movablePts = [HowManyCells/2+0.1]):
    # Update the values based on diffusion of the proteins to nearby cells
    values = values.T # by protein rather than cell
    D = 0.01#get the diffusion parameter
    adjustmentPDE = D * nonLinearAdjustment(movablePts)
    print(adjustmentPDE)
    #simple diffusion is just a convolution
    convolveLinear = np.array([1*D,-2*D,1*D]) 
    oldValues =  values
    # accumulate the changes due to diffusion 
    for rep in range(0, 1):
        #linear diffusion
        oldValues =  oldValues + sig.convolve(oldValues, convolveLinear)[1:-1] #take off first and last
        # non-linear diffusion, add the adjustment
        oldValues = oldValues + np.multiply(oldValues, adjustmentPDE)
    # the total update returned is the difference between the original values and the values after diffusion
    return oldValues
    
    
############################################################################
### Non Linear PDE 
def nonLinearAdjustment(movablePts):
    # adds an adjustment to the material transfer to take into account
    # the actural position of the cell point in space
    # adjustment is constant for each simulation, because the points do
    # not move so compute once
    allAdjustment = np.zeros(HowManyCells)
    for x in movablePts: #only single numbers in x one D
        pointI = int(x)
        thisAdj= []
        totalAdj =0 # accumulate the changes around the center point
        for xI in range(0, HowManyCells):
            if ((pointI == xI - 1 and pointI > 0) or           #just before the movable point
                (pointI == xI + 1 and pointI < HowManyCells)): # center of right to x
                deltaConc = distToConc(abs(x - (xI+0.5)))
                thisAdj.append(deltaConc) #center of left to x)
                totalAdj = totalAdj + deltaConc #accun
            # Otherwise no adjustment   
            else:
                thisAdj.append(0) 
        #print(thisAdj)
         # do a second pass to place the -1* total in the middle 
        newAdj = []
        for xI in range(0, HowManyCells):
            if pointI == xI: #the grid location of this movable point
                newAdj.append(-1*totalAdj) #center of left to x)
            else:
                newAdj.append(thisAdj[xI]) #no change
        #accumulate this movable point into the total adjustment 
        allAdjustment = allAdjustment + np.array(newAdj)
    return allAdjustment
        
def distToConc(distance):
    # maps the distance between two points (in thise case one dimention)
    # positive closer, zero if 1, negative if further away
    return 1 - distance
    

if __name__ == '__main__':
    print(doPDE(values))
    print(values)
    grad_pde = grad(doPDE)
