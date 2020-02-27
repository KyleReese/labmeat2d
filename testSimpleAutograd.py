import os, imageio
import autograd.numpy as np
import autograd.scipy.signal as sig
from autograd import grad
from autograd.builtins import tuple, list, dict
import matplotlib.pyplot as plt
from natsort import natsorted
from autograd.misc.flatten import flatten

HowManyCells = 20

def hillPlus(x, k):
    return x / (x + k)

def hillMinus(x, k):
    return np.power(k+x, -1)

def deltaProduct(x, n, k, isVessel):
    vessalAdjustment = 0
    if isVessel:
        vessalAdjustment = hillPlus(n,k) * hillMinus(x, k)
    return -1.0 * hillPlus(x, k) + vessalAdjustment

def deltaNutrient(x, n, k, isVessel):
    vessalAdjustment = 0
    if isVessel:
        vessalAdjustment = hillPlus(n,k) * hillMinus(x, k)
    return -1.0 * hillMinus(n, k) + vessalAdjustment

def doODE(nutrient_values, product_values, movablePts, xPoints, yPoints, xIntPoints, yIntPoints):
    k = 0.01
    product_result = list([])
    nutrient_result = list([])
    for iy in range(HowManyCells):
        product_result_row = list([])
        nutrient_result_row = list([])
        for ix in range(HowManyCells):
            isVessel = False
            if ix == xIntPoints[0] and iy == yIntPoints[0]:
                isVessel = True
            x = product_values[ix][iy]
            n = nutrient_values[ix][iy]
            dx = deltaProduct(x,n,k, isVessel)
            dn = deltaNutrient(x,n,k, isVessel)
            product_result_row.append(dx)
            nutrient_result_row.append(dn)
        product_result.append(product_result_row)
        nutrient_result.append(nutrient_result_row)

    return (np.array(product_result), np.array(nutrient_result))
        


def doPDE(values, movablePts, xPoints, yPoints, xIntPoints, yIntPoints):
    # Update the values based on diffusion of the proteins to nearby cells
    D = 0.1 # diffusion parameter
    valuesT = np.transpose(values) 
    adjustmentPDEX = D * nonLinearAdjustment(xPoints)
    adjustmentPDEY = D * nonLinearAdjustment(yPoints)

    #simple diffusion is just a convolution
    convolveLinear = np.array([1*D,-2*D,1*D]) 
    # accumulate the changes due to diffusion 
    for rep in range(50):
        # print(rep)
        newValuesX = list([])
        newValuesY = list([])
        for i in range(HowManyCells):
            row =  values[i] + sig.convolve(values[i], convolveLinear)[1:-1] #take off first and last
            rowY =  valuesT[i] + sig.convolve(valuesT[i], convolveLinear)[1:-1] #take off first and last
            # non-linear diffusion, add the adjustment
            if i in xIntPoints:
                row = row + np.multiply(row, adjustmentPDEX)
            if i in yIntPoints:
                rowY = rowY + np.multiply(rowY, adjustmentPDEY)
            newValuesX.append(row)
            newValuesY.append(rowY)
        
        #Merge rows and transposed columns
        values = np.array(newValuesX) + np.array(newValuesY).T
        # add source at each iteration
        values = values + addSources3(xPoints, yPoints)
        #Update transposed values
        valuesT = values.T
    # the total update returned is the difference between the original values and the values after diffusion
    return values

def addSources3(xPoints, yPoints):
    sources = np.zeros((HowManyCells, HowManyCells))
    for i in range(len(xPoints)):
        try:
            xIndex = int(xPoints[i])
            yIndex = int(yPoints[i])
        except:
            xIndex = int(xPoints._value[i])
            yIndex = int(yPoints._value[i])
        one = [x[:] for x in [[0] * HowManyCells] * HowManyCells]
        one[xIndex][yIndex] = 1
        sources = np.array(one) + sources
    return sources 

def addSources2D(moveablePts):
    sources = np.zeros((HowManyCells, HowManyCells))
    for point in moveablePts:
        try:
            xIndex = int(point[0])
        except:
            xIndex = int(point._value[0])
        try:
            yIndex = int(point[1])
        except:
            yIndex = int(point._value[1])
        one = [x[:] for x in [[0] * HowManyCells] * HowManyCells]
        one[xIndex][yIndex] = 1
        sources = np.array(one) + sources
    return sources

def addSources(moveablePts):
    sources = np.zeros((HowManyCells))
    for x in moveablePts:
        try:
            xIndex = int(x)
        except:
            xIndex = int(x._value)
        one = np.array([0]*xIndex + [1] + [0]*(HowManyCells - xIndex-1))
        sources = one + sources
    return sources
    
############################################################################
### Non Linear PDE 
def nonLinearAdjustment(movablePts):
    # adds an adjustment to the material transfer to take into account
    # the actural position of the cell point in space
    # adjustment is constant for each simulation, because the points do
    # not move so compute once
    allAdjustment = np.zeros(HowManyCells)
    for x in list(movablePts): #only single numbers in x one D
        try:
            pointI = int(x)
        except:
            pointI = int(x._value)
        thisAdj= []
        totalAdj =0 # accumulate the changes around the center point
        for xI in range(0, HowManyCells):
            # find the array locations just before or just after the moveable point
            if ((pointI == xI - 1 and pointI > 0) or           
                (pointI == xI + 1 and pointI < HowManyCells)): 
                deltaConc = distToConc(abs(x - (xI+0.5))) #distance off from the center
                thisAdj.append(deltaConc) 
                totalAdj = totalAdj + deltaConc #accun
            # Otherwise no adjustment   
            else:
                thisAdj.append(0) 
        #accumulate this movable point into the total adjustment 
        allAdjustment = allAdjustment + np.array(thisAdj)
    return allAdjustment
        
def distToConc(distance):
    # maps the distance between two points (in thise case one dimention)
    # positive closer, zero if 1, negative if further away
    return 1 - distance
    
def fitness(moveablePts):
    global nutrient_values, product_values

    try:
        xPoints = moveablePts[0::2] #get x points np array view from flat list
        yPoints = moveablePts[1::2] #get y points np array view from flat list
    except:
        xPoints = moveablePts._value[0::2]
        yPoints = moveablePts._value[1::2]
    try:
        xIntPoints = list([int(x) for x in xPoints])
        yIntPoints = list([int(y) for y in yPoints])
    except:
        xIntPoints = list([int(x) for x in xPoints._value])
        yIntPoints = list([int(y) for y in yPoints._value])

    nutrient_values = np.zeros((HowManyCells, HowManyCells))
    product_values = np.zeros((HowManyCells, HowManyCells))

    nutrient_values = doPDE(nutrient_values, moveablePts, xPoints, yPoints, xIntPoints, yIntPoints)
    product_values = doPDE(product_values, moveablePts, xPoints, yPoints, xIntPoints, yIntPoints)

    odeResult = doODE(nutrient_values, product_values, moveablePts, xPoints, yPoints, xIntPoints, yIntPoints)
    nutrient_values += odeResult[0]
    product_values += odeResult[1]
    # nutrient_values.clip(min=0)  #put all negative values to 0
    # product_values.clip(min=0)
    return(nutrient_values[10][10])

def create_remove_imgs():
    fig_folder = 'figs/'

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    if os.path.exists(fig_folder):
        for img_file in os.listdir(fig_folder):
            os.remove(fig_folder + img_file)

def saveFigureImage(iteration):
    fig.savefig('figs/' + str(iteration) + '.png', size=[1600,400])


# print(doPDE(values, (25.99, 3.4)))
# print(fitness((25.99, 3.4)))
if __name__ == "__main__":
    # result = doPDE(values, np.array([[5.5,5.5]]))
    # fig = plt.figure(figsize=(16, 4), facecolor='white')
    # ax_values       = fig.add_subplot(152, frameon=True)
    # ax_values.imshow(result)
    # plt.draw()
    # plt.pause(10)
    # print(result)
    # quit()
    create_remove_imgs()
    allLoss = []
    stepSize = 0.01
    useAdam = True
    saveGif = False

    fig = plt.figure(figsize=(16, 4), facecolor='white')
    ax_loss         = fig.add_subplot(151, frameon=True)
    ax_values       = fig.add_subplot(152, frameon=True)
    ax_product      = fig.add_subplot(153, frameon=True)
    # ax_diffused_img = fig.add_subplot(154, frameon=True)
    # ax_loss_map     = fig.add_subplot(155, frameon=True)

    def callback(mvable_pts, iteration, nowLoss):
        global nutrient_values, product_values
        # ==================================== #
        # ==== LOSS as a function of TIME ==== #
        # ==================================== #
        ax_loss.cla()
        ax_loss.set_title('Train Fitness')
        ax_loss.set_xlabel('t')
        ax_loss.set_ylabel('fitness')
        allLoss.append(nowLoss)
        time = np.arange(0, len(allLoss), 1)
        ax_loss.plot(time, allLoss, '-', linestyle = 'solid', label='fitness') #, color = colors[i]
        ax_loss.set_xlim(time.min(), time.max())
        ax_loss.legend(loc = 'upper left')
        print('moveable points:', mvable_pts)
        
        ax_values.cla()
        ax_values.set_title('Nutrient')
        ax_values.set_xlabel('position')
        ax_values.set_ylabel('value')
        ax_values.imshow(nutrient_values)
        
        ax_product.cla()
        ax_product.set_title('Product')
        ax_product.set_xlabel('position')
        ax_product.set_ylabel('value')
        ax_product.imshow(product_values)

        plt.draw()
        saveFigureImage(iteration)
        plt.pause(0.001)
        return 3

    gradPDE = grad(fitness)
    mvable_pts = list([12.4, 14.99,5.3,6.8]) #flat list for autograd but points are (x,y) (x,y)
    if useAdam:
        m = np.zeros(np.array(mvable_pts).shape, dtype=np.float64)
        v = np.zeros(np.array(mvable_pts).shape, dtype=np.float64)
        b1=0.9
        b2=0.999
        eps=10**-8
    # print(gradPDE((25.99, 3.4)))
    

    for i in range(500):
        grad_pts = gradPDE(mvable_pts)
        print(grad_pts)
        if useAdam:
            m = (1 - b1) * np.array(grad_pts, dtype=np.float64)      + b1 * m  # First  moment estimate.
            v = (1 - b2) * (np.array(grad_pts, dtype=np.float64)**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1**(i + 1))    # Bias correction.
            vhat = v / (1 - b2**(i + 1))

            # mvable_pts = tuple(np.array(mvable_pts, dtype=np.float64) + np.array(grad_pts, dtype=np.float64))
            mvable_pts = mvable_pts + stepSize * mhat / (np.sqrt(vhat) + eps)
        else:
            mvable_pts = list(np.array(mvable_pts) + np.array(grad_pts)* stepSize)

        newfitness = fitness(mvable_pts)
        print('fitness', newfitness)
        callback(mvable_pts, i, newfitness)
    #print(values)
    if saveGif:
        def img_path_generator(path_to_img_dir):
            for file_name in natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower()):
                if file_name.endswith('.png'):
                    file_path = os.path.join(path_to_img_dir, file_name)
                    yield imageio.imread(file_path)

        fig_folder = 'figs/'
        imageio.mimsave('AutoDiff_Figs.gif', img_path_generator(fig_folder), fps=50)