from random import random
def ran_pi():
    dim=[2,3]
    out=[]
    for i in range(dim[0]):
        temp=[]
        for j in range(dim[1]):
            temp.append(random())
        sm=sum(temp)
        out.append([temp[i]/sm for i in range(dim[1])])
    return out


pi={i:[[1.0,0.0],[0.0,1.0]] for i in range(5)}
temp={i:[[0.0,1.0],[1.0,0.0]] for i in range(5,10)}
pi.update(temp)
       
