def pnorm(x,y,p):
    temp = 0
    for i in range(len(x)):
        temp+=(x[i]-y[i])**p
    return temp**(1/p)

x_14=[26,0,2,0,0,0,0]
x_18=[19,0,0,0,0,0,0]


print(pnorm(x_14,x_18,3),"B")
print(pnorm(x_14,x_18,1),"C")
print(pnorm(x_14,x_18,4),"D")
        