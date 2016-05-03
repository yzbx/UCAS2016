# python xls read and write

import xlrd
import numpy as np
import pickle

def hasMissingValue(value):
    assert(value.__len__()>57)

    for i in range(22,25):
        if(value[i]==99):
            return true
    for i in range(46,49):
        if(value[i]==99):
            return True

    if(value[31]==9999 or value[57]==9999):
        return True

    return False

filename='../data/phy_test_with_lable.xls'
data=xlrd.open_workbook(filename)
sheet=data.sheets()[0]
nrows=sheet.nrows
ncols=sheet.ncols

i=0
X_unmissing=[]
Y_unmissing=[]
X_missing=[]
Y_missing=[]
X_unknow=[]
Y_unknow=[]

while i<nrows:
    xi=sheet.row_values(i)
    yi=-1
    if(xi[1]=='1'):
        yi=1
    elif(xi[1]=='0'):
        yi=0

    missingFlag=hasMissingValue(xi)
    xi=xi[2:]
    if(yi!=-1 and (not missingFlag)):
        X_unmissing.append(xi)
        Y_unmissing.append(yi)
    elif(yi!=-1 and missingFlag):
        X_missing.append(xi)
        Y_missing.append(yi)
    else:
        X_unknow.append(xi)
        Y_unknow.append(yi)

unmissing,missing,unknow=range(3)
X=[]
Y=[]

X.append(np.array(X_unmissing))
X.append(np.array(X_missing))
X.append(np.array(X_unknow))

Y.append(np.array(Y_unmissing))
Y.append(np.array(Y_missing))
Y.append(np.array(Y_unknow))

f=file('home/yzbx/build/xls.pkl','wb')
pickle.dump(X,f,True)
pickle.dump(Y,f,True)
f.close()

print "end of programe!"
