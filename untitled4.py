from sklearn import datasets

d=datasets.load_wine()

for i in range(0,len(d.data)):
    print(i+1,d.data[i],d.target[i])
    
from sklearn import svm

s = svm.SVC(gamma=0.1,C=10)
s.fit(d.data,d.target)

new_d = [[14.13, 4.1, 2.74, 24.5, 96, 2.05, 0.76, 0.56, 1.35, 9.2,
 0.61, 1.6, 560],[12.77, 2.39, 2.28, 19.5, 86, 1.39, 0.51, 0.48,  0.64, 9.899999, 0.57, 1.63
, 470.]]
                  
res=s.predict(new_d)
print("새로운 2개의 샘플 부류", res)

