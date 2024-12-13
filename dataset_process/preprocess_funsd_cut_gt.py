import os
from XYcut import recursive_xy_cut


ori_path='../../FUNSD/testing_data/gpt_test_xycut'
files=os.listdir(ori_path)
res_path='../../FUNSD/testing_data/gpt_test_xycut_cut'

record=''
for file in files:
    fpath=os.path.join(ori_path,file)
    with open(fpath,'r',encoding='utf-8') as f:
        data=f.read()
        data=data.split('}')[:-1]
    res=''
    x_min = 1000
    x_max = 0
    y_min = 1000
    y_max = 0
    for j in range(len(data)):
        b1=data[j].split('Box:')[-1].split(',')[0]
        b1=b1.replace('[','').replace(']','').split(' ')
        b1=[int(b) for b in b1]
        if b1[0]<x_min:
            x_min=b1[0]               
        if b1[2]>x_max:
            x_max=b1[2]
        if b1[1]<y_min:
            y_min=b1[1]
        if b1[3]>y_max:
            y_max=b1[3]    
    for j in range(len(data)):
        b1=data[j].split('Box:')[-1].split(',')[0]
        b1=b1.replace('[','').replace(']','').split(' ')
        b1=[int(b) for b in b1]
        b1[0]=b1[0]-x_min+10
        b1[1]=b1[1]-y_min+10
        b1[2]=b1[2]-x_min+10
        b1[3]=b1[3]-y_min+10
        # text=data[j].split('Box:')[0]+'Box:['+str(b1[0])+' '+str(b1[1])+' '+str(b1[2])+' '+str(b1[3])+'],'+data[j].split('],')[-1] +'}'
        text=data[j].split('Box:')[0]+'Box:['+str(b1[0])+' '+str(b1[1])+' '+str(b1[2])+' '+str(b1[3])+']}'
        res+=text
    record=record+file+':['+str(x_min-10)+','+str(y_min-10)+']\n'
    with open(os.path.join(res_path,file),'w',encoding='utf-8') as f:
        f.write(res) 

with open('./funsdtest.txt','w',encoding='utf-8') as f:
    f.write(record)