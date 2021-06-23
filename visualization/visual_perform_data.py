import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Iterable
import math

# function to let multi dementional list to 1
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def visual_perform(data_path, type):
    list1_num = 0
    list2_num = 0
    list3_num = 0
    listb_num = 0

    MSA_base_all = [[] for x in range(int(5))]
    MSA1_all = [[] for x in range(int(5))]
    MSA2_all = [[] for x in range(int(5))]
    MSA3_all = [[] for x in range(int(5))]

    step1 = []
    step2 = []
    step3 = []
    stepb = []
    for d in data_path:
        if type == 'val':
            df = pd.read_csv(f'D:\\Study thingie\\FACH\\S4 WS2021\\PP\\uebung\\model_data\\vallos\\{d}')
        elif type == 'train':
            df = pd.read_csv(f'D:\\Study thingie\\FACH\\S4 WS2021\\PP\\uebung\\model_data\\trainlos\\{d}')
        if 'MSAb' in d:
            stepb = df['Step']
            for loss in df['Value']:
                MSA_base_all[listb_num].append(loss)
            listb_num+=1
        elif 'MSA1' in d:
            #print(d)
            step1 = df['Step']
            for loss in df['Value']:
                #print(loss, 'add to number', list1_num)
                MSA1_all[list1_num].append(loss)
            list1_num+=1
        elif 'MSA2' in d:
            step2 = df['Step']
            for loss in df['Value']:
                MSA2_all[list2_num].append(loss)
            list2_num+=1
        elif 'MSA3' in d:
            step3 = df['Step']
            for loss in df['Value']:
                MSA3_all[list3_num].append(loss)
            list3_num+=1
        else:
            print(d, 'Not in loop')

    MSA1_sum = list(flatten(MSA1_all))
    MSA2_sum = list(flatten(MSA2_all))
    MSA3_sum = list(flatten(MSA3_all))
    MSAb_sum = list(flatten(MSA_base_all))

    MSA1 = ['MSA1']*len(step1)*5
    MSA2 = ['MSA2']*len(step2)*5
    MSA3 = ['MSA3']*len(step3)*5
    MSA_base = ['Baseline']*len(stepb)*5
    All_step = []
    for i in range(5):
        for j in range(len(step1)):
            All_step.append(step1[j])

    #create dataframe
    data1 = {'Type': MSA1,'Step': All_step, 'Mean': MSA1_sum}
    df1 = pd.DataFrame(data1)
    data2 = {'Type': MSA2,'Step': All_step, 'Mean': MSA2_sum}
    df2 = pd.DataFrame(data2)
    data3 = {'Type': MSA3,'Step': All_step, 'Mean': MSA3_sum}
    df3 = pd.DataFrame(data3)
    datab = {'Type': MSA_base,'Step': All_step, 'Mean': MSAb_sum}
    dfb = pd.DataFrame(datab)
    #merge df
    frames = [dfb,df1, df2, df3] 
    merged_vallos = pd.concat(frames)

    sns.set_theme(style="whitegrid")

    result = sns.lineplot(data = merged_vallos,x = 'Step', y = 'Mean',hue="Type",ci = 'sd')
    sns.despine()
    if type=='val':
        #result.set(yscale='log')
        plt.title('Validation Loss during Training')
        plt.savefig("plots/valloss.png", dpi=300)
    if type=='train':
        plt.title('Train Loss Performance')
        plt.savefig("plots/trainloss.png", dpi=300)

    plt.show()


train_data = os.listdir('D:\\Study thingie\\FACH\\S4 WS2021\\PP\\uebung\\model_data\\trainlos')
val_data = os.listdir('D:\\Study thingie\\FACH\\S4 WS2021\\PP\\uebung\\model_data\\vallos')

visual_perform(val_data,'val')
visual_perform(train_data,'train')
