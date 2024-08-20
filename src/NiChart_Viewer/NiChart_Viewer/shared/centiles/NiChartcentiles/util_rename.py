import pandas as pd

df1 = pd.read_csv('List_centiles.csv')
df2 = pd.read_csv('List_Map_Tmp.csv')

name_dict = dict(zip(df2['ROI1'], df2['ROI2']))



for fname in df1.fname:
    dftmp = pd.read_csv(fname)
    
    fnamenew = fname[0:-4] + '_OLD.csv'
    print('From : ' + fnamenew)
    
    dftmp.to_csv(fnamenew)    
    dftmp = dftmp.replace({"ROI_Name": name_dict})
    dftmp.to_csv(fname)    
    
    print('To : ' + fname)
