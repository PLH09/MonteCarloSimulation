%reset -f
import pandas as pd
import numpy as np
import time
import pandas_datareader.data as pdr
import datetime as dt
import os
from scipy.stats import multivariate_normal
from scipy.stats import norm

start_ = time.time()
url = 'https://www.taifex.com.tw/cht/9/futuresQADetail'
ticker = (pd.read_html(url)[0]["證券名稱"][0:50].astype(str)+'.TW').tolist()
#120個交易日
start = dt.datetime(2018,11,25)
end = dt.datetime(2019,5,30)
df = pdr.DataReader(ticker,"yahoo",start,end)
#開始模擬日
s0 = df.loc[end,'Close']
target = 'Adj Close'
df = df[target]
#log return 
df_pct = np.log(df).diff().dropna(axis = 0).astype(float)
#mean & std
mu = []
std = []
[mu.append(np.mean(df_pct[i]) * 252) for i in df.columns]
[std.append(np.std(df_pct[i]) * np.sqrt(252)) for i in df.columns]
mu = np.array(mu)
std = np.array(std)
#covariance & coef
cov = df_pct.cov()
corr = np.corrcoef(df_pct.T)
#dimension of assets
d = len(s0)
#模擬路徑
path = 1000
#模擬2019/5/31-2021/5/31股價資料(總共484天)
yearday = 252 #一年天數
#總天數
n = 484
#increment
T = n/yearday
#無風險利率
r = 0.01
#風險中立下
m = r - std**2/2

#亂數產生
def mv_normal(mu, cov, path, n, seed):
    np.random.seed(seed)
    d = len(mu)
    z = np.zeros([path, n, d])
    for i in range(0,path):
        for j in range(0,n):
            z[i,j,:] = np.random.multivariate_normal(mu,cov)
    return z
z = mv_normal(np.zeros(len(mu)), corr, path, n, 123457)
#幾何布朗運動模擬股價
def GBM(s0, mu, std, T, r, z, path, n, d):
    dt = T/n
    R = np.zeros((path, n+1, d))
    S = np.zeros((path, n+1, d))
    for i in range(0,path):
        S[i,0,:] = s0
        for j in range(0,n):
            #for k in range(d):
                R[i,j+1,:] = m*dt+std*np.sqrt(dt)*z[i,j,:]
                S[i,j+1,:] = S[i,j,:] * np.exp(R[i,j+1,:])
    
    return S

s = GBM(s0, mu, std, T, r, z, path, n, d)
#%%
#模擬股價畫圖
import matplotlib.pyplot as plt
dt = T/n
t = [j*dt for j in range(0,n+1)]
fontsize = 12

def plot_simulation(t,s,title):
    [plt.plot(t, s[i,:], linewidth=0.5) for i in range(path)]
    plt.xlabel("t", fontsize=fontsize)
    plt.ylabel("Sₜ", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

def plot(i):
    plt.figure()
    plt.subplot(111)
    title = "Simulation Paths of "+ticker[i]+" with Path= "+str(path)
    plot_simulation(t, s[:,:,i], title)
    plt.savefig(ticker[i]+'.png')
    plt.show()

[plot(i) for i in range(d)]
