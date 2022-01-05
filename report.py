from os import listdir
import sys
from os.path import isfile, join
import pickle5 as pickle
from statistics import mean, stdev
from math import sqrt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dp", type=lambda v: v=='True', default=False, help="results file only")
opt = parser.parse_args()
print(opt)

paths = ['experiment_results/', 'downloads/experiment_results/']
if opt.dp:
    paths[-1] = 'downloads_dp/experiment_results/'
sources = 'local', 'remote'

try:
    menu = {i:p for i,p in enumerate(sources)}
    choice = int(input(f'Choose file: {menu}'))
except ValueError:
    print("Not a number")

source = sources[choice]
path = paths[choice]
dp = '_dp' if opt.dp else ''
file = path+'results'+dp+'.pkl'
# file, dp = 'downloads/experiment_results/run_oct_12_2021.pkl', ''
print('Loading: ', file)

with open(file, 'rb') as f:
    results = pickle.load(f)

# get unique sets of params, iterations and gans
gans = set([v[0] for v in results.keys()])
params = set([v[1] for v in results.keys()])
iterations = set([v[-1] for v in results.keys()])
datasets = ['mnist', 'fmnist']

iterations
gans, iterations = [sorted(list(l)) for l in [gans, iterations]]
# print(results)
# accumulate data s.t. accum[gan, param] = fids, aucs
accum = dict()
for g in gans:
    for p in params:
        for k, v in results.items():
            if k[0] == g and k[1] == p:
                if len(v) > 0:
                    if isinstance(p, str):
                        val = g,p
                        val2 = [[v[0]], [0]]
                    else:
                        val = g,p[2:4]+p[-3:-2]+p[-2:]
                        val2 = [[v[0]], [v[1][0]]]

                    if val in accum.keys():
                        accum[val][0]+=val2[0]
                        accum[val][1]+=val2[1]
                    else:
                        accum[val]=val2
var = list({v for _,v in accum.keys()})
# print(var)
params_reduced = sorted(var, key = lambda v: (v,0,0,0,0) if isinstance(v, str) else (v[2],v[0],v[3],v[1],v[4]))

import matplotlib.pyplot as plt
import numpy as np
metric_map = {'fid':0, 'auc':1}
for metric, metric_id in metric_map.items():
    fig, axs = plt.subplots(len(gans), 2, figsize=(10,10 if not opt.dp else 5), sharex=False)
    if len(axs.shape) < 2:
        print('Reshaping axis...')
        axs.shape = 1,2
    fig.autofmt_xdate(rotation=45)
    idx_d = lambda d: 0 if d=='mnist' else 1
    for i, g in enumerate(gans):
        axs[i,0].set_ylabel(g.upper())
        axs[0,0].set_title('mnist'.upper()+'\n n='+str(max(iterations)+1))
        axs[0,1].set_title('fmnist'.upper()+'\n n='+str(max(iterations)+1))
        x, y = ([],[]), ([],[])
        for p in params_reduced:
            # second condition filters out placeholders
            # if not isinstance(p,str) and max(accum[g,p][metric_id]) > 0:
            dp_skip = opt.dp and metric=='fid' and 'nondp' in p
            if max(accum[g,p][metric_id]) > 0 and not dp_skip:
                d = p if isinstance(p,str) else p[2]
                x_label = str(p[:2] + p[-2:]) if not isinstance(p,str) else 'clean'
                x[idx_d(d)].append(x_label)
                y[idx_d(d)].append(accum[g,p][metric_id])
        for idx in [0,1]:
            # print(x,y)
            axs[i, idx].boxplot(y[idx], labels=x[idx], notch=True)
            # [tick.set_rotation(15) for tick in axs[i, idx].get_xticklabels()]
    plt.savefig('reports/'+source+'_'+metric+dp+'_report.png', bbox_inches="tight")
    plt.close()


# # CREATE TABLES
# # skip dp
# if opt.dp: sys.exit()
#
# import pandas as pd
# metric_map = {'fid':0, 'auc':1}
# metric, metric_id ='fid', 0
# fid_dict = {g:[] for g in gans}
# fid_error_dict = {g:[] for g in gans}
# auc_dict = {g:[] for g in gans}
# auc_error_dict = {g:[] for g in gans}
#
# #standard deviation divide by baseline mean
# #other option is to use standard error
# #https://stats.stackexchange.com/questions/7554/how-to-express-error-as-a-percentage
# rel_err = lambda l,baseline: (stdev(l)/sqrt(1))/baseline*100
#
# id1,id2=2,1
# columns = [tuple(p[:id1] + p[-id2:]) for p in params_reduced if not isinstance(p,str)]
# for i, g in enumerate(gans):
#     col_check=[]
#     for p in params_reduced:
#         # cln_fid = accum[g,''][metric_id]
#         # second condition filters out placeholders
#         if not isinstance(p,str) and max(accum[g,p][0]) > 0:
#             baseline = mean(accum[g,p[2]][0])
#             # print(g, p[2], metric, baseline, accum[g,p[2]][metric_id])
#             x_label = tuple(p[:id1] + p[-id2:])
#             fid_dict[g].append(round((mean(accum[g,p][0]) - baseline)/baseline*100))
#             fid_error_dict[g].append(round(rel_err(accum[g,p][0], baseline)))
#             auc_dict[g].append(round((mean(accum[g,p][1])),2))
#             auc_error_dict[g].append(round((rel_err(accum[g,p][1], 1.0))))
#             col_check.append(x_label)
#     assert col_check==columns
# assert list(fid_dict.keys())==gans, [fid_dict.keys(),gans]
#
# def save_table(table, table_name='test'):
#     with open(f'reports/{table_name}.tex', 'wb') as f:
#         f.write(table.to_latex().encode())
#
# df = pd.DataFrame(fid_dict.values(), index=gans, columns=[(c[0],c[1],c[2][0]) for c in columns])
# df_fid_err = pd.DataFrame(fid_error_dict.values(), index=gans, columns=[(c[0],c[1],c[2][0]) for c in columns])
# df_fid_and_err = pd.DataFrame(index=gans, columns=[(c[0],c[1],c[2][0]) for c in columns])
#
# # print(df_fid_err)
# for i, col in enumerate(df.columns):
#     df_fid_and_err.iloc[:,i] = df.iloc[:,i].astype(str)+' ('+df_fid_err.iloc[:,i].astype(str)+')'
# print(df)
# print(df_fid_and_err)
#
# save_table(df_fid_and_err.iloc[:,0:6], 'fid_table_fmnist')
# save_table(df_fid_and_err.iloc[:,6:], 'fid_table_mnist')
#
# # print(df.iloc[:,0:6])
# # print(df.iloc[:,6:])
#
# # df.to_pickle('/tmp/df.pkl')
# # df = pd.read_pickle('/tmp/df.pkl')
# # df
#
# def make_table_bogus(ii):
#     df_input = pd.DataFrame()
#     for i in ii:
#         df_input[df.columns[i+1]] = round((df.iloc[:,i+1]-df.iloc[:,i])/df.iloc[:,i]*100)
#         df_input[df.columns[i+1]] = df_input[df.columns[i+1]].astype(int).astype(str)
#         df_input[df.columns[i+1]] += ' ('+(df.iloc[:,i+1]-df.iloc[:,i]).astype(str)+')'+' ('+df_fid_err.iloc[:,i+1].astype(str)+')'
#         df_input[df.columns[i+2]] = round((df.iloc[:,i+2]-df.iloc[:,i])/df.iloc[:,i]*100).astype(int).astype(str)
#         df_input[df.columns[i+2]] += ' ('+(df.iloc[:,i+2]-df.iloc[:,i]).astype(str)+')'+' ('+df_fid_err.iloc[:,i+2].astype(str)+')'
#     return df_input
#
# df_bogus_fmnist = make_table_bogus([0,3])
# save_table(df_bogus_fmnist, 'fid_table_bogus_fmnist')
#
# df_bogus_mnist = make_table_bogus([6,9])
# save_table(df_bogus_mnist, 'fid_table_bogus_mnist')
#
# df = pd.DataFrame(auc_dict.values(), index=gans, columns=[(c[0],c[1],c[2][0]) for c in columns])
# save_table(df.iloc[:,0:6],'auc_table_fmnist')
# save_table(df.iloc[:,6:],'auc_table_mnist')
