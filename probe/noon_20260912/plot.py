from pathlib import Path
import os
import re
import statistics
os.environ.setdefault('MPLCONFIGDIR','/tmp/trsm-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
s=(root/'logs/noon_20260912/390-owner-import47.sh.log').read_text();flag=None;data={0:[],1:[]}
for line in s.splitlines():
    if line.startswith('OWNER_IMPORT_PROFILE'):flag=int(line.split('=')[-1])
    if flag is not None and '每线程平均 ms' in line:
        data[flag].append({k:float(v) for k,v in re.findall(r'(prep|load|kloop|dkloop|ddump|store) ([\d.]+)',line)})
means={flag:{k:statistics.mean(x[k] for x in rows[1:]) for k in rows[0]} for flag,rows in data.items()}
fig,ax=plt.subplots(figsize=(8.3,3.2));left=[0,0]
for key,color,label in [('load','#3b82f6','Input copy'),('kloop','#f97316','Main kernel'),('dkloop','#22c55e','Diagonal compute'),('ddump','#86efac','Diagonal output'),('store','#8b5cf6','Final store'),('prep','#a3a3a3','Preparation')]:
    vals=[means[t][key] for t in (0,1)];ax.barh(['Control','Hosted input'],vals,left=left,color=color,label=label)
    for y,(v,l) in enumerate(zip(vals,left)):
        if v>.1:ax.text(l+v/2,y,f'{v:.3f}',ha='center',va='center',fontsize=9)
    left=[x+y for x,y in zip(left,vals)]
ax.set_xlabel('Mean time per thread (ms), five warm profiled calls')
ax.set_title('TC1: input time saved is outweighed by kernel slowdown')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,frameon=False)
ax.spines[['right','top']].set_visible(False)
fig.tight_layout();fig.savefig(root/'probe/tc1_hosted_input_20260912.svg',bbox_inches='tight')
