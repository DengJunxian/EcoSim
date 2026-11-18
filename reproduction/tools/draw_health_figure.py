"""Render the frozen four-panel design from a sealed nine-row outcome census.

This script has no synthetic-data path. It reads versioned metrics and applies
only display-unit conversion and frozen within-seed subtraction. No world,
model, resampling or hypothesis test is executed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

O=Path(__file__).resolve().parent
D=O/'overleaf_draft'
parser=argparse.ArgumentParser()
parser.add_argument('input',type=Path)
args=parser.parse_args()
input_path=args.input.resolve()
assert input_path.is_file(),'Sealed actual input is required; no placeholder figure.'
contract=json.loads((O/'health_figure_contract.json').read_text())
payload=json.loads(input_path.read_text())
assert payload['status']=='SEALED_COMPLETE_CENSUS'
assert payload['metric_definition']==contract['metric_definition']
assert payload['independent_seed_units']==3 and payload['planned_trajectories']==9
assert payload['source_receipts'],'Actual source receipts must accompany plotted values.'
seeds=contract['seeds'];baselines=contract['baselines']
rows=payload['rows'];assert len(rows)==9
index={(row['seed'],row['baseline']):row for row in rows}
assert len(index)==9 and set(index)=={(seed,baseline) for seed in seeds for baseline in baselines}
metrics=['no_trade_ratio','two_sided_coverage']
for (seed,baseline),row in index.items():
    label=f'seed-{seed}-{baseline}'
    assert row['n_agents']==100 and row['requested_ticks']==100
    assert row['config_file_sha256']==contract['trajectory_config_sha256'][label]
    assert row['resolved_config_hash']==contract['trajectory_resolved_config_hash'][label]
    assert isinstance(row['eligible_for_outcomes'],bool)
    if row['eligible_for_outcomes']:
        assert row['committed_ticks']==100 and row['runtime_status']=='completed'
        assert row['financial_audit_status']=='passed' and row['analysis_status']=='completed'
        for metric in metrics:
            value=row[metric]
            assert isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1
    else:
        assert row.get('failure_reason'),'Every ineligible row needs a reason.'
        assert all(row.get(metric) is None for metric in metrics),'Incomplete outcomes remain null.'

paired={metric:{baseline:[] for baseline in baselines[1:]} for metric in metrics}
for metric in metrics:
    for baseline in baselines[1:]:
        for seed in seeds:
            full=index[(seed,'full')];control=index[(seed,baseline)]
            if full['eligible_for_outcomes'] and control['eligible_for_outcomes']:
                paired[metric][baseline].append((seed,100*(full[metric]-control[metric])))
all_differences=[v for by_baseline in paired.values() for pairs in by_baseline.values() for _,v in pairs]
if not all_differences:
    raise SystemExit('No complete paired metrics: retain the nine-row table; no empty result figure generated.')
half_range=min(100,max(10,10*math.ceil(max(abs(v) for v in all_differences)/10)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.linewidth':0.6,'grid.linewidth':0.45,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','svg.hashsalt':input_sha,'savefig.facecolor':'white'})
colors={'full':'#0072B2','zero_intelligence':'#D55E00','homogeneous_linear_value':'#009E73'}
labels={'full':'Full rule','zero_intelligence':'Zero intelligence','homogeneous_linear_value':'Linear value'}
markers=dict(zip(seeds,['o','^','s']))
offsets=dict(zip(seeds,contract['fixed_horizontal_offsets']))
fig=plt.figure(figsize=(7.05,4.35),facecolor='white')
gs=fig.add_gridspec(3,2,height_ratios=[1,1,0.62],left=0.09,right=0.985,bottom=0.045,top=0.88,hspace=0.85,wspace=0.32)
axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#8b9299')
    ax.tick_params(axis='both',width=0.55,length=3,color='#8b9299')
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e5e8eb')

def valid_count(baseline):
    return sum(index[(seed,baseline)]['eligible_for_outcomes'] for seed in seeds)

for ax,metric,title in [(axes[0],metrics[0],'(a) No-trade fraction'),(axes[1],metrics[1],'(b) Two-sided quotes')]:
    for x,baseline in enumerate(baselines):
        for seed in seeds:
            row=index[(seed,baseline)]
            if row['eligible_for_outcomes']:
                ax.scatter(x+offsets[seed],100*row[metric],marker=markers[seed],s=34,
                    facecolor=colors[baseline],edgecolor='white',linewidth=0.55,zorder=3,clip_on=False)
    ax.set(xlim=(-0.4,2.4),ylim=(0,100),ylabel='Committed steps (%)')
    ax.set_yticks([0,25,50,75,100])
    ax.set_xticks(range(3),[labels[b]+'\n'+str(valid_count(b))+'/3 seeds' for b in baselines])
    ax.set_title(title,loc='left',pad=7,fontweight='bold')

pair_counts={}
for ax,metric,title in [(axes[2],metrics[0],'(c) No-trade contrast'),(axes[3],metrics[1],'(d) Quote-coverage contrast')]:
    ax.axhline(0,color='#505860',linewidth=0.8,zorder=2)
    for x,baseline in enumerate(baselines[1:]):
        points=paired[metric][baseline]
        pair_counts[metric+'__'+baseline]=len(points)
        for seed,value in points:
            ax.scatter(x+offsets[seed],value,marker=markers[seed],s=34,facecolor=colors[baseline],
                edgecolor='white',linewidth=0.55,zorder=3,clip_on=False)
    ax.set(xlim=(-0.4,1.4),ylim=(-half_range,half_range),ylabel='Full rule − comparator (pp)')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True,symmetric=True))
    ax.set_xticks(range(2),['Full − ZI\n'+str(len(paired[metric][baselines[1]]))+'/3 pairs',
        'Full − Linear\n'+str(len(paired[metric][baselines[2]]))+'/3 pairs'])
    ax.set_title(title,loc='left',pad=7,fontweight='bold')

handles=[Line2D([0],[0],linestyle='none',marker=markers[seed],markersize=5,
    markerfacecolor='#47515c',markeredgecolor='white',markeredgewidth=0.5,label='Seed '+str(seed)) for seed in seeds]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(0.54,1),ncol=3,
    frameon=False,handletextpad=0.45,columnspacing=1.5,title_fontsize=8,fontsize=8)

status_ax=fig.add_subplot(gs[2,:]);status_ax.set_axis_off();status_ax.set_xlim(0,4);status_ax.set_ylim(0,4)
status_ax.text(0,3.5,'Committed / planned steps',ha='left',va='center',fontsize=8,color='#4c5660')
for j,seed in enumerate(seeds):
    status_ax.text(1.75+0.9*j,3.5,str(seed),ha='center',va='center',fontsize=8,color='#4c5660')
for group,baseline in enumerate(baselines):
    y=2.5-group
    status_ax.text(0,y,labels[baseline],ha='left',va='center',fontsize=8,color='#3f4852')
    for j,seed in enumerate(seeds):
        row=index[(seed,baseline)];x=1.75+0.9*j;eligible=row['eligible_for_outcomes']
        status_ax.add_patch(Rectangle((x-0.4,y-0.38),0.8,0.76,facecolor='#eef3f6' if eligible else '#f6eeee',
            edgecolor='#d5dce1',linewidth=0.45,hatch=None if eligible else '//'))
        status_ax.text(x,y,f"{row['committed_ticks']}/{row['requested_ticks']}"+('' if eligible else ' NA'),
            ha='center',va='center',fontsize=8,color='#273441')

outputs={}
for suffix in ['pdf','svg','png']:
    p=D/'figures'/('health_diagnostics.'+suffix)
    kwargs={'dpi':300}
    if suffix=='pdf':kwargs['metadata']={'Title':'One-day behavioral-rule health diagnostics','Creator':'Civitas local figure script','CreationDate':None,'ModDate':None}
    if suffix=='svg':kwargs['metadata']={'Title':'One-day behavioral-rule health diagnostics','Date':None}
    fig.savefig(p,**kwargs)
    outputs[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
plt.close(fig)
receipt={'status':'built_from_sealed_actual_metrics','input':str(input_path.relative_to(D)),
    'input_sha256':input_sha,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'contract_sha256':hashlib.sha256((O/'health_figure_contract.json').read_bytes()).hexdigest(),
    'metric_definition':payload['metric_definition'],'independent_seed_units':3,'planned_trajectories':9,
    'valid_by_baseline':{b:valid_count(b) for b in baselines},'valid_pairs':pair_counts,'paired_axis_half_range_pp':half_range,
    'synthetic_data':False,'resampling_or_significance_tests':False,'new_worlds':0,'API_calls':0,'outputs':outputs}
(O/'health_figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
