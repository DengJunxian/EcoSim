"""Render verified engineering tables and a method schematic using local matplotlib."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_build(root, expected_hash):
    root=Path(root)
    if sha(root/'BUILD.json')!=expected_hash:raise ValueError('BUILD_HASH_CHANGED')
    build=json.loads((root/'BUILD.json').read_text())
    if build['schema_version']!='kdd27-measured-build-v1' or build['metric_version']!='kdd27-metrics-v1':
        raise ValueError('UNKNOWN_BUILD_OR_METRIC_VERSION')
    for name,ref in build['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=ref['sha256']:raise ValueError('FIGURE_INPUT_HASH_CHANGED')
    for ref in build['analysis_sources']:
        if sha(ref['path'])!=ref['sha256']:raise ValueError('METRIC_CODE_CHANGED_REANALYZE')
    return build


def render(root, expected_hash, out):
    root,out=Path(root).resolve(),Path(out).resolve();build=validate_build(root,expected_hash)
    if out.exists():raise ValueError('OUTPUT_EXISTS')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    out.mkdir(parents=True)
    def rows(name):
        with (root/name).open() as f:return list(csv.DictReader(f))
    manifest=[]
    colors=['#0072B2','#E69F00','#009E73']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                         'svg.fonttype':'none','svg.hashsalt':'kdd27-measured-v1','savefig.facecolor':'white'}):
        def save(fig,name,caption,inputs,kind):
            outputs=[]
            for ext in ('pdf','svg','png'):
                path=out/(name+'.'+ext)
                metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
                fig.savefig(path,dpi=300,metadata=metadata)
                outputs.append({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
            plt.close(fig)
            manifest.append({'id':name,'caption':caption,'kind':kind,'input_tables':{n:build['files'][n] for n in inputs},
                'outputs':outputs,'scientific_hypotheses_supported':[]})
        fig,ax=plt.subplots(figsize=(10,4.8));fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.9)
        ax.set(xlim=(0,10),ylim=(0,4.6));ax.axis('off')
        stages=[('Boundary + events','State / scheduled intervention',0),('Observe','Eligible fields and memory refs',2.5),
                ('Decide','Rule | heuristic | LLM | hybrid',5),('Validate + execute','Shared resources / ledger / LOB',7.5)]
        for title,subtitle,x in stages:
            ax.add_patch(FancyBboxPatch((x+.05,2.6),2.25,1.15,boxstyle='round,pad=.03',facecolor='#E8F1F6',edgecolor='#0072B2'))
            ax.text(x+1.17,3.35,title,ha='center',weight='bold',fontsize=10)
            ax.text(x+1.17,2.92,subtitle,ha='center',fontsize=8.4)
            if x<7.5:ax.annotate('',xy=(x+2.48,3.18),xytext=(x+2.33,3.18),arrowprops={'arrowstyle':'->','color':'#555555'})
        ax.add_patch(FancyBboxPatch((2.7,1.15),4.6,.8,boxstyle='round,pad=.03',facecolor='#E8F3ED',edgecolor='#009E73'))
        ax.text(5,1.68,'Feedback → commit → immutable receipts',ha='center',weight='bold')
        ax.text(5,1.35,'State hashes · decisions · trades · checkpoints',ha='center',fontsize=9)
        ax.annotate('',xy=(7.3,1.55),xytext=(8.7,2.55),arrowprops={'arrowstyle':'->','color':'#555555'})
        ax.annotate('',xy=(1.15,2.55),xytext=(2.7,1.55),arrowprops={'arrowstyle':'->','color':'#555555'})
        ax.text(5,.62,'RQ1: market statistics       RQ2: paired branches       RQ3: cognition and cost',ha='center',fontsize=10)
        ax.text(5,.12,'Implemented interfaces; model evaluation and formal RQ results remain pending.',ha='center',fontsize=9,color='#555555')
        fig.suptitle('Paper profile: implemented data flow',fontsize=14)
        save(fig,'method_overview','Method schematic of implemented interfaces, not a performance or scientific-result figure.',
             ['claim_evidence.csv','parameters.csv'],'method_schematic')
        metrics=rows('run_level.csv');conditions=['full','zero_intelligence','homogeneous_linear_value']
        labels=['Full','Zero intelligence','Linear value']
        data={(r['condition_id'],r['metric_id']):r for r in metrics if r['study']=='RQ1'}
        fig,axs=plt.subplots(1,2,figsize=(8.7,4.2));fig.subplots_adjust(left=.16,right=.97,top=.81,bottom=.22,wspace=.5)
        for i,c in enumerate(conditions):
            for offset,name,color in [(-.18,'no_trade_ratio',colors[1]),(.18,'two_sided_coverage',colors[0])]:
                row=data[c,name];v=float(row['value']);axs[0].barh(i+offset,v,height=.31,color=color,label=name if i==0 else None)
                axs[0].text(v+.025,i+offset,f"{row['n_valid']}/{row['n_total']}",va='center',fontsize=9)
            row=data[c,'two_sided_spread_bps']
            if row['value']:
                axs[1].scatter(float(row['value']),i,color=colors[0],s=45)
                axs[1].annotate(f"{float(row['value']):.2f} ({row['n_valid']}/{row['n_total']})",(float(row['value']),i),xytext=(7,7),textcoords='offset points',fontsize=9)
            else:axs[1].text(.05,i,'Unavailable',va='center')
        for ax in axs:ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.16)
        axs[0].set(xlim=(0,1.18),xlabel='Fraction of committed steps');axs[1].set(xlabel='Conditional mean two-sided spread (bp)')
        axs[1].margins(x=.5)
        axs[0].legend(['No trade','Two-sided quote'],loc='upper left',bbox_to_anchor=(-.15,-.16),ncol=2,fontsize=9,frameon=False)
        fig.suptitle('Engineering liquidity diagnostics — N=128, T=12',fontsize=13)
        fig.text(.5,.035,'One paired world per condition; observed steps, not independent samples. No full 100-tick day; no SD/CI.',ha='center',fontsize=8.8)
        save(fig,'engineering_liquidity','Complete three-baseline engineering census; conditional spread mean with coverage; one world per condition and no SD/CI.',
             ['run_level.csv','summary.csv'],'measured_engineering')
        arms=rows('rq2_diagnostics.csv')
        label_map={'full_control':'Control','no_secondary_contagion':'No contagion','no_herding_feedback':'No herding','both_off':'Both off',
            'full_loss_off':'All loss off','lambda_only_1':'Lambda = 1','reverse_retrospective_mapping':'Reverse mapping',
            'halt25':'Halt 25','tax0015':'Tax .0015','same_day_lock':'Same-day lock','narrow_band005':'Band .005'}
        fig,axs=plt.subplots(1,3,figsize=(10.8,5.7));fig.subplots_adjust(left=.17,right=.98,top=.85,bottom=.2,wspace=.5)
        for i,row in enumerate(arms):
            axs[0].scatter(100*float(row['mdd']),i,color=colors[0],s=36)
            if row['relative_turnover']:axs[1].scatter(float(row['relative_turnover']),i,color=colors[1],s=36)
            else:axs[1].text(.03,i,'NA',va='center')
            censored=row['censored']=='True';axs[2].scatter(float(row['restricted_time']),i,marker='>' if censored else 'o',color=colors[2],s=40)
            axs[2].annotate(f"{row['tradable_steps']}/{row['total_steps']}",(float(row['restricted_time']),i),xytext=(6,0),textcoords='offset points',va='center',fontsize=9)
        for i,ax in enumerate(axs):
            ax.set_yticks(range(len(arms)),[label_map[r['condition']] for r in arms] if i==0 else ['']*len(arms));ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
        axs[0].set(xlabel='MDD (%)',xlim=(-.02,max(100*float(r['mdd']) for r in arms)*1.2+.02))
        axs[1].set(xlabel='Turnover / control');axs[1].axvline(1,color='#888888',linestyle=':',linewidth=1)
        axs[2].set(xlabel='Restricted recovery (steps)',xlim=(0,38))
        fig.suptitle('Engineering policy and mechanism branches — all 11 arms',fontsize=13)
        fig.text(.5,.095,'N=20, T=32; one seed; 27 event steps; no cross-world CI. Recovery: > = right censored.',ha='center',fontsize=9)
        fig.text(.5,.045,'Recovery labels show tradable / total steps. Halt-period unchanged prices remain in the frozen recovery definition.',ha='center',fontsize=9)
        save(fig,'engineering_branches','All eleven branches of one N20 engineering world; MDD, within-world turnover ratio and restricted recovery with censoring/tradable denominator.',
             ['rq2_diagnostics.csv','paired_effects.csv','interaction.json'],'measured_engineering')
        timing=rows('timing_pairs.csv');fig,ax=plt.subplots(figsize=(6.8,4.1));fig.subplots_adjust(left=.14,right=.73,top=.82,bottom=.22)
        for i,row in enumerate(timing):
            vals=[float(row['A_seconds']),float(row['B_seconds'])]
            ax.plot([0,1],vals,marker=['o','s','^'][i],color=colors[i],linewidth=1.6,label=f"Pair {row['pair']}: {float(row['reduction_percent']):.2f}%")
        ax.set(xticks=[0,1],xticklabels=['A: previous path','B: adopted path'],ylabel='Total wall time (seconds)')
        ax.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=9);ax.grid(axis='y',alpha=.16)
        fig.suptitle('Measured timing pairs — N=100, T=12',fontsize=13)
        fig.text(.5,.07,'Three repeats of the same world on one Mac; no scale extrapolation or statistical CI.',ha='center',fontsize=9)
        save(fig,'engineering_timing','Three AB/BA/AB timing repeats of the same rule world; original measured seconds; paired reductions are descriptive.',
             ['timing_pairs.csv','traces.json'],'measured_engineering')
    receipt={'schema_version':'kdd27-render-v1','build_sha256':expected_hash,'render_script_sha256':sha(__file__),
        'matplotlib_version':matplotlib.__version__,'figures':manifest,'formal_result_figures':0}
    (out/'FIGURES.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--build-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=render(a.build,a.build_sha256,a.out)
    print(json.dumps({'figures':len(r['figures']),'formal_result_figures':0,'matplotlib':r['matplotlib_version']}))


if __name__=='__main__':main()
