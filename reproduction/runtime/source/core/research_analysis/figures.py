"""Data-only publication figures for the complete analytic formal census."""
from pathlib import Path
import csv
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import provider_effects as e


LABELS = {'single':'Single', 'independent':'Independent', 'free_debate':'Free debate', 'abn':'ABN',
    'cdr_s0_v0':'CDR 00', 'cdr_s0_v1':'CDR 01', 'cdr_s1_v0':'CDR 10', 'cdr_s1_v1':'CDR 11',
    'central_rule':'Central rule'}
CONTRASTS = {'full_vs_single':'CDR 11 − Single', 'factor_S':'S main effect',
             'factor_V':'V main effect', 'interaction_SV':'S × V interaction'}


def validate_summary(path):
    path = Path(path).resolve()
    result = e.read(path)
    e.require(e.read(path.parent/'COMPLETED.json')['summary'] == e.ref(path), 'SUMMARY_HASH_CHANGED')
    e.require(result['status'] == 'COMPLETE_ORIGINAL_FORMAL_ANALYSIS_IN_PAUSED_SCOPE'
              and result['study_stage'] == 'formal' and result['independent_tasks'] == 384
              and result['trials'] == 3456 and result['all_registered_trials_included'], 'COMPLETE_ANALYTIC_SUMMARY_REQUIRED')
    for ref in result['tables'].values():
        e.checked_reference(ref)
    e.require(len(result['methods']) == 9 and all(m['independent_tasks'] == 384 for m in result['methods'])
              and len(result['primary_contrasts']) == 4, 'ORIGINAL_DENOMINATORS_REQUIRED')
    return result


def build_paper_figures(summary, out):
    summary, out = Path(summary).resolve(), Path(out).resolve()
    data = validate_summary(summary)
    e.require(not out.exists(), 'FIGURE_OUTPUT_EXISTS')
    out.mkdir(parents=True)
    figures = []
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'cscwd-phase8-v1'}):
        def save(fig, name, caption, values, rq):
            table = out/(name+'.csv')
            with table.open('x', newline='') as h:
                writer = csv.DictWriter(h,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
            outputs = []
            for extension in ('png','svg','pdf'):
                p = out/(name+'.'+extension)
                meta = {'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None} if extension=='svg' else None
                fig.savefig(p, dpi=200, facecolor='white', metadata=meta)
                outputs.append(e.ref(p))
            plt.close(fig)
            figures.append(dict(id=name,rq=rq,caption=caption,summary=e.ref(summary),table=e.ref(table),
                                outputs=outputs,filter='None. All registered methods / contrasts.',
                                independent_unit='original task',independent_tasks=384))
        methods = data['methods']; y=np.arange(len(methods)); labels=[LABELS[m['method']] for m in methods]
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        rates=[m['certified_ITT']/384*100 for m in methods]
        ax.barh(y,rates,color='#0072B2');ax.set_yticks(y,labels);ax.invert_yaxis()
        for n,(r,m) in enumerate(zip(rates,methods)):
            ax.text(r+1,n,f"{m['certified_ITT']}/384",va='center',fontsize=9)
        ax.set(xlim=(0,100),xlabel='Certified completion (% of 384 original tasks)',title='Held-out formal results: all nine methods')
        save(fig,'certified_completion','ITT: METHOD_FAILED=0; every method retains all 384 tasks. Descriptive rates, without marginal confidence intervals.',
             [dict(method=m['method'],certified=m['certified_ITT'],tasks=384,failed=m['method_failed']) for m in methods],'RQ3')
        fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained'); rows=[]
        for n,c in enumerate(data['primary_contrasts']):
            est=c['estimate']*100;ci=c['approximate_interval']
            if ci is None:
                ax.plot(est,n,marker='D',color='#333333');text=f'{est:.2f} pp; CI not estimable'
            else:
                lo,hi=[x*100 for x in ci]
                ax.errorbar(est,n,xerr=[[est-lo],[hi-est]],fmt='o',color='#0072B2',capsize=5)
                text=f'{est:.2f} [{lo:.2f}, {hi:.2f}]'
            ax.text(.99,n,text,transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=9)
            rows.append(dict(contrast=c['contrast'],estimate=c['estimate'],lower=None if ci is None else ci[0],
                             upper=None if ci is None else ci[1],interval_status=c['interval_decision'],tasks=384))
        ax.axvline(0,color='#777777',ls='--',lw=1);ax.set_yticks(range(4),[CONTRASTS[c['contrast']] for c in data['primary_contrasts']]);ax.invert_yaxis()
        ax.margins(x=.28,y=.22);ax.set(xlabel='Paired effect (percentage points)',title='Four registered contrasts · 98.75% intervals · N = 384 tasks')
        save(fig,'registered_contrasts','Original equal-stratum paired Welch–Satterthwaite kernel. Family alpha .05; four comparisons, each alpha .0125. Missing intervals remain not estimable.',rows,'RQ2/RQ3')
        fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained',sharey=True)
        for ax,key,title in [(axes[0],'false_A_method_claims','False local policy agreements'),(axes[1],'live_calls','Actual generation requests')]:
            vals=[m[key] for m in methods];ax.barh(y,vals,color='#D55E00' if key.startswith('false') else '#0072B2')
            ax.set_yticks(y,labels);ax.set_xlabel('Count across 384 tasks');ax.set_title(title,fontsize=11)
            ax.set_xlim(0,max(vals)*1.25 if max(vals) else 1)
            for n,v in enumerate(vals):ax.text(v+max(max(vals)*.015, .1),n,str(v),va='center',fontsize=9)
        axes[0].invert_yaxis()
        save(fig,'errors_and_requests','False A_method entries are policy counts, not a binary episode error rate. Central rule uses no language generation. Unknown token usage remains unresolved.',
             [dict(method=m['method'],false_A_method_claims=m['false_A_method_claims'],generation_requests=m['live_calls'],tasks=384) for m in methods],'RQ1/RQ3')
    manifest = dict(schema_version=1,status='FIGURES_GENERATED_FOR_COMPLETE_ANALYTIC_SCOPE',
        source_summary=e.ref(summary),code=e.ref(__file__),python=platform.python_version(),
        matplotlib=matplotlib.__version__,font='DejaVu Sans',figures=figures,
        venue_layout='provisional general scientific figures; publisher layout not certified',
        missing_evidence=['Complete natural replay','Financial formal experiments','validation/OOD','Market welfare frontier'])
    e.durable_json(out/'figure_manifest.json',manifest)
    return manifest
