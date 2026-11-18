"""Render archived data and explicit schedule arithmetic; no simulation/API calls.
Run with NumPy and Matplotlib from any directory. Palette/type follow the
user's KDD27_Economica reference; empirical values come only from source_data.
"""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, TABLE = ROOT/'source_data', ROOT/'figures', ROOT/'tables'
COLORS = ['#202020', '#507DB3', '#EDA600', '#F38555', '#949494']
HATCHES = ['///', 'xx', '...', '\\\\', '++']
MARKERS = ['o', 's', '^', 'D', 'v']
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10.5, 'axes.labelsize': 11,
    'axes.titlesize': 11.5, 'legend.fontsize': 9, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .8, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'savefig.dpi': 300, 'svg.hashsalt': 'civitas-real-20261115',
})

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    with (DATA/name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def style(ax, axis='y'):
    ax.grid(axis=axis, color='#dedede', linewidth=.55)
    ax.set_axisbelow(True)

rendered = []

def save(fig, name, note=None):
    if note:
        fig.text(.5, .014, note, ha='center', va='bottom', fontsize=8, color='#555555')
    for ext in ('pdf', 'png', 'svg'):
        metadata = {'Creator': 'scripts/render_real_figures.py'}
        if ext == 'pdf':
            metadata.update(CreationDate=None, ModDate=None,
                            Subject='Archived evidence or explicitly labeled arithmetic; no new experiment')
        if ext == 'svg':
            metadata['Date'] = None
        fig.savefig(FIG/(name+'.'+ext), bbox_inches='tight', pad_inches=.08, metadata=metadata)
    rendered.append(name)
    plt.close(fig)

def table(name, caption, headers, rows, widths):
    text = '% Generated from packaged archive data by scripts/render_real_figures.py\n'
    text += '\\begin{table*}[t]\n\\centering\\small\n\\caption{'+caption+'}\n'
    text += '\\label{tab:'+name+'}\n\\begin{tabularx}{\\textwidth}{@{}'+widths+'@{}}\n\\toprule\n'
    text += ' & '.join(headers)+r' \\'+'\n\\midrule\n'
    text += '\n'.join(' & '.join(row)+r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabularx}\n\\end{table*}\n'
    (TABLE/('tab_'+name+'.tex')).write_text(text)

# Preserve all Figure 1 source/raster/vector bytes.
figure1_hashes = {p.name: digest(p) for p in FIG.glob('fig1_methodology_overview.*')}
assert figure1_hashes
health = read('health.csv')
seeds = [932201, 932202, 932203]
groups = ['full', 'zero_intelligence', 'homogeneous_linear_value']
labels = ['Full model', 'Zero intelligence', 'Linear value']
idx = {(int(r['seed']), r['baseline']): r for r in health}
assert len(idx) == 9 and set(idx) == {(s, g) for s in seeds for g in groups}
assert all(r['higher_moment_status'] == 'UNDEFINED_ONE_MODEL_DAY' for r in health)
assert all(int(r['committed_ticks']) == 100 and float(r['cash_residual_minor']) == 0 for r in health)

# Figure 2: paired observations, means, and the corresponding seed differences.
fig, axes = plt.subplots(2, 2, figsize=(10.7, 5.9))
fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.93, wspace=.28, hspace=.43)
for col, (metric, title, ymax) in enumerate([
    ('no_trade_ratio', 'No-trade frequency', 90),
    ('two_sided_coverage', 'Two-sided quote coverage', 30),
]):
    values = np.array([[100*float(idx[s, g][metric]) for g in groups] for s in seeds])
    ax = axes[0, col]
    for k in range(3):
        ax.bar(k, values[:, k].mean(), width=.58, color=COLORS[k], alpha=.25,
               edgecolor=COLORS[k], hatch=HATCHES[k], linewidth=.7, zorder=2)
        ax.text(k, values[:, k].max()+ymax*.045, f'{values[:, k].mean():.2f}',
                ha='center', fontsize=9)
    for j in range(3):
        ax.plot(np.arange(3)+(j-1)*.065, values[j], color=COLORS[j], marker=MARKERS[j],
                markersize=5.5, linewidth=1.1, markeredgecolor='white', markeredgewidth=.6, zorder=4)
    ax.set(xticks=range(3), xticklabels=labels, ylabel='Committed steps (%)',
           ylim=(0, ymax), title=f'({chr(97+col)}) {title}')
    style(ax)
    ax = axes[1, col]
    differences = values[:, :1]-values[:, 1:]
    for j in range(3):
        ax.plot(np.array([0, 1])+(j-1)*.065, differences[j], color=COLORS[j], marker=MARKERS[j],
                markersize=6, linewidth=1.1, zorder=4)
    ax.axhline(0, color='#666666', linewidth=.8, linestyle='--')
    ax.set(xticks=[0, 1], xticklabels=['Full − zero intelligence', 'Full − linear value'],
           ylabel='Difference (percentage points)', xlim=(-.35, 1.35),
           ylim=(-8.5, 8.5), yticks=[-8, -4, 0, 4, 8], title=f'({chr(99+col)}) Paired {title.lower()}')
    style(ax)
fig.legend([Line2D([], [], color=COLORS[j], marker=MARKERS[j], lw=1.1) for j in range(3)],
           [f'Seed {s}' for s in seeds], loc='lower center', bbox_to_anchor=(.5, .057),
           ncol=3, frameon=False)
save(fig, 'fig2_rq1_stylized_facts', 'Lines pair the same seed; offsets separate tied points; translucent bars show means.')
rows = []
for g, label in zip(groups, ['Civitas full model', 'Zero intelligence', 'Homogeneous linear value']):
    values = []
    for metric in ['no_trade_ratio', 'one_sided_ratio', 'two_sided_coverage', 'trade_count']:
        a = np.array([float(idx[s, g][metric]) for s in seeds])*(1 if metric == 'trade_count' else 100)
        values.append(f'{a.mean():.2f} [{a.min():.0f}, {a.max():.0f}]')
    rows.append([label, '3']+values)
table('rq1_results', r'\tr{RQ1 日内市场活动与盘口流动性。每种模型包含三个匹配种子的轨迹（$N=100,T=100$）。各指标报告均值及 [最小值, 最大值]。}{RQ1 Intraday Market Activity and Quoted Liquidity. Each model has three matched-seed trajectories ($N=100,T=100$). Entries report mean [minimum, maximum].}',
      ['Baseline', 'Worlds', r'No trade (\%)', r'One-sided book (\%)', r'Two-sided quotes (\%)', 'Trades'], rows, 'l r X X X X')

branches = read('branch_metrics.csv')
bn = {r['condition']: r for r in branches}
paths = read('branch_paths.csv')
steps = read('branch_step_metrics.csv')
sidx = {(r['condition'], int(r['event_step'])): r for r in steps}
names = {'full_control':'Full control', 'no_secondary_contagion':'Secondary spread off',
    'no_herding_feedback':'No herding', 'both_off':'Both off', 'full_loss_off':'Loss feedback off',
    'lambda_only_1':r'$\lambda=1$', 'reverse_retrospective_mapping':'Valuation sign reversed',
    'halt25':'Halt (25 steps)', 'tax0015':'Seller tax 0.15%', 'same_day_lock':'T+1 lock',
    'narrow_band005':'Price band 5%'}
assert len(branches) == 11 and set(bn) == set(names) and len(sidx) == 308
assert set(sidx) == {(c, u) for c in names for u in range(28)}
step_provenance = json.loads((DATA/'branch_step_metrics_provenance.json').read_text())
assert step_provenance['artifact_sha256'] == digest(DATA/'branch_step_metrics.csv')
assert step_provenance['assertions']['raw_state_hashes'] == 308
for row in paths:
    assert math.isclose(float(row['price']), float(sidx[row['condition'], int(row['event_step'])]['price']), abs_tol=1e-12)
for condition, r in bn.items():
    p = np.array([float(sidx[condition, u]['price']) for u in range(28)])
    assert math.isclose(np.max(1-p/np.maximum.accumulate(p)), float(r['mdd']), abs_tol=1e-12)
    assert max(float(sidx[condition, u]['infected_percent']) for u in range(1, 28)) == float(r['peak_infection_percent'])
    assert sidx[condition, 0]['volume_shares'] == ''
interaction = sum(sign*float(bn[c]['mdd']) for c, sign in [
    ('full_control', 1), ('no_secondary_contagion', -1), ('no_herding_feedback', -1), ('both_off', 1)])
assert interaction == 0 and bn['halt25']['tradable_steps'] == '2'

# Figure 3: five event trajectories and all eleven branch outcome rows.
assert {float(sidx[c, 0]['infected_percent']) for c in bn} == {30.0}
assert {float(sidx[c, 1]['infected_percent']) for c in bn} == {5.0}
chosen = ['full_control', 'no_secondary_contagion', 'reverse_retrospective_mapping', 'halt25']
fig = plt.figure(figsize=(10.7, 7.0))
grid = fig.add_gridspec(3, 6, left=.19, right=.985, top=.905, bottom=.095,
                        hspace=.60, wspace=1.05, height_ratios=[1, 1, 1.6])
axes = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]), fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 0:3]), fig.add_subplot(grid[1, 3:6])]
metrics = ['price', 'infected_percent', 'confidence_mean', 'emotion_mean', 'volume_shares']
ylabels = ['Price (pre-event = 100)', 'Infected agents (%)', 'Mean confidence', 'Mean emotion', 'Volume (shares)']
titles = ['(a) Transaction price', '(b) Infection', '(c) Confidence', '(d) Internal emotion', '(e) Trading activity']
styles = ['-', '--', '-.', ':']
for ax, metric, ylabel, title in zip(axes, metrics, ylabels, titles):
    for k, condition in enumerate(chosen):
        us = np.arange(1, 28) if metric in ('infected_percent', 'volume_shares') else np.arange(28)
        y = np.array([float(sidx[condition, int(u)][metric]) for u in us])
        if metric == 'price':
            y = 100*y/float(sidx[condition, 0]['price'])
        ax.plot(us, y, color=COLORS[k], marker=MARKERS[k], markevery=5, markersize=3.3,
                linewidth=1.25, linestyle=styles[k], label=names[condition])
    ax.axvline(1, color='#bbbbbb', lw=.7, zorder=0)
    ax.set(xlim=(-.4, 27.4), xticks=[0, 10, 20, 27], xlabel='Event step', ylabel=ylabel, title=title)
    ax.tick_params(labelsize=8.5)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.title.set_size(10)
    style(ax)
axes[0].set_ylim(98, 105.2)
axes[1].plot(0, float(sidx['full_control', 0]['infected_percent']), marker='o',
             markerfacecolor='white', markeredgecolor=COLORS[0], markersize=4.5, linestyle='none')
axes[1].set_ylim(-2, 50)
axes[1].set_yticks([0, 25, 50])
axes[2].set_ylim(.50, .61)
axes[2].set_yticks([.50, .55, .60])
axes[3].set_ylim(-.28, .14)
axes[3].axhline(0, color='#bbbbbb', lw=.7)
axes[4].set_ylim(bottom=-1)
fig.legend([Line2D([], [], color=COLORS[k], marker=MARKERS[k], lw=1.25, linestyle=styles[k]) for k in range(4)],
           [names[c] for c in chosen], loc='upper center', bbox_to_anchor=(.57, .991), ncol=2,
           frameon=False, fontsize=9, columnspacing=1.5)
outcome_axes = [fig.add_subplot(grid[2, start:start+2]) for start in [0, 2, 4]]
for ax, metric, title, xlabel in zip(outcome_axes,
        ['mdd', 'relative_turnover', 'restricted_time'],
        ['(f) Maximum drawdown', '(g) Relative turnover', '(h) Confirmation'],
        ['Drawdown (%)', 'Relative to full control', 'Restricted time (steps)']):
    for j, r in enumerate(branches):
        value = float(r[metric])*(100 if metric == 'mdd' else 1)
        color = COLORS[chosen.index(r['condition'])] if r['condition'] in chosen else COLORS[4]
        marker = '>' if metric == 'restricted_time' and r['censored'] == 'True' else 'o'
        ax.plot(value, j, marker=marker, color=color, markersize=4.3, linestyle='none')
        if metric == 'restricted_time':
            ax.text(29.3, j, r['tradable_steps']+'/27', fontsize=7.5, va='center')
    ax.set(ylim=(10.7, -.7), yticks=range(11), xlabel=xlabel, title=title)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.title.set_size(10)
    style(ax, 'x')
outcome_axes[0].set_yticklabels([names[r['condition']] for r in branches], fontsize=8)
for ax in outcome_axes[1:]:
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
outcome_axes[0].set_xlim(-.05, 1.82)
outcome_axes[0].set_xticks([0, .5, 1, 1.5])
outcome_axes[1].set_xlim(-.03, 1.95)
outcome_axes[1].set_xticks([0, .5, 1, 1.5])
outcome_axes[2].set_xlim(0, 38)
outcome_axes[2].set_xticks([0, 10, 20, 27])
outcome_axes[2].text(29.3, -1.2, 'Tradable', fontsize=7.5)
save(fig, 'fig3_rq2_mechanisms_policy', 'One parent world; 27 event steps.  |  >: right censoring; confirmation-panel labels: tradable steps')
rows = []
for r in branches:
    recovery = r['restricted_time']+(r'$^{*}$' if r['censored'] == 'True' else '')
    rows.append([names[r['condition']].replace('%', r'\%'), f"{100*float(r['mdd']):.4f}",
        f"{float(r['peak_infection_percent']):.0f}", f"{float(r['relative_turnover']):.4f}",
        recovery, 'Yes' if r['censored'] == 'True' else 'No', r['tradable_steps']+'/27'])
table('rq2_results', r'\tr{RQ2 共同父世界的十一条干预分支（$N=20$，27 步事件窗口）。感染峰值取事件窗口内逐步感染比例的最大值。星号标记窗口终点的右删失。}{RQ2 Outcomes for Eleven Intervention Branches from a Shared Parent World ($N=20$, 27 event steps). Peak infection is the maximum stepwise infection share within the event window. An asterisk marks right censoring at the window endpoint.}',
      ['Intervention', r'MDD (\%)', r'Peak infection (\%)', 'Relative turnover', 'Confirmation (steps)', 'Censored', 'Tradable steps'], rows, 'l c c c c c X')

cases, pairs, capacity = read('snapshot_cases.csv'), read('snapshot_pairs.csv'), read('capacity.csv')
backends = ['rule', 'heuristic', 'gpt6_luna_live']
backend_labels = ['Rules', 'Heuristics', 'Live LLM']
facts = ['price', 'locked_inventory', 'cash']
assert len(cases) == 162 and len(pairs) == 81
live = [r for r in cases if r['backend'] == 'gpt6_luna_live']
assert len(live) == 54 and all(r['action'] == 'hold' for r in live)
assert sum(int(r['input_tokens']) for r in live) == 119637
assert sum(int(r['output_tokens']) for r in live) == 9306
changed = {}
for backend in backends:
    changed[backend] = []
    for fact in facts:
        pp = [r for r in pairs if r['backend'] == backend and r['fact_pair'] == fact]
        assert len(pp) == 9
        changed[backend].append(sum(r['economic_action_changed'] == 'True' for r in pp))
assert changed == {'rule':[9, 0, 9], 'heuristic':[7, 1, 8], 'gpt6_luna_live':[0, 0, 0]}
mac = sorted([r for r in capacity if r['host'] == 'Mac'], key=lambda r:int(r['requested_ticks']))
assert len(mac) == 2
phase_values = []
fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.85))
fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.85, wspace=.33)
for k, backend in enumerate(backends):
    xx = np.arange(3)+(k-1)*.23
    axes[0].bar(xx, changed[backend], width=.21, color=COLORS[k], edgecolor='white',
                hatch=HATCHES[k], label=backend_labels[k])
    for x, y in zip(xx, changed[backend]):
        axes[0].text(x, y+.18, str(y), ha='center', fontsize=9)
axes[0].set(xticks=range(3), xticklabels=['Price', 'Inventory lock', 'Cash'],
            ylabel='Changed signatures (of 9 pairs)', ylim=(0, 11.5),
            title='(a) Fixed-observation sensitivity')
axes[0].legend(frameon=False, ncol=3, loc='upper center', fontsize=8.5)
style(axes[0])
for k, r in enumerate(mac):
    total = float(r['profile_total_seconds'])
    phase = [float(r['step_wall_seconds']), float(r['checkpoint_wall_seconds'])]
    phase.append(total-sum(phase))
    assert phase[-1] > 0 and math.isclose(sum(phase), total, abs_tol=1e-9)
    phase_values.append({'requested_ticks':int(r['requested_ticks']),
        'step_loop_seconds':phase[0], 'terminal_checkpoint_seconds':phase[1],
        'other_seconds':phase[2], 'total_seconds':total})
    bottom = 0
    for j, value in enumerate(phase):
        axes[1].bar(k, value/60, bottom=bottom, width=.55, color=COLORS[j],
                    edgecolor='white', hatch=HATCHES[j], linewidth=.5)
        bottom += value/60
    axes[1].text(k, total/60+3, f'{total/60:.2f} min\nother: {phase[-1]:.2f} s',
                 ha='center', fontsize=9, linespacing=1.3)
axes[1].set(xticks=[0, 1], xticklabels=['100 steps', '200 steps'],
            ylabel='Profile runtime (minutes)', ylim=(0, 152), title='(b) Measured rule-world runtime')
axes[1].legend([Patch(facecolor=COLORS[j], hatch=HATCHES[j], edgecolor='white') for j in range(3)],
               ['Step loop', 'Terminal checkpoint', 'Other'], frameon=False,
               loc='upper left', bbox_to_anchor=(0, 1.17), ncol=3, fontsize=8, columnspacing=.9)
style(axes[1])
save(fig, 'fig4_rq3_cognition_cost', '54 live requests on fixed observations  |  Same Mac, N=100; one measured run per horizon')
summary = json.loads((DATA/'snapshot_summary.json').read_text())
rows = [['Live snapshot batch', '54', '119,637 / 9,306', f"{summary['actual_wall_seconds']:.3f}"+'$^{a}$', '---', '54/54 responses; all HOLD']]
for r in [mac[0], mac[1], next(r for r in capacity if r['host'] == 'ASUS')]:
    n = 100 if r['host'] == 'Mac' else 1000
    seconds = f"{float(r['profile_total_seconds']):.3f}" if r['profile_total_seconds'] else 'Missing'
    memory = f"{int(r['peak_rss_bytes'])/2**30:.3f}"+('$^{b}$' if r['host'] == 'ASUS' else '')
    status = 'Runner/controller complete' if r['host'] == 'Mac' else 'Runner saved 12 steps; controller stopped'
    rows.append([f"{r['host']}: $N={n},T={r['requested_ticks']}$", '0', '0 / 0', seconds, memory, status])
table('rq3_results', r'\tr{模型请求与规则世界的实测计算记录。$^{a}$ 快照批次壁钟时间。规则世界的运行剖面包括终态检查点。$^{b}$ ASUS 内存为采样峰值，最终运行剖面缺失。}{Measured Compute for Model Requests and Rule Worlds. $^{a}$~Snapshot batch wall time. Rule-world profiles include terminal checkpoints. $^{b}$~ASUS memory is the sampled peak. Its final profile is unavailable.}',
      ['Measured workload', 'Live calls', 'Input / output tokens', 'Time (s)', 'Memory (GiB)', 'Completion status'], rows, 'l r c c c X')

# Archetype-level signed order changes; white is an actual zero, not missing data.
archetypes = ['retail', 'mutual_fund', 'pension', 'insurance', 'proprietary', 'market_maker',
              'arbitrage', 'rumor_sensitive', 'stabilization']
archetype_labels = ['Retail', 'Mutual fund', 'Pension', 'Insurance', 'Proprietary',
                    'Market maker', 'Arbitrage', 'Rumor-sensitive', 'Stabilization']
pair_index = {(r['backend'], r['archetype'], r['fact_pair']):r for r in pairs}
columns = [(backend, fact) for backend in ['rule', 'heuristic'] for fact in facts]
values = np.array([[int(pair_index[b, a, f]['signed_quantity_delta']) for b, f in columns] for a in archetypes])
assert values.shape == (9, 6) and values.min() == -102 and values.max() == 10
fig, ax = plt.subplots(figsize=(10.7, 4.5))
fig.subplots_adjust(left=.18, right=.87, bottom=.19, top=.84)
cmap = LinearSegmentedColormap.from_list('civitas_diverging', [COLORS[1], '#ffffff', COLORS[2]])
im = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-102, vcenter=0, vmax=102), aspect='auto')
quote_only = []
for i, a in enumerate(archetypes):
    for j, (backend, fact) in enumerate(columns):
        r = pair_index[backend, a, fact]
        dagger = int(r['signed_quantity_delta']) == 0 and r['quote_delta_ticks'] not in ('', '0')
        if dagger:
            quote_only.append({'backend':backend, 'archetype':a, 'fact':fact,
                               'quote_delta_ticks':int(r['quote_delta_ticks'])})
        label = str(values[i, j])+('†' if dagger else '')
        ax.text(j, i, label, ha='center', va='center', fontsize=10,
                color='white' if abs(values[i, j]) > 70 else '#202020')
ax.set(xticks=range(6), xticklabels=['Price', 'Inventory lock', 'Cash']*2,
       yticks=range(9), yticklabels=archetype_labels)
ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
ax.grid(which='minor', color='#e5e5e5', linewidth=.7)
ax.tick_params(which='both', length=0)
ax.axvline(2.5, color='#555555', lw=1.0)
for x, label in [(1, 'Numerical rules'), (4, 'Persona heuristics')]:
    ax.text(x, 1.06, label, transform=ax.get_xaxis_transform(), ha='center', fontsize=11)
cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
cbar.set_label('Δ signed quantity (shares)', fontsize=10)
cbar.set_ticks([-100, -50, 0, 50, 100])
assert quote_only == [{'backend':'heuristic', 'archetype':'insurance', 'fact':'price', 'quote_delta_ticks':1002}]
save(fig, 'archetype_order_response', 'State 1 − state 0; numbers include all zero changes.  |  † Quote rises by 1,002 ticks with unchanged quantity.')

# Theoretical request opportunity reduction and archived event-tick scheduling.
schedule = read('schedule_exposure.csv')
schedule_prov = json.loads((DATA/'schedule_exposure_provenance.json').read_text())
assert schedule_prov['artifact_sha256'] == digest(DATA/'schedule_exposure.csv')
assert [int(r['seed']) for r in schedule] == list(range(20261110, 20261130))
full = np.array([int(r['full_text_requests']) for r in schedule])
periodic = np.array([int(r['periodic_text_requests']) for r in schedule])
assert full.sum() == 101 and periodic.sum() == 10 and (periodic == 0).sum() == 10
assert all(r['scheduled_only'] == '1' for r in schedule)
fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.85))
fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.88, wspace=.32)
periods = np.array([1, 2, 5, 10, 20])
for k, slow in enumerate([5, 10, 20]):
    reduction = 100*(1-slow/(100*periods))
    axes[0].plot(periods, reduction, color=COLORS[k], marker=MARKERS[k], lw=1.5,
                 markersize=5, label=f'k = {slow}')
axes[0].set(xscale='log', xticks=periods, xticklabels=[str(x) for x in periods],
            xlabel=r'Slow period $T_{\mathrm{slow}}$ (steps)', ylabel='Request opportunity reduction (%)',
            ylim=(78, 101), yticks=[80, 85, 90, 95, 100],
            title=r'Theory  $1-k/(N T_{\mathrm{slow}})$, $N=100$')
axes[0].minorticks_off()
axes[0].legend(frameon=False, loc='lower right')
style(axes[0])
x = np.arange(20)
for values, offset, color, hatch, label in [(full, -.18, COLORS[0], HATCHES[0], 'Full schedule'),
                                           (periodic, .18, COLORS[1], HATCHES[1], 'Periodic schedule')]:
    axes[1].bar(x+offset, values, width=.34, color=color, hatch=hatch, edgecolor='white', linewidth=.5, label=label)
    for xx, value in zip(x+offset, values):
        axes[1].text(xx, value+.15, str(value), ha='center', fontsize=7, color=color)
axes[1].set(xticks=x, xticklabels=[str(int(r['seed']) % 100) for r in schedule],
            xlabel='Seed (202611xx)', ylabel='Scheduled announcement requests', ylim=(0, 13),
            yticks=[0, 2, 4, 6, 8, 10, 12], title='Archived event-step schedule')
axes[1].tick_params(axis='x', labelsize=8)
axes[1].legend(frameon=False, ncol=2, loc='upper left', fontsize=8.5)
style(axes[1])
save(fig, 'schedule_exposure', r'Scheduled opportunities at tick 1,200 ($N=100$, $k=10$, $T_{\mathrm{slow}}=10$, $p_{\mathrm{act}}=0.05$).')

# Strict-validator timing: a separate legend avoids overlapping Pair 2/3 labels.
timing = read('timing_pairs.csv')
assert len(timing) == 3
fig, ax = plt.subplots(figsize=(5.5, 3.25))
fig.subplots_adjust(left=.15, right=.67, bottom=.23, top=.87)
for k, r in enumerate(timing):
    a, b = float(r['A_seconds']), float(r['B_seconds'])
    reduction = float(r['reduction_percent'])
    assert math.isclose(100*(1-b/a), reduction, abs_tol=1e-10)
    ax.plot([0, 1], [a, b], color=COLORS[k], marker=MARKERS[k], lw=1.6, markersize=5,
            label=f"Pair {r['pair']}: −{reduction:.2f}%")
ax.set(xticks=[0, 1], xticklabels=['Generic', 'Scalar-optimized'], ylabel='Total wall time (seconds)',
       title='Strict-validation timing', xlim=(-.1, 1.1), ylim=(25.65, 26.95))
ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, .5), fontsize=9)
style(ax)
save(fig, 'engineering_timing', 'Three timing repeats of the same rule world; N=100, T=12')

# Remove only figures replaced by the integrated main panels.
removed = []
for stem in ['health_diagnostics', 'engineering_branches']:
    for ext in ['pdf', 'png', 'svg']:
        path = FIG/(stem+'.'+ext)
        if path.exists():
            path.unlink()
            removed.append(path.name)
assert figure1_hashes == {p.name:digest(p) for p in FIG.glob('fig1_methodology_overview.*')}
receipt = {
    'health_trajectories':9, 'health_independent_seeds':3, 'branch_parent_worlds':1,
    'branches':11, 'branch_step_rows':308, 'branch_step_extraction_assertions':step_provenance['assertions'],
    'drawdown_interaction':interaction, 'live_requests':54, 'live_changed_action_pairs':0,
    'snapshot_changed_counts_by_price_lock_cash':changed, 'quote_only_quantity_zero_cells':quote_only,
    'capacity_phase_seconds':phase_values, 'schedule_seed_count':20,
    'schedule_full_event_requests':int(full.sum()), 'schedule_periodic_event_requests':int(periodic.sum()),
    'schedule_periodic_zero_seeds':int((periodic == 0).sum()),
    'new_simulations':0, 'new_model_calls':0, 'confidence_intervals_drawn':False,
    'theory_panel':'Explicit arithmetic only: 100*(1-k/(N*L)); not a measured cost saving',
    'figure3_canvas_inches':[10.7,7.0], 'figure3_selected_trajectories':chosen,
    'style_reference':'KDD27_Economica reference palette and serif typography',
    'figure1_unchanged_sha256':figure1_hashes, 'rendered_figure_stems':rendered,
    'replaced_figure_stems':['health_diagnostics','engineering_branches'],
    'source_hashes':{p.name:digest(p) for p in sorted(DATA.glob('*.csv'))},
    'outputs_sha256':{p.name:digest(p) for name in rendered for p in sorted(FIG.glob(name+'.*'))},
    'generated_table_sha256':{p.name:digest(p) for p in sorted(TABLE.glob('tab_rq[123]_results.tex'))},
}
(DATA/'render_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('Rendered six figure families and three tables; all archive/data assertions passed; Figure 1 unchanged.')
