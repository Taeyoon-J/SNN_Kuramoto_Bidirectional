"""Render an editable, code-grounded architecture figure; no measured tensors fabricated."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'collaborative_testing/generated/architecture_figures/native64_current'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
INK, MUTED, LINE = '#20252b', '#68737d', '#cbd1d6'
BLUE, GREEN, ORANGE, RED = '#427fb6', '#53a084', '#d99b45', '#c96560'
fig = plt.figure(figsize=(19, 12), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, 19), ylim=(0, 12)); ax.axis('off')

def txt(x, y, s, size=10, color=INK, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, **kw)

def box(x, y, w, h, title, sub='', color=LINE, fill='white', size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.10',
                              ec=color,fc=fill,lw=1.25))
    txt(x+w/2,y+h*.64 if sub else y+h/2,title,size,weight='bold',ha='center')
    if sub: txt(x+w/2,y+h*.28,sub,8.6,MUTED,ha='center',linespacing=1.5)

def arrow(points, color=INK, dashed=False, lw=1.2):
    for p, q in zip(points[:-2], points[1:-1]):
        ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,ls=(0,(4,3)) if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=11,
                  color=color,lw=lw,linestyle=(0,(4,3)) if dashed else '-'))

def panel(y, letter, title, subtitle):
    txt(.55,y,letter,15,BLUE,weight='bold')
    txt(1.0,y,title,15,weight='bold')
    txt(1.0,y-.31,subtitle,9,MUTED)

txt(.55,11.57,'SNN + KURAMOTO  /  NATIVE 64 × 64  /  VERIFIED REFERENCE: SW0227',9,MUTED,
    fontfamily='DejaVu Sans Mono')
txt(.55,11.10,'From image-conditioned phase dynamics to object masks',25,
    fontfamily='DejaVu Serif',weight='bold')
txt(.55,10.66,'4,096 patch nodes · 4 phase components per node · 1,024 recurrent steps · spike-derived readout',11,MUTED)
ax.plot([.55,18.45],[10.35,10.35],color=LINE,lw=.9)

panel(9.96,'a','Image encoding and graph construction',
      'The graph changes with the input image; its weights remain constant during one rollout.')
# Explicitly illustrative image, not a checkpoint tensor.
x,y,w,h=.65,7.73,1.48,1.48
ax.add_patch(Rectangle((x,y),w,h,fc='#e8e8e3',ec=LINE))
ax.add_patch(Circle((x+.40,y+.70),.19,fc=ORANGE,ec='white'))
ax.add_patch(Rectangle((x+.76,y+.32),.35,.44,fc=GREEN,ec='white'))
ax.add_patch(Circle((x+1.12,y+1.04),.16,fc=BLUE,ec='white'))
for v in np.linspace(0,w,9):
    ax.plot([x+v,x+v],[y,y+h],color='white',alpha=.55,lw=.5)
    ax.plot([x,x+w],[y+v,y+v],color='white',alpha=.55,lw=.5)
txt(x+w/2,7.47,'RGB image · 128 × 128',9,ha='center')
txt(x+w/2,7.24,'Schematic input',8,MUTED,ha='center')
box(2.65,7.95,2.12,1.15,'CNN encoder','Trainable in latest adaptation',BLUE,'#eff5fb')
arrow([(2.17,8.48),(2.63,8.48)])
box(5.28,7.95,2.32,1.15,'Feature maps γ','8 × 64 × 64 → 8 × 4,096')
arrow([(4.8,8.48),(5.25,8.48)])
box(8.40,8.57,3.18,.95,'Image-conditioned graph A','Learned similarity + spatial geometry')
box(8.40,7.10,3.18,.95,'Sensory projection 8 → 4','Trainable · normalize + tanh',BLUE,'#eff5fb')
arrow([(7.64,8.48),(7.96,8.48),(7.96,9.04),(8.37,9.04)])
arrow([(7.96,8.48),(7.96,7.58),(8.37,7.58)])
box(12.40,7.62,5.86,1.55,'Graph-coupled oscillator field',
    '4,096 nodes × 4 phase components\nImage-dependent drive + coupling + phase pinning',GREEN,'#f1f7f3',13)
arrow([(11.62,9.04),(12.0,9.04),(12.0,8.66),(12.37,8.66)])
arrow([(11.62,7.58),(12.0,7.58),(12.0,8.04),(12.37,8.04)])
txt(15.33,7.26,'Graph / oscillator parameters frozen in latest teacher adaptation',8.5,MUTED,ha='center')

ax.plot([.55,18.45],[6.82,6.82],color=LINE,lw=.8)
panel(6.45,'b','Recurrent phase-to-spike computation',
      'One step shown below; state is carried across 1,024 steps. Four phase components are processed separately.')
nodes=[(.70,2.65,'Kuramoto phases','θ(t) ∈ ℝ⁴ per patch',GREEN,'#f1f7f3'),
       (3.92,2.70,'Sinusoidal gate','Carrier sin θ · delayed gate u',ORANGE,'#fdf6eb'),
       (7.18,2.70,'Dendritic state','4 branches · gated integration',ORANGE,'#fdf6eb'),
       (10.44,2.70,'Membrane / LIF','Leaky state · reset · threshold',RED,'#fcf1ef'),
       (13.70,2.70,'Component spikes','S = H(v − threshold) · u',BLUE,'#eff5fb')]
for xx,ww,tt,ss,cc,ff in nodes: box(xx,4.70,ww,1.0,tt,ss,cc,ff)
for a,b in zip(nodes[:-1],nodes[1:]): arrow([(a[0]+a[1]+.03,5.2),(b[0]-.04,5.2)])
txt(17.45,5.26,'× 1,024',12,weight='bold',ha='center')
txt(17.45,4.93,'time steps',8.5,MUTED,ha='center')
arrow([(15.07,4.67),(15.07,4.10),(8.52,4.10),(8.52,4.67)],BLUE)
arrow([(11.79,4.10),(11.79,4.67)],BLUE)
txt(11.8,4.25,'ACTIVE: previous spike enters dendrite and membrane reset',8.5,BLUE,ha='center')
arrow([(14.55,4.67),(14.55,3.55),(2.0,3.55),(2.0,4.67)],RED,True)
txt(8.22,3.70,'OFF: spike → Kuramoto pulse feedback',8.7,RED,ha='center')
txt(17.03,3.68,'OFF: synchrony → graph',8.5,RED,ha='center')
txt(.76,3.12,
    r'$\theta_{i,c}^{t+1}=\theta_{i,c}^{t}+\Delta t\,[\omega_{i,c}+g_f d_{i,c}'
    r'+\frac{K}{N}\sum_j\widetilde{A}_{ij}\sin(\theta_{j,c}^{t}-\theta_{i,c}^{t}-\alpha_{ij})'
    r'+\kappa_{i,c}\sin(d_{i,c}-\theta_{i,c}^{t})]$',13)
txt(.76,2.72,'d: projected sensory drive   ·   Ã: symmetrized positive coupling   ·   α: learned-direction / graph-cost phase lag',8.5,MUTED)

ax.plot([.55,18.45],[2.39,2.39],color=LINE,lw=.8)
panel(2.02,'c','Spike-based grouping and mask readout',
      'Use the last 512 steps. Ground-truth masks do not enter inference; matching is used only for evaluation.')
readouts=[(.72,2.80,'Component spike traces','4 × 4,096 × 512'),
          (4.02,3.02,'Spike synchrony','Component-wise correlation'),
          (7.54,3.02,'Spatial components','Foreground support'),
          (11.06,3.02,'Graph partition','Synchrony + A · preserve foreground'),
          (14.58,3.02,'Object masks','64 × 64 hard labels')]
for xx,ww,tt,ss in readouts: box(xx,.43,ww,.91,tt,ss,size=10.3)
for a,b in zip(readouts[:-1],readouts[1:]): arrow([(a[0]+a[1]+.03,.88),(b[0]-.04,.88)])

stem=OUT/'native64_architecture'
for extension in ('svg','pdf','png'):
    fig.savefig(stem.with_suffix('.'+extension),dpi=220,facecolor='white')
plt.close(fig)

# Companion data/provenance preserves precise scores without crowding the diagram.
summary_path=ROOT/'collaborative_test/SW_0227_native64_joint_projection_full320/results_archive/completed_v1/summary.json'
scores=json.loads(summary_path.read_text(encoding='utf-8'))['scores']
manifest={
 'reference':'SW0227 / SW0223 checkpoint lineage',
 'visual_type':'Architecture schematic; no measured activation panels or sample predictions',
 'training_scope':'Latest teacher adaptation trains encoder and gamma 8-to-4 projection; core and graph parameters frozen. Mask-supervised lineage.',
 'feedback':{'spike_to_phase':'disabled','synchrony_to_graph':'disabled','previous_spike_local_recurrence':'present; causal benefit not established by this figure'},
 'evaluation_scope':'Fixed 320 validation images, IDs 1320..1639, three-seed mean; reused validation, not an untouched test set. Same evaluation, different training supervision.',
 'scores':scores,
 'sources':['collaborative_test/SW_0192_native64_incumbent_mini/model64.py',
 'collaborative_test/SW_0175_critical_component_binding/train_v2.py',
 'collaborative_test/SW_0134_native_spike_binding/rollout.py',
 'collaborative_test/SW_0130_phase_state_integration/model.py',
 'snn_kuramoto_bidirectional/kuramoto_layer.py',
 'snn_kuramoto_bidirectional/sinusoidal_gating.py',
 'collaborative_test/SW_0193_native64_probe_screen/run.py']}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(stem.with_suffix('.png'))
