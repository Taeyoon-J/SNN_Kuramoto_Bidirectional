"""Self-contained visual architecture, with explicitly illustrative internal tensors."""
from pathlib import Path
from io import BytesIO
import base64
import html
import json
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'collaborative_testing/generated/architecture_figures/native64_interactive'
OUT.mkdir(parents=True, exist_ok=True)

def uri(im):
    f=BytesIO(); im.save(f,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(f.getvalue()).decode()

# Archived real input only. No archived GT mask or previous model prediction is used.
source=ROOT/'collaborative_testing/server_results/clevr_patch_masks_3images/sample_0000.png'
rgb=Image.open(source).convert('RGB').crop((7,52,720,765)).resize((128,128))
small=np.asarray(rgb.resize((64,64)),dtype=float)/255
yy,xx=np.mgrid[:64,:64]
labels=np.zeros((64,64),dtype=np.uint8)
labels[((xx-17)**2+(yy-17)**2)<13]=1
labels[((xx-50)/7)**2+((yy-23)/8)**2<1]=2
labels[(xx>14)&(xx<25)&(yy>30)&(yy<44)]=3
labels[(xx>32)&(xx<47)&(yy>27)&(yy<47)]=4
labels[(xx>18)&(xx<44)&(yy>41)&(yy<61)]=5
palette=np.array([[245,245,242],[79,154,111],[92,136,190],[214,116,97],[72,173,183],[213,169,70]],np.uint8)
assets={'rgb':uri(rgb),'labels':uri(Image.fromarray(palette[labels]))}
for k in range(1,6):
    arr=np.full((64,64,3),248,dtype=np.uint8);arr[labels==k]=palette[k]
    assets['mask'+str(k)]=uri(Image.fromarray(arr))
for k in range(8):
    a=small[:,:,k%3]*(.7+.03*k)+.18*np.sin((xx+yy)*(k+1)/50)
    assets['gamma'+str(k)]=uri(Image.fromarray((colormaps['viridis'](np.clip(a,0,1))[:,:,:3]*255).astype('uint8')))
frames=[]
for ti,t in enumerate([1,129,513,1024]):
    maps=[]
    for c in range(4):
        # Deliberately schematic, not a surrogate call to the scientific model.
        phase=.18*xx+.08*yy+labels*.8+c*.9+t*.023
        arr=(np.cos(phase)+1)/2
        data=uri(Image.fromarray((colormaps['viridis'](arr)[:,:,:3]*255).astype('uint8')))
        assets[f'phase{ti}_{c}']=data;maps.append(data)
    frames.append(maps)
rng=np.random.RandomState(27)
raster=np.ones((100,160,3),dtype=np.uint8)*250
for r in range(24):
    times=np.arange((r*3)%19,160,19)+rng.randint(-2,3,size=len(np.arange((r*3)%19,160,19)))
    for t in times:
        if 0<=t<160:raster[r*4:r*4+3,t]=[35,43,52]
assets['spikes']=uri(Image.fromarray(raster))
q=np.exp(-np.abs(np.arange(64)[:,None]//12-np.arange(64)[None,:]//12)*1.2)
assets['q']=uri(Image.fromarray((colormaps['magma'](q)[:,:,:3]*255).astype('uint8')))

P=[]
def emit(s):P.append(s)
def text(x,y,s,size=16,fill='#20272e',anchor='start',weight='normal',family='Arial'):
    emit(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(s)}</text>')
def rect(x,y,w,h,fill='white',stroke='#c8cdd2',radius=9):
    emit(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')
def image(x,y,w,h,key,id=''):
    emit(f'<image {f"id={chr(34)}{id}{chr(34)}" if id else ""} x="{x}" y="{y}" width="{w}" height="{h}" href="{assets[key]}" style="image-rendering:pixelated"/>')
def path(points,color='#3e4851',dash=False):
    emit(f'<path d="{points}" fill="none" stroke="{color}" stroke-width="1.6" {"stroke-dasharray="+chr(34)+"6 5"+chr(34) if dash else ""} marker-end="url(#{"red" if color=="#c4645e" else "arrow"})"/>')
def group(key):emit(f'<g class="stage" tabindex="0" role="button" data-stage="{key}">')
def end():emit('</g>')
def block(x,y,w,h,title,sub,color='#c8cdd2',fill='white',key=None):
    if key:group(key)
    rect(x,y,w,h,fill,color);text(x+w/2,y+h/2-3,title,15,anchor='middle',weight='bold')
    text(x+w/2,y+h/2+20,sub,11,'#63717d','middle')
    if key:end()

emit('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1120" width="1600" height="1120" aria-label="Native64 SNN Kuramoto architecture">')
emit('<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#3e4851"/></marker><marker id="red" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#c4645e"/></marker></defs>')
rect(0,0,1600,1120,'white','white',0)
text(38,32,'SNN + KURAMOTO  /  NATIVE64  /  SW0227 REFERENCE',12,'#7b858c',family='monospace')
text(38,76,'Image-conditioned rhythms → spike-derived object masks',29,weight='bold',family='Georgia')
text(38,105,'4,096 patch nodes · 4 phase components per node · 1,024 recurrent steps',15,'#67737d')
text(38,133,'Visual guide: input RGB is archived data; all internal maps, traces and output masks below are illustrative.',12,'#a16b34')

text(38,178,'(a) Image conditioning and oscillator field',21,weight='bold',family='Georgia')
group('input'); image(42,255,154,154,'rgb');rect(42,255,154,154,'none')
for i in range(1,16):
    emit(f'<path d="M{42+i*154/16} 255 v154 M42 {255+i*154/16} h154" stroke="white" stroke-opacity=".25" stroke-width=".5"/>')
text(119,434,'RGB + patch grid',13,anchor='middle');text(119,454,'128² input → 64² nodes',11,'#67737d','middle');end()
path('M200 333 H231')
group('encoder')
emit('<path d="M236 270 L291 292 L291 372 L236 394 Z" fill="#edf4fa" stroke="#6f9fc8"/>')
text(260,335,'CNN',14,anchor='middle',weight='bold');text(263,419,'trainable',11,'#427fb6','middle');end()
path('M294 333 H330')
group('gamma')
for k in reversed(range(4)):
    rect(336+k*7,284-k*8,77,77);image(339+k*7,287-k*8,71,71,'gamma'+str(k))
text(380,390,'γ feature maps',13,anchor='middle');text(380,410,'8 × 64 × 64',11,'#67737d','middle');end()
path('M435 324 H461 V225 H482');path('M461 324 V427 H482')
block(487,194,238,73,'Image-conditioned graph A','Frozen parameters; varies per image',key='graph')
block(487,393,238,73,'Sensory projection 8 → 4','Trainable · normalized drive', '#6f9fc8','#edf4fa','projection')
path('M728 230 H756 V288 H791');path('M728 430 H756 V358 H791')
text(606,296,'A is held constant',12,'#67737d','middle');text(606,315,'within each rollout',12,'#67737d','middle')
group('phase')
for column,ti in enumerate([0,2,3]):
    for c in range(4):
        x=799+column*174;y=209+c*67
        rect(x-3,y-3,64,64,'white',['#66a886','#d4a64d','#6d9fd2','#ce8277'][c],4)
        image(x,y,58,58,f'phase{ti}_{c}',f'phase-{column}-{c}')
        if column<2:path(f'M{x+66} {y+29} H{x+155}')
        if column==0:text(884,y+33,['θ₁','θ₂','θ₃','θ₄'][c],12,'#74808b')
    text(828+column*174,497,['t = 1','t = 513','t = 1024'][column],12,'#67737d','middle')
text(1088,190,'Phase field: 4 components per patch',14,anchor='middle',weight='bold')
text(1088,525,'Illustrative phase maps — not measured synchronization',12,'#a16b34','middle');end()
text(1407,257,'θ(t + 1)',18,anchor='middle',weight='bold')
text(1407,285,'frequency + drive',12,'#67737d','middle');text(1407,307,'+ graph coupling',12,'#67737d','middle')
emit('<path d="M1380 338 A28 28 0 1 0 1428 321" fill="none" stroke="#3e4851" stroke-width="1.6" marker-end="url(#arrow)"/>')
text(1407,393,'× 1,024',15,anchor='middle');text(1407,417,'recurrent steps',12,'#67737d','middle')

emit('<path d="M38 553 H1562" stroke="#d8dde1"/>')
text(38,593,'(b) Phase-to-spike computation',21,weight='bold',family='Georgia')
row=[(42,218,'Kuramoto phases','Image-conditioned dynamics','phase','#edf6f1','#64a286'),
     (302,240,'Sinusoidal gating','sin θ × delayed gate u','gate','#fdf6ec','#d4a64d'),
     (584,230,'Dendritic state','4 branches · gated update','dendrite','#fdf6ec','#d4a64d'),
     (856,230,'Membrane / LIF','Leak · gated update · reset','membrane','#fcf1ef','#ce8277'),
     (1128,224,'Component spikes','S = H(v − threshold) × u','spikes','#edf4fa','#6f9fc8')]
for x,w,title,sub,key,fill,col in row:block(x,630,w,86,title,sub,col,fill,key)
for a,b in zip(row[:-1],row[1:]):path(f'M{a[0]+a[1]+3} 674 H{b[0]-8}')
group('spikes');image(1393,625,150,94,'spikes');text(1468,742,'Illustrative raster',11,'#a16b34','middle');end();path('M1356 674 H1381')
path('M1240 719 V763 H699 V720');path('M971 763 V720')
text(944,784,'ACTIVE: previous spike → dendrite input + membrane reset',12,'#427fb6','middle')
emit('<g id="disabled-paths">');path('M1210 719 V815 H151 V720','#c4645e',True)
text(665,838,'OFF: spike → Kuramoto pulse feedback',12,'#c4645e','middle')
text(1365,816,'OFF: synchrony → graph',12,'#c4645e','middle');end()

emit('<path d="M38 858 H1562" stroke="#d8dde1"/>')
text(38,895,'(c) Late spikes → foreground support → object masks',21,weight='bold',family='Georgia')
group('readout');image(42,929,121,76,'spikes');text(103,1027,'Last 512 steps',12,'#67737d','middle');end()
path('M171 969 H201')
group('readout');image(212,929,76,76,'q');text(250,1027,'Spike synchrony',12,'#67737d','middle');end()
path('M298 969 H330')
block(340,929,221,76,'Spatial components','Foreground support',key='readout')
path('M568 969 H599')
block(607,929,244,76,'Graph partition','Synchrony + A · preserve support',key='partition')
path('M859 969 H892')
group('masks');image(903,918,100,100,'labels');text(953,1039,'64 × 64 hard labels',12,'#67737d','middle')
path('M1014 967 H1044')
for k in range(1,6):
    image(1062+(k-1)*95,934,80,80,'mask'+str(k));text(1102+(k-1)*95,1033,'object '+str(k),11,'#67737d','middle')
end()
text(1237,1062,'Illustrative masks — not model predictions',12,'#a16b34','middle')
text(38,1094,'Training: latest adaptation updates CNN + γ projection only; mask-supervised lineage. Inference: no ground-truth mask input.',12,'#67737d')
emit('</svg>')
svg='\n'.join(P)
(OUT/'native64_visual_architecture.svg').write_text(svg,encoding='utf-8')

details={
'input':('RGB와 patch grid','128×128 RGB 이미지를 입력하고 64×64개의 patch node로 처리합니다. 각 node에는 4개의 phase component가 있습니다. 여기의 RGB는 이전에 보관한 실제 데이터 이미지이며, 이 이미지로 현재 checkpoint를 재실행한 결과를 보여주는 것은 아닙니다.'),
'encoder':('CNN encoder','최근 teacher adaptation에서 encoder를 학습합니다. Feature maps를 생성하며 gamma의 8개 channel은 object 8개를 의미하지 않습니다.'),
'gamma':('γ feature maps','γ shape은 [B, 8, 4,096]입니다. 표시된 map은 RGB에서 만든 설명용 이미지이며 encoder의 실측 activation이 아닙니다.'),
'graph':('Image-conditioned graph','A는 입력 γ에 따라 달라집니다. 최근 adaptation에서는 graph parameter를 고정하지만, 모든 이미지가 같은 A를 사용하는 것은 아닙니다. 하나의 1,024-step rollout 안에서는 A가 유지됩니다.'),
'projection':('학습하는 sensory projection','실제 8→4 projection을 encoder와 함께 학습합니다. 정규화와 tanh를 거친 drive가 phase 초기화와 Kuramoto dynamics에 사용됩니다. 나머지 core parameter는 최근 adaptation에서 고정합니다.'),
'phase':('Kuramoto phase dynamics','자연 주파수, image-dependent frequency drive, graph coupling, phase-pinning 항을 사용해 1,024 step을 계산합니다. 여기의 시간 슬라이더는 설명용 map을 바꾸며 checkpoint dynamics를 시뮬레이션하지 않습니다.'),
'gate':('Sinusoidal gate','현재 phase의 sin carrier를 delayed phase에서 만든 연속 gate u로 조절합니다. Gate는 dendritic state와 membrane state의 integration에도 연결됩니다. 이 그림 자체가 gating의 인과적 성능 기여를 증명하지는 않습니다.'),
'dendrite':('Dendritic state','각 phase component의 carrier와 이전 spike를 받아 4개 branch의 state를 갱신하고 합칩니다. Phase-conditioned integration으로 이전 state와 새 state의 차이를 조절합니다.'),
'membrane':('Membrane / LIF','Leaky integration, 이전 spike에 따른 reset, phase-conditioned integration을 수행합니다. Component threshold는 0.06 × exp(b_c)이며 고정 0.06으로만 동작한다고 표현하지 않습니다.'),
'spikes':('Component spikes와 feedback','Threshold event에 gate를 곱해 component spike를 만듭니다. 이전 spike는 dendrite 입력과 membrane reset에 사용됩니다. 이 checkpoint에서 spike→Kuramoto pulse와 synchrony→graph feedback은 꺼져 있습니다.'),
'readout':('Spike synchrony와 spatial components','마지막 512 step의 component spike correlation으로 grouping을 수행하고 spatial components에서 foreground support를 구합니다. Ground-truth mask는 inference에 사용하지 않습니다.'),
'partition':('Foreground-preserving graph partition','Spike-derived synchrony와 graph A로 foreground 안에서 object partition을 조정합니다. 이 단계는 이미 정해진 foreground support를 보존합니다. Hungarian matching은 여기의 model inference 단계가 아니라 Object IoU 평가에서 수행합니다.'),
'masks':('Object masks','최종 출력은 64×64 patch hard label입니다. 오른쪽 각 mask는 object별 label을 분리하는 방식을 설명하는 예시이며 실제 모델 예측이 아닙니다. GT와의 비교는 inference가 끝난 뒤 평가 단계에서만 합니다.')}
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native64 · visual architecture</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f4f5;color:#20272e;font:14px Arial,sans-serif}.toolbar{display:flex;align-items:center;gap:18px;padding:16px 24px;background:white;border-bottom:1px solid #ddd;flex-wrap:wrap}.toolbar strong{margin-right:auto}button{padding:8px 14px;border:1px solid #cbd1d6;background:white;border-radius:7px;cursor:pointer}.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:18px;padding:18px}.canvas{background:white;border:1px solid #dde1e5;border-radius:10px;overflow:hidden}.canvas svg{width:100%;height:auto;display:block}.stage{cursor:pointer;outline:none}.stage:hover{filter:drop-shadow(0 0 4px #5897c080)}.stage.selected{filter:drop-shadow(0 0 5px #3b80b4a0)}aside{background:white;border:1px solid #dde1e5;border-radius:10px;padding:22px;align-self:start;position:sticky;top:15px}aside h2{font-size:20px;line-height:1.4}aside p{line-height:1.85;color:#56626d}#zoom{border-top:1px solid #eee;margin-top:20px;padding-top:14px}#zoom svg{width:100%;height:auto}.note{font-size:12px;color:#8a672f}.legend{line-height:1.8;color:#6a7580}input[type=range]{width:160px}@media(max-width:1050px){.layout{grid-template-columns:1fr}aside{position:static}}@media print{body{background:white}.toolbar,aside{display:none}.layout{display:block;padding:0}.canvas{border:0;border-radius:0}@page{size:420mm 294mm;margin:0}}</style>
<div class="toolbar"><strong>Native64 architecture · 단계 클릭 / 이미지 확대</strong><button id="play">▶ 설명용 시간 재생</button><label>t <input id="time" type="range" min="0" max="3" value="3" step="1"> <output id="time-label">1024</output></label><label><input id="feedback" type="checkbox" checked> OFF 경로 표시</label><button id="save">SVG 저장</button><button onclick="window.print()">PDF / 인쇄</button></div>
<main class="layout"><div class="canvas">SVG_HERE</div><aside><div class="note">Checkpoint replay가 아닌 architecture illustration</div><h2 id="detail-title">계산 경로를 눌러보세요</h2><p id="detail-body">기존 도식의 구조를 유지하면서 이미지 중심으로 표현했습니다. 각 단계를 클릭하면 설명과 확대 그림을 볼 수 있습니다. 시간 슬라이더는 설명용 phase map만 바꿉니다.</p><div id="zoom"></div><p class="legend">파란 block: 최근 학습 범위<br>빨간 점선: 비활성 feedback<br>이미지 안 숫자/색: object ID와 phase component는 서로 다른 개념입니다.</p></aside></main>
<script>const details=DETAILS_HERE,frames=FRAMES_HERE;const svg=document.querySelector('.canvas svg');let timer=null;const times=[1,129,513,1024];
function select(g){document.querySelectorAll('.stage.selected').forEach(x=>x.classList.remove('selected'));g.classList.add('selected');const d=details[g.dataset.stage];document.getElementById('detail-title').textContent=d[0];document.getElementById('detail-body').textContent=d[1];const b=g.getBBox();const z=document.createElementNS('http://www.w3.org/2000/svg','svg');z.setAttribute('viewBox',`${b.x-8} ${b.y-8} ${b.width+16} ${b.height+16}`);z.appendChild(g.cloneNode(true));document.getElementById('zoom').replaceChildren(z);}
svg.querySelectorAll('.stage').forEach(g=>{g.addEventListener('click',()=>select(g));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(g)}})});
function update(){const i=+document.getElementById('time').value;document.getElementById('time-label').textContent=times[i];for(let c=0;c<4;c++)document.getElementById(`phase-2-${c}`).setAttribute('href',frames[i][c]);const labels=svg.querySelectorAll('text');for(const t of labels)if(t.getAttribute('x')==='1176'&&t.getAttribute('y')==='497')t.textContent='t = '+times[i];const g=svg.querySelector('.stage.selected');if(g)select(g)}
document.getElementById('time').addEventListener('input',update);document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ 설명용 시간 재생'}else{timer=setInterval(()=>{const r=document.getElementById('time');r.value=(+r.value+1)%4;update()},850);document.getElementById('play').textContent='❚❚ 정지'}};
document.getElementById('feedback').onchange=e=>document.getElementById('disabled-paths').style.display=e.target.checked?'':'none';document.getElementById('save').onclick=()=>{const s=svg.cloneNode(true);s.querySelectorAll('.selected').forEach(x=>x.classList.remove('selected'));const a=document.createElement('a');const u=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(s)],{type:'image/svg+xml'}));a.href=u;a.download='native64_visual_architecture.svg';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script></html>'''.replace('SVG_HERE',svg).replace('DETAILS_HERE',json.dumps(details,ensure_ascii=False)).replace('FRAMES_HERE',json.dumps(frames))
(OUT/'native64_interactive.html').write_text(page,encoding='utf-8')
(OUT/'provenance.json').write_text(json.dumps({'reference':'SW0227/SW0223 lineage','actual_data':'Only RGB input cropped from '+str(source.relative_to(ROOT)),'internal_maps':'Illustrative RGB transforms / analytic phase patterns / synthetic spike raster and synchrony matrix; NOT measured model activations','output_masks':'Manually drawn illustrative regions; NOT ground truth loaded from data, NOT predictions','training_and_feedback_scope':'Preserved from architecture_figures/draw_native64.py','paper_layout_reference':'https://arxiv.org/pdf/2006.15055 Figure 1','contains_checkpoint_or_GT':'No'},ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT/'native64_interactive.html')
