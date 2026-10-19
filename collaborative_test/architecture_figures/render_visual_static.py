"""Render this generator's limited SVG vocabulary without a browser subprocess."""
from pathlib import Path
from io import BytesIO
import base64
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,PathPatch
from matplotlib.path import Path as MPath
from fontTools.svgLib.path import parse_path
from fontTools.pens.recordingPen import RecordingPen

OUT=Path(__file__).resolve().parents[2]/'collaborative_testing/generated/architecture_figures/native64_interactive'
root=ET.parse(OUT/'native64_visual_architecture.svg').getroot()
fig=plt.figure(figsize=(16,11.2));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1600),ylim=(1120,0));ax.axis('off')
plt.rcParams['pdf.fonttype']=42
def render(e):
    tag=e.tag.rsplit('}',1)[-1];a=e.attrib
    if tag=='defs':return
    if tag=='rect':
        ax.add_patch(FancyBboxPatch((float(a['x']),float(a['y'])),float(a['width']),float(a['height']),
          boxstyle=f"round,pad=0,rounding_size={a.get('rx',0)}",facecolor=a.get('fill','white'),edgecolor=a.get('stroke','none'),lw=.72))
    elif tag=='image':
        im=Image.open(BytesIO(base64.b64decode(a['href'].split(',',1)[1])))
        x,y,w,h=[float(a[k]) for k in ('x','y','width','height')]
        ax.imshow(im,extent=(x,x+w,y+h,y),interpolation='nearest',aspect='auto',zorder=2)
    elif tag=='text':
        family={'Arial':'DejaVu Sans','Georgia':'DejaVu Serif','monospace':'DejaVu Sans Mono'}.get(a.get('font-family'),'DejaVu Sans')
        ax.text(float(a['x']),float(a['y']),e.text or '',fontsize=float(a.get('font-size',16))*.72,
                color=a.get('fill','#20272e'),ha={'start':'left','middle':'center','end':'right'}.get(a.get('text-anchor'),'left'),
                va='baseline',fontfamily=family,weight=a.get('font-weight','normal'),zorder=3)
    elif tag=='path':
        pen=RecordingPen();parse_path(a['d'],pen)
        vertices=[];codes=[]
        for op,points in pen.value:
            if op=='moveTo':vertices.append(points[0]);codes.append(MPath.MOVETO)
            elif op=='lineTo':vertices.append(points[0]);codes.append(MPath.LINETO)
            elif op=='curveTo':vertices.extend(points);codes.extend([MPath.CURVE4]*len(points))
            elif op=='closePath':vertices.append(vertices[0]);codes.append(MPath.CLOSEPOLY)
        if vertices:
            ax.add_patch(PathPatch(MPath(vertices,codes),facecolor=a.get('fill','none'),edgecolor=a.get('stroke','none'),
                     lw=float(a.get('stroke-width',1))*.72,alpha=float(a.get('stroke-opacity',1)),
                     linestyle='--' if 'stroke-dasharray' in a else '-',zorder=1))
            if 'marker-end' in a and len(vertices)>1:
                end=np.asarray(vertices[-1]);start=np.asarray(vertices[-2]);delta=end-start
                if np.linalg.norm(delta):
                    start=end-delta/np.linalg.norm(delta)*10
                    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',color=a.get('stroke','#3e4851'),lw=.8),zorder=2)
    for child in e:render(child)
render(root)
for extension in ('png','pdf'):fig.savefig(OUT/('native64_visual_architecture.'+extension),dpi=180,facecolor='white')
plt.close(fig)
print('PNG/PDF rendered from SVG. This does not assert browser interaction verification.')
