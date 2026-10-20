"""Diagram of the possible project paths from the Bellman equation, for 0_project.md.

Run from anywhere: python tools/project_paths.py  (writes _static/img/project_paths.svg)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import pathlib
OUT = pathlib.Path(__file__).resolve().parent.parent / '_static' / 'img' / 'project_paths'

colors = {'theory': ('#dbe9f6', '#4a78a8'), 'synthetic': ('#e2f1e4', '#4b8a55'),
          'real': ('#fbe8d3', '#b8742d'), 'top': ('#eeeeee', '#555555'), 'shared': ('#f3e6f5', '#8a5a99')}
W, H = 960, 458                                          # canvas in points, y downwards
FS, LH = 11, 20                                           # font size and line height

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

def box(cx, y, w, lines, kind, bold=False, pad=18):
    '''A rounded box centered at cx, top at y, with the given lines of text; returns (top, bottom)'''
    h = LH * len(lines) + pad
    fill, edge = colors[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h, boxstyle='round,pad=0,rounding_size=10',
                                fc=fill, ec=edge, lw=1.5))
    for i, line in enumerate(lines):
        ax.text(cx, y + h / 2 + (i - (len(lines) - 1) / 2) * LH, line, ha='center', va='center',
                fontsize=FS, weight='bold' if bold else 'normal', color='#222222')
    return y, y + h

def arrow(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color='#666666', lw=1.4, shrinkA=0, shrinkB=0, mutation_scale=14))

L, M, R, bw = 165, 480, 795, 280                          # column centers and box width
_, top_b = box(M, 12, 340, ['The model: Bellman equations'], 'top', bold=True)

hdr = {}                                                  # branch headers
for cx, kind, name in ((L, 'theory', 'Theoretical paper'), (M, 'synthetic', 'Synthetic data'), (R, 'real', 'Real data')):
    t, b = box(cx, 92, 210, [name], kind, bold=True, pad=12)
    arrow(M + (cx - M) * 0.25, top_b, cx, t)
    hdr[cx] = b

def column(cx, kind, items):
    '''A chain of boxes under the header of the column; returns the bottom of the last box'''
    y, prev = hdr[cx] + 28, hdr[cx]
    for lines in items:
        t, b = box(cx, y, bw, lines, kind)
        arrow(cx, prev, cx, t)
        y, prev = b + 28, b
    return prev

column(L, 'theory', [['Solution algorithms', 'Class of problems', 'Econometric methods', 'Characterization of the solution'],
                     ['Monte Carlo study of the', 'performance of the', 'proposed methods']])
mid_end = column(M, 'synthetic', [['Simulate data from the model'], ['Calibration to', 'aggregate moments'],
                                  ['Monte Carlo study of the', 'estimation procedure and code']])
right_end = column(R, 'real', [['Data description'], ['Estimation of', 'structural parameters'], ['Goodness of fit']])

C = (M + R) / 2                                           # shared: counterfactuals
t, b = box(C, mid_end + 45, 240, ['Counterfactuals'], 'shared', bold=True)
arrow(M, mid_end, C - 60, t)
arrow(R, right_end, C + 60, t)

fig.savefig(str(OUT) + '.svg', transparent=True)
print('saved', str(OUT) + '.svg')
