import json
c = json.load(open('/work/USERS/tkim1/sw0255b/curve252.json'))
rows = {k: v for k, v in c.items() if isinstance(v, dict)}
other = {k: v for k, v in c.items() if not isinstance(v, dict)}
print("252 evaluation curve (CLEVR):")
print(f"  {'ckpt':8s} {'updates':>8s} {'images':>8s} {'fg_ari':>9s} {'fg_iou':>9s} {'obj_iou':>9s}")
for k in sorted(rows, key=lambda k: rows[k].get('updates', 0)):
    r = rows[k]
    print(f"  {k:8s} {r.get('updates',0):8d} {r.get('images_seen',0):8d} "
          f"{r['fg_ari']:9.4f} {r['foreground_iou']:9.4f} {r['matched_object_iou']:9.4f}")
ious = sorted({round(r['foreground_iou'], 8) for r in rows.values()})
print(f"\n  distinct foreground_iou across all {len(rows)} checkpoints: {ious}")
if other: print("  other keys:", other)
