"""Do the multi_object_datasets tfrecords parse with our own reader?

Our clevr_with_masks reader is pure python (gzip + struct), no TensorFlow. The
other datasets in that suite are written by the same code, so the feature names
and dtypes are what decides whether Tier 1 costs an afternoon or a week.
"""
import sys, struct, gzip
sys.path.insert(0, "/export_home/tkim1/snn_kuramoto_bidirectional")
from training.prepare_clevr_with_masks import iter_records, _parse_example

for name, path in (
    ("tetrominoes", "/work/USERS/tkim1/mod/tetrominoes_train.tfrecords"),
    ("multi_dsprites_coc", "/work/USERS/tkim1/mod/multi_dsprites_colored_on_colored.tfrecords"),
    ("clevr (ours)", "/work/USERS/tkim1/clevr/with_masks/clevr_with_masks_train.tfrecords"),
):
    print("=== %s ===" % name)
    try:
        rec = next(iter(iter_records(path, limit=1)))
        f = _parse_example(rec)
        for k, v in sorted(f.items()):
            try:
                print("   %-12s shape %-18s dtype %s" % (k, str(getattr(v, "shape", "-")), getattr(v, "dtype", type(v))))
            except Exception as e:
                print("   %-12s %r" % (k, e))
    except Exception as e:
        print("   FAILED: %s: %s" % (type(e).__name__, e))
    print()
