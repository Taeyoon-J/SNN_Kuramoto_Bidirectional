# Current status

SW0097 complete: matched graph adaptation loses all3 mean metrics and FG-ARI
in every seed versus graph-frozen continuation. Frozen mean
.776348/.577070/.601537; graph-adaptive mean .769307/.575107/.596571.
Do not expand this exact graph-learning objective to a full70k pass.
Next investigate temporal-window mismatch (training64/32 vs evaluation1024/512)
on original validation only; no new independent-test selection.

Latest2026-10-07: SW0096 finished frozen holdout comparison. Independent
IDs90000??0319: ours **.759817/.562696/.588620**, Slot
**.777602/.196114/.201021**. Reference1000??319: ours
.763595/.561184/.584245, Slot .773277/.194499/.200663.
FG-ARI exceedance did not generalize; final goal is not achieved.
Keep the substantial IoU gains and improve object separation on validation
only. Both holdouts are now observed; reserve90640??0959 unread for the next
final frozen confirmation. All12 evaluations and per-seed records archived.

2026-10-07 update: SW0094 four-arm pilot and full320 joint evaluation completed.
Joint adaptation trades higher foreground IoU for lower FG-ARI/object IoU;
the encoder/graph-frozen aligned spike-loss recipe was retained for SW0095.
SW0095 completed an additional whole70k pass for every source seed0/1/2.
Full320 validation mean is **.775645/.574029/.599779**, numerically above the
matched SW0092 Slot epoch10 mean **.774933/.203589/.206937** on all3 metrics.
FG-ARI margin is only .000711; no reliable statistical superiority claim.
Training budgets differ (140k downstream exposures plus pretraining vs700k
scratch Slot exposures). Archived raw results and audits are in SW0095.
SW0096 registers frozen-checkpoint comparisons on new IDs90000??0319 and
previously inspected reference IDs1000??319; independent evidence remains
pending. Goal remains active. If tkim1 occupies GPU2 or3, our new jobs use0/1
only; SW0096 is restricted to0/1. Older process/status paragraphs below are
historical, superseded by this update and the numbered experiment records.

As of 2026-10-06, the user requested renewed analysis of degradation with
70,000-image training and improvement toward beating Slot Attention. The
[SW0093 analysis](SW_0093_scaling_failure_analysis/README.md) verifies a mismatch
between the absolute spike correlation used by training and positive-only
correlation used by prediction, and documents loss/FG-ARI divergence and
oversegmentation. Its causal contribution still needs controlled testing.
The user has restarted goal mode, explicitly targeting full70k training.
[SW0094](SW_0094_aligned_joint_pilot/README.md) registers four short controlled
continuations: old spike loss, classifier-aligned loss, graph adaptation, and
joint encoder/graph/core adaptation. Server synthetic CPU checks and all4
real-asset CPU backward/update checks passed. The server queue is live as
PID3557827 after live-child adoption from the initial queue, without restarting
training. The user explicitly authorized free GPU2; allowed GPUs are now0/1/2.
The two frozen-graph arms completed training and pilot evaluation. At the latest
check positive_joint remains live onGPU0 (PID3558132); positive_graph completed.
The aligned frozen pilot scores .847604/.729511/.676380 on80 images, above
the old-loss pilot .830922/.722971/.669574 on all3 metrics. Single-source pilot
only. Full320 validation queue PID3560324 waits for GPU0; GPU2 is now occupied
again by the colleague. SW0095 queue PID3562158 waits for pilot/full validation
before three complete70k continuations. New source-seed CPU checks passed.
Update2026-10-07: the full320 evaluator actually runs as PID3567321, GPU0,
batch1, sharing only our own workers with ample free VRAM. The joint worker
is CPU-I/O-bound and remains live. SW0095 manager was replaced before any
training launch; PID3568428 waits only for full validation and can use our
own GPU's8GiB-or-more free VRAM without waiting for the independent encoder arm.
Full320 batch1 result completed (.822781/.719916/.662139), but overlapping
images reveal batch sensitivity; original batch8 revalidation is live.
SW0095 seed0 actual full70k continuation runs as PID3569312 onGPU0, source
seed1/2 pending. This is not three-seed success evidence yet.
Original batch8 full320 confirmation completed: .822115/.718331/.661153,
above unchanged source0 by .003644/.003339/.000818. First80 per-image scores
match the batch8 pilot exactly. Batch1 is diagnostic only. SW0095's actual
70k pass remains running and will use batch8 for all three model seeds.
An independent CPU diagnosis completed on4 fixed validation images and the
unchanged seed0/2 epoch1/10 cores: seed2 long-window same-object edge recall falls
from .9109 to .4802 despite high pair AUC. Sign disagreement is larger in the
training window than long evaluation. See SW0094 for limitations and raw data;
queued GPU arms remain unchanged.

Experiments1/2/3 requested by the user are complete: all three training seeds
and epoch1/3/10 evaluations per experiment, plus experiment1's three-image mask
visualizations. See [final comparison](TEST123_FINAL_RESULTS.md). The required
next action after sharing those results was to pause goal mode, which was done.
The new analysis request supersedes the earlier restriction on research scope.

## Earlier research history

- User clarified classifier/count and stage-wise activation analysis are useful research direction, not strict sequential gates. Run classifier and core diagnostics in parallel when either can clarify the other.
- The earlier 90-minute cap applied to the prior goal run and expired before the user explicitly restarted goal mode. No new runtime cap was set for the restarted goal.
- Current aligned SW0042 three-seed result is complete. At the common long
  threshold .35, mean FG-ARI / foreground IoU / matched-object IoU is
  `.415879 / .418226 / .296118` (sample std `.210397 / .249434 / .227312`).
  Both IoUs exceed the aligned official single-checkpoint Slot reference, but
  FG-ARI is far below it and variance is unacceptable. Seed2 collapses to
  about `.1857/.1352/.0370`; its phase product-PLV mean/std across images is
  `.4454/.00627`, versus seed0 `.7150/.1433` and seed1 `.6487/.1707`.
- SW0047 completed the correct-position control. Randomly permuting the
  Gaussian kernel's patch correspondence drops the best seed0 short row from
  correctly aligned about `.5993/.6085/.4637` to
  `.3192/.4945/.2543`; long permuted best is
  `.3249/.4802/.2630`. Spatial correspondence is a real readout prior, but it
  is not a learned-core improvement.
- SW0050 is the active model-training intervention. Seed2/no-diversity at LR
  `3e-4` completed all 40 epochs. Long threshold .35 scores
  `.347156/.167773/.135250`, versus original seed2
  `.185651/.135165/.036969`; substituting it raises the diagnostic three-seed
  mean to `.469714/.429096/.328878`. This rescues all three seed2 metrics but
  remains below seeds0/1. The matched LR `1e-3` arm with measured
  approximately `0.1x`-gradient diversity also completed at
  `.305639/.226015/.123898`; it trades lower ARI/object IoU for much higher
  foreground IoU than low LR. Its seed2-substitution mean is
  `.455875/.448510/.325094`. Future launches require asset-bound one-update
  preflight; remote Bash, Python compile, 3+5 direct unit tests, and both real
  asset preflights pass.
- SW0051 seed0 completed both windows. Long membrane-freeze scores
  `.630592/.606040/.449873` versus frozen spike-CC
  `.598427/.606040/.461779`; membrane-restricted-dynamic scores
  `.588565/.606040/.465839`. The foreground mask is exactly unchanged. This
  separates an ARI gain from a small object-IoU gain but provides no single
  dominant hybrid yet.
- SW0052 evaluated both SW0050 trajectories at saved epochs20/25/30/35 after
  per-checkpoint real-asset smoke tests. Low-LR epoch25 is the clear model
  selection: long common .35 `.395707/.189112/.159085`, strictly above its
  epoch40 `.347156/.167773/.135250`. Diversity epoch30
  `.330248/.222187/.157874` and epoch35 `.302799/.231287/.132006` retain a
  foreground-IoU tradeoff. Substituting low-LR epoch25 for collapsed seed2
  gives diagnostic three-seed mean about `.485898/.436209/.336823`; this is
  not yet a same-recipe three-seed result.
- SW0053 completed the same-recipe low-LR/epoch25 expansion. Long common `.35`
  seeds0/1/2 are `.700964/.613168/.484189`,
  `.583643/.343275/.404509`, and `.395707/.189112/.159085`; the formal mean
  is `.560105/.381852/.349261`. At `.50` it is
  `.562902/.381328/.356857`, the new formal three-seed lead. The matched
  seed2 LR1e-3/epoch25 control is only `.345872/.172243/.126184`, proving low
  LR contributes beyond early stopping. Relative to SW0042, mean ARI and
  object IoU improve strongly; foreground IoU trades down but stays above the
  audited Slot single-checkpoint reference. Goal remains unmet because FG-ARI
  is still below Slot `.894641`.
- SW0055 prepared the unique-data scaling contract. Training IDs are 0-999
  plus 1640-3139, explicitly excluding held-out/reference IDs1000-1639. The
  frozen encoder and saved scalar normalization reproduce the original 1,000
  gamma rows exactly (max difference 0.0); the expanded gamma is finite with
  shape `[2500,8,256]` and SHA-256 `71394392...c742aa`. The matched-exposure
  2500x10 training arm now has a staged, GPU0/1-only scheduler: SW0054 pilot,
  seeds0/1 concurrently after SW0055 preflight, then seed2 on GPU0 only after
  both first seeds finish and validate. It records wrapper/worker PIDs and
  phase timestamps, verifies evaluation markers, and refuses partial outputs
  or stale progress rather than restarting. No server run was launched.
- SW_0034 full320 complete: distance-controlled AUC phase .5788, gating .5871, h-wave .5842, membrane .5870, gated spike .5863, binary threshold .5000. Binary threshold is constant 1 after settle across all aggregate patch histories; gated spike variation is gate-driven. Continuous signal is weak upstream and roughly preserved downstream; no single sharp loss stage. Spatial-only/membrane-spatial/gated-spike-spatial readouts .491272/.190837/.178026, .494639/.191070/.170568, .479762/.185651/.164179. Phase PLV bypasses downstream gradients; membrane/spike losses connect them. No improvement claim.
- SW_0035: adaptive object-count classifier baseline code prepared. It compares GT-free eigengap/eigen-threshold count inference, existing adaptive slots, and fixed-k grouping controls on membrane and gated-spike patterns. Not yet run.
- SW_0035 pilot: gated-spike spatial eigengap range 5-10 reduces predicted groups 9.0??.06 and count MAE 2.95??.05; FG IoU .1746??1997, but FG-ARI .4871??4759 and object IoU .1605??1413. Exact count only 4.7%. Existing membrane slots average 6.06 versus true 6.11 but exact count 7.8%, showing mean count is misleading. Selected full320 baseline running.
- SW_0035 full320: gated-spike spatial eigengap 5-10 scores .490649/.204761/.140716, count MAE 1.925 and 7.066 groups. It improves fixed spike k10 ARI/FG IoU/count MAE but lowers object IoU; exact count 12.8%, within-one 44.7%, and 301/320 images choose eight total clusters. Keep a three-track classifier reference suite rather than one strict winner: adaptive eigengap for count/grouping, nonspatial membrane k10 for FG IoU, membrane-spatial k10 for ARI/object IoU.
- SW_0036 full320: frozen inference vth2 repairs binary event saturation (rate .299, constant .00070) and improves gated-spike spatial k10 on all three metrics .479762/.185651/.164179??487351/.192635/.172932. Membrane ARI/object IoU decline and distance-controlled spike AUC remains weak .5553. Treat as dynamics calibration, not trained improvement.
- SW_0037 full320: matched seed-0 40-epoch membrane-PLV training at vth2 repairs event saturation (rate .547, constant .00323), but distance-controlled AUC falls to membrane .5113/spike .5143/binary .5179. Fixed membrane spatial k10 .478921/.200110/.174031 and gated-spike spatial k10 .461878/.204486/.174888 trade lower FG-ARI for higher IoUs versus SW_0003. Adaptive rows do not rescue all three metrics. Do not expand to seeds 1/2.
- Peer `patch_v2:c76492d` review: new PV2_0008 uses a different render/split/foreground prevalence and geodesic graph, so scores are not pooled. Transfer only its additive component-spike PLV mechanism and warning that lower training loss did not predict better task metrics. Full affinity MSE distillation failed on peer due dense fusion. See `peer_updates/REVIEW_0002_20261004.md`.
- SW_0038 full320 stage result: matched phase-only control spike .407526/.228462/.194464; balanced spike weight.25 .416059/.228732/.190776; transferred weight5 .435727/.226240/.198260. Stronger spike loss restores event activity and partly recovers ARI/object IoU over the vth2 phase control, but remains below the .494639 ARI lead and does not dominate prior Pareto candidates. Do not expand this exact configuration to seeds 1/2. Selected classifier JSONs are being finalized before the runtime pause.
- 2026-10-04 new-chat resume: created a new active goal; source branch `patch_v2_sw`, HEAD before preparation `28d24b0`. Handoff and start prompt were present as untracked files. Peer fetch succeeded and remains `b99fac0`, with no new tested results. Server SSH master socket is absent; live processes/GPU/checkpoint cannot yet be verified. User will restore SSH after chat preparation.
- SW_0034: planned stage-wise object-signal diagnostic prepared under `SW_0034_stage_signal_diagnostic`. Observe unchanged core with hooks; compare phase, gating, h-wave, membrane, gated spikes and ungated binary threshold; distance-stratified foreground-pair AUC and phase/membrane/spike loss gradient connectivity. Same spatial-only and membrane-spatial k10 controls. Syntax check passed; real-data execution pending server connection. No new training or improvement claim.
- SW_0034 preparation follow-up: add per-stage constant-node fraction/std/activation statistics; synthetic CPU smoke check in local kuramoto environment passed exact observation-hook output parity, binary-times-gate spike reconstruction and finite membrane-loss downstream gradients. No full diagnostic/checkpoint validation yet; h5py missing locally and SSH master still absent.

- Goal: active. The three-seed mean of a spike or membrane readout must exceed the comparable Slot Attention mean on all three patch metrics.
- Slot baseline audit: the saved .890115/.212251/.235487 reference is one deterministic inference run from Google's official pretrained `ckpt-500`, trained on a different CLEVR distribution with seven slots and evaluated here with 11 slots. It is not a comparable three-seed training mean. Keep it as the provisional numeric bar; a formal goal claim still needs a matched three-seed Slot baseline or an explicit limitation. See `baselines/slot_attention.md`.
- Baseline seed 0 checkpoint: `/Data0/kevinswk/patch_v2_sw/trained_models/baseline_best_seed0_20261003/core.pt`.
- SW_0001: completed classifier-only validation experiment. Fixed-k spike synchrony clustering improves all three metrics over the baseline spatial-components readout, but does not meet the Slot Attention reference.
- Gradient connectivity confirmed: phase-only loss gives no gradients to dendritic or membrane parameters (0/3 and 0/1 tensors connected). See `baseline_gradient_connectivity.json` on server.
- SW_0002: finished 10-epoch membrane-PLV pilot. Spike-synchrony FG-ARI rose to 0.270323, but foreground IoU and object IoU fell to 0.235452 and 0.123274 versus the 40-epoch starting checkpoint. No all-metric gain; training duration also differs. See SW_0002 report.
- SW_0003: matched 40-epoch membrane-PLV validation complete. FG-ARI improved to 0.2889, but foreground IoU and object IoU fell to 0.2075 and 0.1027. No all-metric gain.
- Graph diagnostic on validation IDs 1320??351: mean same-object graph weight 0.254288 versus different-object 0.021720. SW_0004 graph-to-membrane loss pilot finished: spike FG-ARI 0.202940, foreground IoU 0.245920, matched object IoU 0.142077. Dendritic 3/3 and membrane 1/1 parameters have gradients, but there is no all-metric gain. Ten epochs versus 40 in baseline limits causal comparison.
- SW_0005: background-border rule screen completed. Best foreground IoU 0.3220 came with lower FG-ARI 0.0495 and object IoU 0.1386; not adopted.
- SW_0006: peer `patch_v2:5a29422` connected-components spike readout was re-evaluated on our fixed validation split. At correlation threshold 0.95, aggregate spike readout reached FG-ARI 0.077165, foreground IoU 0.328720, matched object IoU 0.173560. Both IoUs beat SW_0001, but FG-ARI fell from 0.136286. Prediction splits into about 14.88 foreground groups per image. Full threshold sweep in SW_0006/results.json; no production classifier merge yet.
- SW_0007: hybrid readout kept SW_0006 foreground but reclustered only those patches into 3/5/7 spike-synchrony groups. Best FG-ARI was 0.080966 (threshold 0.95, k=5); foreground IoU 0.328720, matched object IoU 0.134053. This did not recover SW_0001 FG-ARI and degraded object IoU; not adopted.
- SW_0008: pairing peer `patch_v2:5a29422` component-product connected-component readout (threshold 0.50) with SW_0004 graph-teacher core gave validation FG-ARI 0.195269, foreground IoU 0.275711, matched object IoU 0.238803. All exceed SW_0001 seed-0 validation, but there are about 31 predicted foreground groups/image. This is a seed-0, 10-epoch validation candidate, not a goal result. Pairing the same readout with SW_0003 was poor. Freeze the selected readout for reference-test and three-seed follow-up; no test selection.
- SW_0009: the frozen SW_0008 readout on reference IDs 1000??319 scored FG-ARI 0.190909, foreground IoU 0.270928, matched object IoU 0.229805 (seed 0). Slot single-checkpoint reference on the same IDs is 0.890115 / 0.212251 / 0.235487. Only foreground IoU exceeds the reference. About 30.14 foreground groups/image indicate severe fragmentation. Goal not met.
- SW_0009 count diagnostic: on the same 320 reference masks, true foreground object count averages 6.04 (range 2??0), while predictions average 30.14 (range 9??4). Exact count matches: 0/320; within one object: 0/320. This readout has no explicit count inference; its connected components overfragment objects. See object_counts.json.
- SW_0010: validation-only lower threshold sweep for the peer component-product readout reduced the average predicted foreground groups from 30.99 at 0.50 to 18.77 at 0.10, but lowered all three metrics from 0.195269 / 0.275711 / 0.238803 to 0.183586 / 0.266737 / 0.197857. Other lower thresholds were also worse. Do not adopt a lower threshold merely to reduce count.
- SW_0011: matched 40-epoch graph-teacher seed-0 training and fixed-split validation completed. Relative to SW_0008's 10-epoch matched readout, component-product threshold 0.50 changed FG-ARI 0.195269??.179760, foreground IoU 0.275711??.271468, and matched-object IoU 0.238803??.295250. Predicted groups grew from ~31 to 77.03/image against true 6.20/image, so this is not an all-metric gain. The first-16-image theta?�spike affinity correlation was 0.7265 (10-epoch: 0.8160), an association rather than proof of causation.
- SW_0012: examined peer `patch_v2:0dd2115` geodesic graph on 16 of our validation images, with our existing SW_0004 graph weights and no code merge. Same-object/different-object edge-weight ratio fell from 15.00 to 13.59, and 6.27% of computed distances were negative. This read-only pilot does not justify changing the core yet. Peer PV2_0001 3-seed results use a different render/split and are not compared numerically to ours.
- SW_0013: adaptive spike-pattern slots were evaluated on SW_0011's checkpoint with the fixed 320-image validation contract. The best-ARI setting (cosine threshold .7, six initial slots) scored FG-ARI 0.247376, foreground IoU 0.248332, object IoU 0.204032, versus the existing SW_0011 readout 0.179760 / 0.271468 / 0.295250. Predicted count fell from 77.03 to 17.37 but remained above true 6.20; exact count 0/320. This is a mixed result; no classifier replacement. `patch_sw` edge membrane separation loss remains a separate future ablation because our gamma-only training batch has no aligned RGB images.
- SW_0014: tested border-based versus largest-slot background on SW_0013's adaptive slots, same checkpoint/split. The strongest border rule increased foreground IoU .248332??338240 but reduced FG-ARI .247376??071979 and object IoU .204032??177311. Foreground recall fell .8259??4796. No background-rule replacement.
- SW_0015: checked gamma/HDF5 ordering. The CLEVR_1k PNG directory did not match HDF5 indices, but direct `wm` gamma-versus-HDF5 boundary fingerprints did: same-index rank top-1 for 27/128 train and 26/128 validation images (random expectation 1/128); median ranks 5 and 4 vs random 64.5. This strongly supports, but does not prove, actual gamma/HDF5 index alignment. RGB-edge training ablation is now technically plausible; use a paired gamma+RGB dataloader, not the old gamma-only batches.
- SW_0016: optional RGB-edge membrane loss completed on a fresh matched seed-0, 40-epoch run; core architecture unchanged. At the same component-product spike threshold .50, FG-ARI .069476 / foreground IoU .331876 / object IoU .164980, compared with SW_0011 .179760 / .271468 / .295250. Group count fell 77.03 to 11.98, but object separation deteriorated. First-16 phase-to-spike affinity correlation .7287; spike same/different-object synchrony contrast .1001 versus SW_0011 approximately .1540. No all-metric gain; do not adopt the weight-.1 edge loss. See SW_0016 report and result JSONs.
- SW_0017: changing only SW_0013's adaptive-slot input to continuous membrane raised FG-ARI .247376??261507 and object IoU .204032??207637 at threshold .7/six slots, but FG IoU fell .248332??246264; 18.18 predicted groups versus 6.20 true. Still no all-metric improvement over existing readout. No production change.
- Peer review at `origin/patch_v2:fa658ee`: rate-target loss reduced rate variance but worsened all three mean metrics on the peer's dataset; geodesic graph improved peer ARI/object IoU but lowered foreground IoU. Their render/split/reference differ, so scores are not pooled with ours. See `peer_updates/REVIEW_0001_20261003.md`.
- SW_0018: peer-inspired low-spike-synchrony foreground selection paired with our adaptive slots was tested on our fixed 320-image validation. The best FG-IoU setting reached .291989, but FG-ARI .130456 and object IoU .177966, below existing SW_0011 readout. No all-metric gain or merge.
- SW_0019: classifier-only pilot mixed actual per-component spike and membrane correlations from the SW_0011 checkpoint. On the first 64 fixed validation images, 25% membrane at threshold .50 improved FG-ARI .157762 to .188753 and foreground IoU .257242 to .268401 versus pure spike, but reduced matched-object IoU .296283 to .259725. No production classifier change; all 35 sweep rows recorded under SW_0019.
- SW_0020: matched fresh seed-0 10-epoch spike-source graph-teacher training and fixed 320-image validation completed. At component-product threshold .50, FG-ARI .122560 / FG IoU .260356 / object IoU .287220, versus membrane-source SW_0004 .195269 / .275711 / .238803. The object-IoU improvement is retained as a secondary insight, while the priority FG-ARI decline and 144.21 predicted groups/image argue against using the exact weight-.1 variant as the next lead. Actual spike rate fell .419544 to .195464; first-16 phase-to-spike affinity correlation .816019 to .805587. No core architecture change, no goal claim.
- SW_0021: frozen high-FG-ARI SW_0003 checkpoint plus existing adaptive slots was evaluated on all 320 fixed validation images. Actual spike slots scored .313070/.202942/.107462; membrane slots scored .357053/.209870/.113992 (FG-ARI/FG IoU/object IoU). The prior fixed-k readout on the same core was .288924/.207549/.102696. Mean predicted membrane groups 5.98 versus 6.20 true, but exact count only 11.9%. Retain this as the ARI-first track; investigate background selection and IoU, rather than rejecting it for mixed metrics.
- SW_0022: on the same SW_0003 checkpoint and 320-image validation, membrane-border-prototype background selection with largest-slot AND similarity >=.90 yielded .361770/.215081/.119675, a small all-metric increase over SW_0021 .357053/.209870/.113992. Predicted foreground still covers .677 of patches versus .217 GT, so background rejection is a major IoU limitation (not necessarily the FG-ARI limitation). This is a seed-0 validation improvement candidate, not a three-seed goal result.
- SW_0023: on the same SW_0003 checkpoint and validation split, single spike/membrane feature AUCs for foreground were weak (best inverse border-template similarity .611). Explicitly nondeployable oracle-GT-foreground diagnostics raised FG IoU .210 to 1.000 and matched-object IoU .114 to .385, but left FG-ARI exactly .357. Thus background rejection is the IoU bottleneck, while grouping itself is the FG-ARI bottleneck. Prior statements about background as the main limitation must be read as IoU-specific. No model/prediction change or goal claim.
- SW_0024: same SW_0003 seed-0 checkpoint, actual membrane histories, and fixed validation contract. Absolute-correlation spectral k=10 was selected on first 64 validation images and confirmed on all 320: FG-ARI .393974 / FG IoU .228821 / object IoU .150706, all above the adaptive-slot control .357053 / .209870 / .113992. It yields nine predicted groups/image against true mean 6.20, so fragmentation remains. This is now the leading ARI-first classifier candidate, not a three-seed/reference-test goal result.
- SW_0025: a larger spectral-k sweep on the same checkpoint/split showed the 64-image k=12 ARI uptick did not persist over all 320 validation images; k=10 remains top FG-ARI .393974. Increasing to k=20 raised matched-object IoU .150706 to .228019 but lowered FG-ARI to .348207 and produced 19 groups/image. Retain k=20 as an object-IoU tradeoff candidate rather than reject the gain; no three-seed or test claim.
- SW_0026: peer-inspired spike-product threshold sweep .0005-.50 on three frozen seed-0 cores and all 320 validation images. SW_0004's .50 control remains best of its sweep on all three metrics. SW_0011 at .10 lifts foreground IoU .271468 to .331330 but lowers FG-ARI .179760 to .098892 and object IoU .295250 to .180793; retain it as a foreground-IoU track, not an ARI lead. This different-data peer setting does not transfer unchanged. No production merge, reference-test selection, or three-seed claim.
- SW_0027: on the same frozen SW_0003 core and 320 validation IDs, comparing membrane spectral affinity constructions at fixed k=10 preserved the aggregate absolute FG-ARI leader .393974/.228821/.150706. Per-component positive-mean affinity produced .386094/.232912/.157926, a small two-IoU gain at a small FG-ARI cost. Both tracks remain candidates; no core/loss/training change, production merge, reference-test selection, or three-seed claim.
- Peer `patch_v2:b99fac0` review: the new commit changes only its status/resume notes, not a tested score. It prioritizes downstream readout and slow dendrite/membrane time constants; its detached queue and split are peer-specific and were not run or merged here.
- SW_0028: on SW_0003 seed-0 core, 320 fixed validation IDs, spatial Gaussian weighting of the existing membrane spectral affinity at sigma 1.5 patch units increases FG-ARI .393974 to .494639 and matched object IoU .150706 to .170568, while foreground IoU falls .228821 to .191070. This is the new FG-ARI-first lead, not an all-metric or three-seed/reference-test win. Core, losses, optimizer and training are unchanged; next isolate foreground/background handling without abandoning this grouping gain.
- SW_0029: sigma .75/1/1.25/1.5/2 crossed with k6/8/10 on first 64 fixed validation images. Existing sigma1.5/k10 remains FG-ARI leader .530110; reducing k lowers FG-ARI substantially. Sigma1.25/k10 gives only a small pilot object-IoU gain .176630 to .178952 at lower FG-ARI .520789. No candidate promoted to full320; keep SW_0028 lead and diagnose foreground assignment next. No model/training/evaluator change.
- SW_0030: first-64 validation pilot of adding entire border-touching spatial spectral groups to background. Predicted foreground fraction .865 versus true .199; border fraction threshold .2 brings predicted fraction to .228, but foreground IoU only .184 to .198 and FG-ARI collapses .530 to .261. This whole-group border rule is not adopted; matching foreground area is not enough. No full320 promotion or model change.
- SW_0031: full 320 validation on frozen SW_0003 core. Spatial membrane grouping .494639/.191070/.170568 remains FG-ARI/object-IoU lead versus independent nonspatial positive membrane foreground gating .432699/.205366/.161077. This all-patch gate helps foreground IoU slightly but loses substantial FG-ARI; do not replace the ARI lead. Predicted foreground fraction remains .659 versus true .217, suggesting selective veto rather than blanket intersection. No production merge or training change.
- SW_0032: border-limited independent membrane veto on same spatial grouping and fixed 320 validation IDs. Width2 scores FG-ARI .467718 / foreground IoU .210417 / object IoU .174335, versus spatial control .494639/.191070/.170568. Retain as a two-IoU tradeoff, not an ARI lead. First-64 pilot agrees directionally. Predicted foreground .748 versus true .217 means background remains the major IoU problem. No production merge or training change.
- SW_0033: critical 320-image diagnostic on the frozen SW_0003 core. Spatial-only kernel and k10 spectral readout scores .491272/.190837/.178026, nearly the actual membrane×spatial .494639/.191070/.170568; a random patch-position permutation of membrane affinity scores .481408/.192980/.165653. Raw membrane same-vs-different-object foreground neighbor-pair AUC .6365 versus spatial-only .6979 (distance-confounded diagnostic). The spatial prior explains nearly all of the ARI gain; do not describe SW_0028 as strong learned object binding. Shift attention to better membrane training signals while retaining the spatial readout control. GT used only for evaluation/diagnosis, never prediction.
- No training process has been started for SW_0001.
- Peer review 0003 (2026-10-04): direct review of peer commit `a026c1a` corrects the old spike-vs-phase claim: BIM6 spike mean .675574/.687838/.472764 remains below same-checkpoint phase .7120/.7150/.4994. Peer uncommitted server snapshot reports long T1024/sync.10 seed FG-ARI .6362/.7100/.7199 (mean .6887), negative per-region transfer, population inference can rescue dead seed 5 but harm normal seeds, and population training is poor. Sources, cautions, and variant details are in `peer_updates/REVIEW_0003_20261004.md`; snapshot read around 16:xx ET and may change.
- SW_0039: per-region projection experiment held; do not launch while pairing/architecture review remains open and peer per-region result is negative.
- SW_0040: complete diagnostic on SW0038 spike5 seed0, IDs 1320-1639. Peer component readout threshold .05 scores HDF5 .132102/.197586/.151268 and identity-mapped peer targets .272610/.124795/.281338; same predictions only, cross-target diagnostic, not classifier replacement. Pairing evidence is recorded in the SW0040 JSON and review 0003.
- SW_0041: complete for seeds 0/1/2, shared projection. Formal peer validation T256/64 means at threshold .20 are .560392/.610688/.404574; .35 gives .556789/.609456/.406034, .50 .546366/.602581/.402983. T1024/512 threshold .10 is best among common rows at .574026/.605911/.415638 (per-seed FG-ARI .3700/.6543/.6978; seed0 collapsed); .05 gives .572251/.608677/.413222 and .15 .573070/.604528/.415632. Same predictions on HDF5 score approximately .0369/.2502/.1294, consistent with target/renderer mismatch. Direct peer-exact seed0 epoch-1 loss 23.52399481 matches original 23.523995. See SW0041 README and `peer_updates/REVIEW_0003_20261004.md`.
- SW_0043: seed-0 factorized training completed all 40 epochs with logs matching the peer exact recipe. A learned-state comparison finds every learned tensor equal; only `sc` differs (peer identity versus local Pearson matrix). Because `structural_weight=0` and learned graph forward ignores `sc`, this does not affect computation. Same-input CPU T16 graph/theta/membrane/spike outputs are bitwise equal. Formal peer-validation short and long evaluations also completed; long T1024/settle512 scores against peer targets are .640695/.632759/.463078 at threshold .10 and .656050/.614347/.480345 at .35. See `SW_0043_peer_exact_dynamics/results/local_factorized_vs_peer_original.json`. Do not attribute prior seed-0 divergence to factorized arithmetic.
- The direct peer-code seed-0 retraining completed with final loss 17.22858517. All 17 tensors and the serialized checkpoint SHA-256 are exactly identical to the original BIM6 seed-0 checkpoint (`503c589c...be75be`).
- SW_0042 uses the verified factorized backend. Seed0 completed 40 epochs at final loss 18.82954237 and HDF5 validation IDs 1320-1639: short T256/64 threshold .35 scored .582002/.600063/.447548 (FG-ARI/foreground IoU/matched-object IoU). Long T1024/512 threshold .35 scored .598174/.605847/.461779; threshold .20 scored .596839/.615626/.457246 and had the best FG IoU. Long count exact/MAE is .1906/1.6344 at .35 and .2156/1.6906 at .50. These are seed0 observations, not common three-seed threshold selection. Seeds1/2 remain training on GPUs2/0.
- Peer uncommitted spatial-spike readout completed on validation300 at T1024/512, sync .10: sigma 1.5 seed triples are .6646/.6242/.4786, .7149/.7238/.4879, and .7228/.6802/.4893; mean .700767/.676067/.485267. Treat as a promising classifier candidate, not pooled evidence. SW0044 ports it opt-in and will compare S-only, S×G, and G-only under a fair threshold grid on aligned HDF5 validation.
- SW0044 spatial-affinity short/long evaluation is waiting for assigned GPU3 to become idle; SW0042 seed0 base evaluation is complete. SW0045 full320 aligned stage-signal diagnostic remains queued after the SW0044 completion marker, so GPU3 work is serialized.
- SW_0046 aligned official Slot Attention checkpoint audit completed on HDF5 IDs 1320-1639 (320 valid): FG-ARI .8946414575, foreground IoU .2235111789, matched object IoU .2459682554. IDs 1000-1319 scored .8901145722/.2122510283/.2354868700, supporting stability across the split shift. Against this single-checkpoint reference, the current SW0042 seed0 gap is concentrated in ARI; both IoUs exceed the reference. This is not a three-seed mean or goal success. See `SW_0046_aligned_slot_audit/README.md`.
- SW_0040 peer BIM6 count diagnostic (peer validation 6000-6999, 1000 images/seed, sync .35, steps256/settle64): exact count .192/.189/.204 (mean .195), MAE 1.635/1.624/1.543 (mean 1.601), within-one .522/.547/.572 (mean .547), bias -.819/-.852/-.227 (mean -.633). Predicted mean count 5.416/5.383/6.008 vs target 6.235. Dynamic count inference works, but accurate count is not solved; results and method are documented in SW0040 README and peer review 0003.
- SW_0040 peer BIM6 long-count diagnostic (peer validation 6000-6999, 1000 images/seed, T1024/settle512/sync .10): exact .215/.187/.209 (mean .203667), MAE 1.499/1.595/1.488 (mean 1.527333), within-one .571/.552/.581 (mean .568), bias -.211/-.867/-.358 (mean -.478667); predicted mean count 6.024/5.368/5.877 vs target 6.235. This is a modest improvement over T256/64/.35, but exact count remains about 20.4%; dynamic count inference works, accurate count is not solved. JSONs are in `SW_0040_peer_transfer/results/peer_count_long/`.
- SW0048 checkpoint provenance is now explicit: the aligned predictor accepts checkpoint source, training seed/protocol, and inference RNG seed; its defaults preserve the existing official-transfer source/seed semantics. The shared scorer describes whichever checkpoint source its protocol records.
- SW0056 is prepared but not launched. It uses train IDs 0-999 plus 1640-3139, exactly ten real exposures per image (25,000 total), 1,562 full batches plus a true 8-image final batch (1,563 updates). The CPU-only sequential runner uses a batch-8 auxiliary official model for that last update, verifies ordered variable signatures, and applies gradients to primary variables without padding. Protocol tests cover exact counts/order; a 24-image two-update smoke also exercises the batch-8 path, finite loss, checkpoint saving, and primary weight change. See `SW_0056_matched_slot_2500/README.md`.

- SW0054 pilot completion: real-asset preflight and the seed0 32-image short pilot are complete; raw JSON and validator-recomputed summary are present under `collaborative_test/SW_0054_causal_mechanism_ablation/results/`. Normal spike-CC was .6685/.6920/.4899. Permuted gate fell to .0055/.0880/.0278 with event-rate ratio 1.0005 and membrane-variance ratio .9990; permuted carrier remained .6637/.6798/.4823; K=0 scored .2377/.4388/.2212 with activity scale largely preserved. This is a small pilot, not a full-validation mechanism claim.
- SW0058 kept the SW0053 1,000-scene gamma and low-LR/epoch25 recipe fixed while changing only base `primary_loss_weight` from 1 to 0; component-spike weight remained 5. The canonical preflight passed. Seeds 0/1 completed the registered long T1024/settle512 spike-CC threshold .50 endpoint and were lower than matched SW0053 baselines on all three primary metrics. Seed2 was not launched; see `SW_0058_component_spike_only_loss/results/stage1_seed0_seed1_stop.{json,md}`.

- SW0059 completed after static checks, five focused unit tests, strict checkpoint loading, and a four-image real-asset smoke test. On the fixed 32-image pilot, centered delayed gating changed FG-ARI/foreground IoU/matched-object IoU from .668520/.692026/.489879 to .677768/.674981/.511379. Because foreground IoU fell while the other two improved, the exact centered mode is not promoted directly to training. No training or data changed.

- SW0060 preserved the per-component carrier and centered only the delayed mask. Static checks, six unit tests, and a real-asset smoke passed. The fixed 32-image pilot lowered all three metrics slightly, so this gate variant is rejected. Peer commits 084a2ef/96d74d6 instead identify input feature separation as the main bottleneck; see peer_updates/REVIEW_0006_20261005.md.

- SW0061 tested the peer feature-bottleneck implication without labels by substituting RGB/chroma/edge patch gamma into the fixed checkpoint. It fell to .308029/.383089/.190871 and strongly under-counted, showing that the trained core does not accept an arbitrary feature basis. The next feature intervention must jointly train the encoder/core or adapt the core to the new basis.

- SW0062 implements the peer-guided feature intervention: frozen DINOv2-small 16x16 patch tokens, an 8D PCA basis fitted only on training IDs0-999, and a freshly trained SNN/Kuramoto core. Feature extraction is label-free and running on CPU; a validated five-epoch seed0 direction pilot is queued after the existing SW0055/SW0057 GPU contract finishes.

- SW0055 and SW0057 completed. Unique-data training improved all three long-window fixed-threshold means over SW0053 to .626733/.423340/.440119, but seed FG-ARI ranged .817116 to .448238. The membrane readout is more stable at FG-ARI .738478 while its IoUs remain .222631/.368848.
- SW0062 stopped generic DINO/PCA after a five-epoch pilot and feature diagnostics showed worse CLEVR within-object consistency than native gamma. SW0063 preserved the native basis and improved FG-ARI +.0293 through label-free smoothing, but failed the all-three endpoint gate.
- SW0064 completed the activation-flow reference. Identical gamma diverges most at graph-to-Kuramoto conversion: late PLV object margin is .8167 for seed0 and .5167 for seed2. Dendritic-to-membrane propagation is nearly lossless; membrane-to-spike is a smaller secondary loss.
- SW0065 isolated that instability causally. Swapping seed0's trained graph generator into seed2 improves the fixed 32-image result from .3895/.2003/.1822 to .7036/.3081/.5376; drive and Kuramoto parameter swaps do not improve all three.
- SW0066 completed. Fixed graph initialization improves FG-ARI and matched-object IoU for both weak seeds and all three metrics for seed2, but seed1 foreground IoU falls .0285, so the registered per-seed gate fails. This retains graph-init0 as an activation-stability baseline without claiming a final endpoint win.
- SW0067 rejected fixed foreground-area calibration: three-seed pilot mean changed from .6187/.4785/.4369 to .5120/.4917/.3464 and count MAE worsened 1.927 to 2.490. Matching area removes object patches; no further coverage sweep is planned.
- SW0068 completed. Joint LR3e-6 and LR3e-5 both lower all seed1 pilot metrics. Component swaps show LR3e-5 learned features in the frozen old core improve FG-ARI/object IoU to .6371/.4773 but lose foreground IoU, while the jointly updated core with native gamma sharply lowers both IoUs. The failure is harmful core co-adaptation, not wholesale feature collapse.
- SW0069 completed. Frozen-core encoder tuning scores .6094/.3821/.4380 without anchoring and .6141/.3790/.4586 with anchor100, both below the SW0066 seed1 baseline .6317/.3852/.4617. Anchor100 reduces gamma RMS drift .0487 to .0090, but the binding loss remains misaligned as an encoder objective; further LR/anchor/epoch tuning is stopped.
- SW0070 establishes the new fixed-contract three-seed spike reference at `.677448/.435439/.479878`. Against SW0055, graph-init0 improves FG-ARI/foreground IoU/matched-object IoU by `+.050715/+.012099/+.039759`; seed2 FG-ARI rises from `.448238` to `.547068`. This is the stable-core baseline for subsequent tests.
- SW0071 completed all three seeds. Relative to its same-run spike-CC baseline, spike-freeze changes FG-ARI/foreground IoU/matched-object IoU by `+.022007/0/-.033346`; the closest balanced dynamic-spike row changes them by `+.001059/0/-.010966`. Continuous activation carries grouping information, but whole-component repartitioning harms overlap, so no readout is accepted and this direction stops.
- SW0072 is the new all-metric lead. Loading and freezing the trained seed0 graph while retraining seed1/2 downstream dynamics gives full seed results `.817116/.730071/.643890`, `.754097/.377045/.570852`, and `.773010/.326999/.635497`; the mean `.781407/.478038/.616746` improves SW0070 by `+.103960/+.042599/+.136868`. Graph tensors remain bitwise equal to the source. This strongly supports the paper mechanism: a stable learned graph enables downstream Kuramoto-to-spike activation to form better object groups, while seed1/2 foreground localization remains the next bottleneck.
- Peer `patch_v2:b9c31ed` shows why prior joint feature training collapsed: PLV can be minimized by making patches alike, while a moderate phase-slot RGB reconstruction term keeps features image-informative. Its weight1 result improves peer FG-ARI/foreground IoU but lowers object IoU, and weight5 collapses; scores are not pooled across protocols. SW0073 tests weights .3/1.0 on our aligned data with the SW0072 graph frozen. See `peer_updates/REVIEW_0009_20261005.md`.
- SW0073 completed without an advancing joint arm. Weight1 is closest at `.68802/.46849/.50854` versus SW0072 seed1 `.70876/.46810/.50988`. Component swaps show its learned features improve FG-ARI/object IoU in the unchanged SW0072 core to `.71894/.53055`, while the jointly updated core on native gamma falls to `.65326/.45116/.46236`. Reconstruction fixes feature collapse, but core co-adaptation remains harmful; freeze the full core next and use an anchor to recover foreground localization.
- SW0074 freezes the complete SW0072 seed1 core and tunes only the encoder with reconstruction weight1 plus gamma anchors 10/100. Unit tests and a real HDF5/checkpoint smoke confirm finite reconstruction gradient, encoder change, and bitwise unchanged core/graph. The two seed1 arms are running on GPUs0/1.
- SW0074 completed with no advancing arm. Anchor10 scores `.67210/.44354/.50297`; anchor100 scores `.68271/.41791/.50016`, both below SW0072 seed1. Core/graph changes are zero, and anchor100 reduces gamma drift to `.01004`, yet foreground IoU falls more than with anchor10. Stop scalar gamma-anchor tuning; preserve the actual patchwise membrane activation profile next.
- SW0075 completed with no advancing arm. Activity-anchor 1k scores `.679576/.441083/.499716`; 10k scores `.706576/.477694/.500626` versus SW0072 seed1 `.708757/.468105/.509881`. The stronger anchor improves foreground IoU by `.009590` but lowers FG-ARI/object IoU by `.002181/.009254`; the weaker arm lowers all three. Core and graph remain exact. Time-mean membrane preservation is insufficient for object binding, so this route stops and the next test preserves the image-conditioned graph topology directly.
- SW0076 completed with no advancing arm. Graph-anchor 1k scores `.680963/.419769/.500319`; 10k scores `.667527/.442960/.505828`, both below SW0072 seed1 `.708757/.468105/.509881` on all three metrics. The stronger arm reduces final graph MSE to `1.62804e-6`, and core/graph parameters remain exact, so failure is not loose preservation. Stop gamma/activity/graph anchor refinement and address the frozen SW0072 foreground/count classifier bottleneck directly.
- SW0077 completed. The peer-inspired binary crossing classifier produces no foreground because SW0072's settled component traces have crossing rate `.999938`, 98.82% always-on units, and gated/binary affinity correlation only `.00342`; the peer non-saturation result is configuration-specific. A fixed GT-free RGB-border whole-component veto raises foreground IoU `.468105 -> .515963` but lowers FG-ARI/object IoU `.708757/.509881 -> .476408/.311105` and worsens count MAE `1.375 -> 3.156`. Stop both controls without threshold sweeps.
- SW0078 completed. Injecting a label-free RGB bilateral graph at the SW0065 causal bottleneck does not help: RGB-only scores `.568771/.418120/.353497`, while learned/RGB blends at alpha .25/.50 score `.680919/.422639/.493573` and `.686318/.441609/.475547`, all below the exact SW0072 baseline `.708757/.468105/.509881`. Activation still propagates, so the loss is topology quality. Stop RGB graph tuning and preserve/prune the learned graph next.
- SW0079 completed. Coarse pruning of the frozen learned graph lowers every metric: top-k16 scores `.635492/.460102/.461217` and top-k8 `.614964/.427776/.445628` versus top-k32 `.708757/.468105/.509881`. Group counts remain similar, indicating poorer component membership rather than a simple count shift. Together with SW0078, this stops inference-time graph surgery; graph improvement must come from a better training signal.
- SW0080 completed with no advancing arm. Graph-only reconstruction .3 scores `.682508/.464112/.492403`; weight1 scores `.667994/.471223/.492245` versus SW0072 seed1 `.708757/.468105/.509881`. Weight1 gains only `.003119` foreground IoU while losing `.040763/.017635` FG-ARI/object IoU. Graph parameters change by `.004097/.003508`, all non-graph tensors stay exact, and reconstruction loss does not decrease. Stop reconstruction/graph-LR tuning; the next loss must teach graph edge consistency directly.
- SW0081 completed. The calibrated 0.1x arm scores `.706200/.464029/.513687` versus the fixed seed1 pilot baseline `.708757/.468105/.509881`: matched-object IoU improves `.003806`, while FG-ARI and foreground IoU fall `.002556/.004075`. The 1x arm scores `.707583/.445272/.507302` and is rejected. Encoder and non-graph core remain exact. One fixed 0.1x epoch-1--5 trajectory check is warranted; no weight sweep or full-320 promotion is warranted.
- SW0082 completed the exact 0.1x epoch trajectory. Epochs1--4 score `.699289/.463458/.504113`, `.697220/.467795/.507192`, `.686575/.465248/.507033`, and `.682034/.463918/.501284`; all are below the SW0072 seed1 pilot on all three metrics. Epoch5 is bitwise/re-evaluation exact with SW0081 at `.706200/.464029/.513687` and improves only object IoU. Stop horizontal-flip graph equivariance without a weight sweep or full-320 promotion.
- SW0083 completed the frozen SW0072 membrane-threshold diagnostic. The `.06` control has binary event/always-on fractions `1.0/1.0` and scores `.708757/.468105/.509881`. vth `.5` restores some variation (`.9388` event rate, `.7018` always-on) but lowers all metrics to `.595236/.330631/.481278`; vth `1.0` nearly collapses and `2.0` is empty. The stable graph's scored structure is gate-amplitude dominated; stop inference threshold tuning without a finer sweep or full promotion.
- The matched Slot SW0056 seed0 process remains live at step 1200/1563 under the original PID. TensorFlow 2.15 sees no GPU; its validated partial artifacts are preserved and the run has never been restarted.
- Peer PV2_0037/0038 confirms that frozen DINO, arbitrary feature clustering, and self-bootstrap do not supply object information. Joint RGB-to-gamma/core training is the next major feature route after the graph-stability gate; see peer_updates/REVIEW_0008_20261005.md.
- Peer `patch_v2:4b214e2` ported SW0072's graph freeze and reached `.7250/.7188/.4953`, above its single transferred Slot reference on all three metrics. The same Slot checkpoint scores FG-ARI `.6195` there and `.8946` here because the render, resolution, patch footprint, split, and foreground prevalence differ. No score is pooled; the reciprocal graph-stability evidence is incorporated while SW0084 tests the remaining joint feature/graph route. See `peer_updates/REVIEW_0011_20261005.md`.
- SW0056 matched-data Slot training is now live for all three seeds on CPU. Seed0 continues its original validated run; seeds1/2 were started concurrently after a 48-CPU/>80-GiB resource check. The data, exposure count, optimizer schedule, seed orders, and aligned validation contract are unchanged, and a separate durable completion queue will validate all three outputs before writing the comparison summary.

2026-10-07 update: SW0098 completed all three seeds on the registered
256/128 training window. Matched-control mean is .773412/.579282/.597119
versus SW0097 frozen .776348/.577070/.601537. FG-ARI fell in every seed,
foreground IoU rose in every seed, and matched-object IoU fell in every seed.
The long-window setting does not meet the all-metric validation target and
does not exceed Slot mean FG-ARI. All three completed records and factual
comparisons are archived in SW_0098_long_window; independent reserve
90640-90959 remains unread.

2026-10-07 update: SW0099 completed a corrected validation-only trace on 16 IDs 1320-1335, batch 8, for SW0095/97/98 seeds 0/1/2. The graph-generator states are bitwise equal. SW0098 endpoint means were .810284/.555407/.641283, slightly below matched SW0097 on all three metrics; these small-subset scores are not full-320 results. Product-PLV AUC rises slightly, while the first repeatable distance-controlled decline is at the delayed raw gate: near and mid-bin AUC fall for all three seeds and the decline persists at spike. This is an observational stage association, not a causal bottleneck finding. Fixed-threshold connectivity worsens only for seed 0, so there is no universal fragmentation result. The first run's dendrite trace was invalidated for a fold-order error; corrected traces passed a synthetic ordering check, exactly matched all endpoints, and reproduced all non-dendrite stages. Only GPU 0 was used after checking ownership. See SW_0099_stage_flow_diagnosis/README.md and summary.json for results and execution provenance.

2026-10-07 update: SW0100 phasor-imaginary raw gate completed its matched
three-seed 256-update continuation and full320 validation. Mean
.763415/.706827/.590691 (FG-ARI/foreground IoU/object IoU); vs frozen SW0097,
paired mean deltas are -.012933/+.129757/-.010846. FG-ARI falls all3, foreground
IoU rises all3, and object IoU rises 1/3. The exact user-priority 70k gate
requires mean FG-ARI +.01 and gains in at least2 seeds, and both IoUs +.05 over
matched Slot; the FG-ARI clauses fail, so no 70k training was launched.
Corrected fixed16 source/candidate gate-swap records are archived; first-pass
schema/settle errors and their outputs are retained and labeled. See
SW_0100_phasor_imag_raw_gate/README.md and results_archive/summary.json.
The user set a hard two-hour work deadline at 2026-10-07 10:37:49 UTC; the
operator must not queue work beyond that deadline. The next approved work is
SW0101 Stage1 teacher-mask QA only, before loss implementation/training.


2026-10-07 update: SW0101 Stage 1 froze the prescribed graph/cosine teacher masks on validation IDs1320-1335 before opening labels, then failed the fixed acceptance gate. Near-bin positive precision was .9130 (483/529) and mid-bin positive precision .6757 (50/74), below .95 in both; mid coverage was 10/16, below 12/16. Negative precision was 1.0000 and .9984 respectively. Positive components merged multiple foreground instances in 8 cases and connected foreground to background in 19 cases. The teacher-mask recipe is stopped with no threshold tuning, loss implementation, or training. See SW_0101_teacher_mask_qa/README.md and qa_results.json; the frozen pre-GT mask artifact is preserved alongside the result.

2026-10-07 update: SW0102 completed the matched three-seed cannot-link continuation (256 updates each) and full320 validation. Against matched SW0097 frozen controls, mean FG-ARI/foreground IoU/object IoU was .773720/.577959/.600421 versus .776348/.577070/.601537; paired deltas were -.002628/+.000888/-.001116, and FG-ARI fell in all three seeds. The registered promotion gate failed; no paired bootstrap or 70k continuation was run. The cut loss remained active; its mean was .05227/.09453/.10773 by seed, and 33.8%/59.0%/75.2% of selected directed negatives were connected under the strict q>.40 forest margin. SW0097 controls do not contain matching mask/connectivity counts, so no conflict-reduction comparison is claimed. Full contracts, code SHAs, GPU ownership, source hashes, finite histories, 320-image valid counts and per-seed results are in SW_0102_cannot_link_draft/results_archive/summary.json. No model weights were archived.

2026-10-07 update: SW0103 applied the fixed one-pass triangle-supported q>=.50 readout to three frozen SW0097 controls on IDs1320-1335. The raw classifier reproduced all archived per-image metrics exactly. Triangle mean deltas were -.000969/-.000425/+.002844 and only one seed gained FG-ARI; the first16 gate failed, so no full320 run followed. Cross-instance coassignment fell .027603->.026014, same-instance recall drop was .00185, and no new zero-group predictions occurred. All predictions were built before label access. Per-image records, edge classes, graph bridges and execution provenance are archived under SW_0103_triangle_readout/results_archive/first16.json.

2026-10-07 update: SW0104 completed a read-only output-factor audit on the three frozen SW0097 controls. Actual event-times-gate spikes reproduce archived fixed16 metrics exactly. The strict gradient decomposition fails four checks in seed2, and the gate scalar formula check remains above its 1e-6 limit; binary-event Pearson readout degenerates under nearly constant event traces. No training or full320 follow-up. See SW_0104_causal_credit/README.md and results_archive/summary.json. The hard stop is 2026-10-07 10:37:49 UTC; do not queue work past it.
