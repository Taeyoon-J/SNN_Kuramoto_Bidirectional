"""Finish experiment 2 without restarting a live or completed seed."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/Data0/kevinswk/patch_v2_sw')
DIR = ROOT / 'collaborative_test/SW_0092_cross_dataset_training'
STATE = ROOT / 'trained_models/SW0092_OUR_OFFICIAL_QUEUE.json'
PY = '/Data0/kevinswk/envs/snn/bin/python'


def state(status, **details):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps({'status': status, **details}) + '\n')
    temporary.replace(STATE)


def gpu_available(reserved=()):
    for gpu in (0, 1, 3):
        if gpu in reserved:
            continue
        if gpu == 3 and not all(
            (ROOT / f'trained_models/SW0092_slot_our70000_eval/seed{seed}_epoch{epoch}/SCORING_COMPLETED').is_file()
            for seed in (1, 2) for epoch in (1, 3, 10)
        ):
            continue
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True, text=True, check=True,
        )
        if not any(character.isdigit() for character in result.stdout):
            return gpu
    return None


def wait_gpu():
    while True:
        gpu = gpu_available()
        if gpu is not None:
            return gpu
        time.sleep(30)


def matching_train_process(output):
    listing = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    return any('snn_kuramoto_bidirectional.training.train_s2net_core' in line
               and f'--save-path {output}/core.pt' in line for line in listing.splitlines())


def recover_finished_training(output):
    log = output / 'training.log'
    files = [output / 'core.pt', *(output / 'checkpoints' / f'epoch_{epoch:02d}.pt' for epoch in (1, 3, 10))]
    if not log.is_file() or not all(path.is_file() and path.stat().st_size for path in files):
        return False
    text = log.read_text()
    if 'Epoch 0010/0010 |' not in text or f'trained S2NetCore: {output}/core.pt' not in text:
        return False
    import hashlib
    import torch
    models = [torch.load(path, map_location='cpu', weights_only=True) for path in files]
    if not models[0] or any(set(model) != set(models[0]) for model in models):
        raise RuntimeError(f'Checkpoint state keys do not match: {output}')
    if any(not torch.isfinite(value).all().item() for model in models for value in model.values()):
        raise RuntimeError(f'Nonfinite saved model: {output}')
    if any(not torch.equal(models[0][key], models[-1][key]) for key in models[0]):
        raise RuntimeError(f'Final model differs from epoch10: {output}')
    evidence = {'reason': 'missing wrapper marker; completed trainer and saved states verified',
                'epoch10_matches_final_state': True,
                'checkpoint_sha256': {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    (output / 'completion_recovery.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (output / 'TRAINING_COMPLETED').write_text('complete; recovered from validated artifacts\n')
    return True


def train_all_seeds():
    launched = {}
    last_seen = {}
    while True:
        completed, running, pending = [], [], []
        for seed in (0, 1, 2):
            output = ROOT / f'trained_models/SW0092_our_on_official_s{seed}'
            child = launched.get(seed)
            if child is not None and child[0].poll() not in (None, 0):
                raise RuntimeError(f'Training seed {seed} exited {child[0].returncode}')
            if (output / 'TRAINING_COMPLETED').is_file():
                required = [output / 'core.pt', *(output / 'checkpoints' / f'epoch_{ep:02d}.pt' for ep in (1, 3, 10))]
                if not all(path.is_file() and path.stat().st_size for path in required):
                    raise RuntimeError(f'Incomplete artifacts under {output}')
                completed.append(seed)
            elif (child is not None and child[0].poll() is None) or matching_train_process(output):
                running.append(seed)
                last_seen[seed] = time.monotonic()
            elif output.exists():
                # Allow the preserved trainer's wrapper to write its completion marker.
                if recover_finished_training(output):
                    completed.append(seed)
                elif seed in last_seen and time.monotonic() - last_seen[seed] < 60:
                    running.append(seed)
                else:
                    raise RuntimeError(f'Partial training output preserved; inspect {output}')
            else:
                pending.append(seed)
        if len(completed) == 3:
            return
        reserved = {gpu for process, gpu in launched.values() if process.poll() is None}
        for seed in pending.copy():
            gpu = gpu_available(reserved)
            if gpu is None:
                break
            process = subprocess.Popen(['bash', str(DIR / 'run_our_official.sh'), str(gpu), str(seed)])
            launched[seed] = (process, gpu)
            reserved.add(gpu)
            running.append(seed)
            pending.remove(seed)
        state('training', completed_seeds=completed, running_seeds=running, waiting_seeds=pending)
        time.sleep(30)


def main():
    train_all_seeds()
    # An independent early evaluation may still own seed1's output files.
    while any(f'bash {DIR}/evaluate_seed1_early.sh' in line
              for line in subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()):
        state('waiting_for_seed1_early_evaluation')
        time.sleep(30)
    evaluation = ROOT / 'trained_models/SW0092_our_on_official_eval'
    for epoch in (1, 3, 10):
        for seed in (0, 1, 2):
            result = evaluation / f'seed{seed}_epoch{epoch}.json'
            if result.is_file():
                report = json.loads(result.read_text())
                if report['ids'] != [1320, 1639] or report['ground_truth_used_for_prediction']:
                    raise RuntimeError(f'Invalid evaluation contract: {result}')
                continue
            gpu = wait_gpu()
            state('evaluating', seed=seed, epoch=epoch, gpu=gpu)
            subprocess.run(['bash', str(DIR / 'evaluate_our_official.sh'), str(gpu), str(seed), str(epoch)], check=True)
    subprocess.run([PY, str(DIR / 'summarize_our_official.py'), str(evaluation), str(evaluation / 'summary.json')], check=True)
    state('complete')
if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed', error=str(error))
        raise
