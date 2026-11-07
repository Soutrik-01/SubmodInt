"""Trains one DSPN run with validation-based early stopping followed by a short anneal, then evaluates it.

main.py saves a checkpoint every --eval_every epochs, which is scored on the validation split (evaluate.py; score =
mean of FL Eval, offline and online accuracy). When the score has not improved by --min_delta for --patience epochs,
the run is stopped and resumed from that checkpoint with --anneal_from, which decays the learning rate linearly to 0
over --anneal_epochs: a checkpoint from the middle of the cyclic schedule is still at a high learning rate. The
checkpoint with the best validation score is the run's result and the only one evaluated on the test split.
Re-running the driver resumes from the latest checkpoint.
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
AUTHORS = ("--nesting --seed 10 --dsf_weight_decay 1e-4 --kw 0.3 --K 100 --nesting_interval 10 --beta 0.01 "
           "--scheduler cyclic --lr_gamma 0.7 --MAX 1 --tau 20 --set_size 100 --bsz 120 --n_views 4 --lr 1e-3 "
           "--roof_consistency_loss 0.25 --nested_peripteral_loss_with_augmentation 0.25 "
           "--singleton_roof_consistency_loss 0.01 --prop_fullGS 0.2 --submodular_feedback 0 --feedback_every 15 "
           "--feedback_coefficient 0.1 --sg_budget 5 --r_size 1000 --use_training_pair_fb --matroid_rank 1 "
           "--matroid_rank_tradeoff 0.2 --matroid_v_non_matroid --class_balanced_feedback --target_computation_bsz 40 "
           "--target_feedback --nnkmeans_feedback").split()
METRICS = ("fl_eval", "offline_acc", "online_acc")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model_type", required=True)
    ap.add_argument("--extra", default="", help="extra main.py arguments")
    ap.add_argument("--dset", default="IN100")
    ap.add_argument("--root", required=True, help="its X_test is the test split")
    ap.add_argument("--val_root", required=True, help="training data, and its X_test is the validation split")
    ap.add_argument("--n_epochs", type=int, default=500)
    ap.add_argument("--stop_after", type=int, default=None,
                    help="end training after this epoch, keeping the --n_epochs learning-rate schedule")
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--min_delta", type=float, default=0.005)
    ap.add_argument("--anneal_epochs", type=int, default=25)
    ap.add_argument("--eval_seeds", type=int, default=5)
    ap.add_argument("--port", type=int, default=29700)
    cli = ap.parse_args()

    run_dir = f"{cli.runs_dir}/{cli.name}"
    ckpts = f"{run_dir}/trainlogs/{cli.name}"
    os.makedirs(run_dir, exist_ok=True)
    state_path = f"{run_dir}/early_stopping.json"
    state = {"scores": {}, "anneal_from": None}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)

    def save():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)

    def ckpt(epoch):
        return f"{ckpts}/dspn_ckpt_{epoch}.pt"

    def saved_epochs():
        return sorted(int(p.rsplit("_", 1)[1][:-3]) for p in glob.glob(ckpt("*")))

    def last_epoch():
        a = state["anneal_from"]
        last = cli.n_epochs - 1 if a is None else a + cli.anneal_epochs
        return last if cli.stop_after is None else min(last, cli.stop_after - 1)

    def launch(resume):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "1", "--master_port", str(cli.port),
               f"{CODE}/main.py", "--model_type", cli.model_type, "--path_directory", cli.name, "--fl_on_the_fly",
               "--save_every", str(cli.eval_every), "--n_epochs", str(cli.n_epochs), "--root", cli.val_root,
               "--dset", cli.dset, *AUTHORS, *cli.extra.split()]
        if resume is not None:
            cmd += ["--load_from_ckpt", ckpt(resume)]
        if state["anneal_from"] is not None:
            cmd += ["--anneal_from", str(state["anneal_from"]), "--anneal_epochs", str(cli.anneal_epochs)]
        log(f"{cli.name}: training" + ("" if resume is None else f" from the epoch-{resume + 1} checkpoint")
            + ("" if state["anneal_from"] is None else f", annealing to 0 by epoch {last_epoch() + 1}"))
        with open(f"{run_dir}/train.log", "a") as out:
            return subprocess.Popen(cmd, cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)

    def stop(proc):
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def evaluate(epoch, split, root):
        out = f"{ckpts}/eval_{split}_epoch{epoch}.json"
        if not os.path.exists(out):
            with open(f"{run_dir}/eval.log", "a") as log_file:
                subprocess.run([sys.executable, f"{CODE}/evaluate.py", "--run_dir", run_dir, "--uid", cli.name,
                                "--epoch", str(epoch), "--root", root, "--split", split, "--n_seeds", str(cli.eval_seeds)],
                               cwd=CODE, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        with open(out) as f:
            return json.load(f)

    def plateaued(epoch):
        best, best_epoch = float("-inf"), None
        for e in sorted(int(k) for k in state["scores"]):
            if state["scores"][str(e)] > best + cli.min_delta:
                best, best_epoch = state["scores"][str(e)], e
        return epoch - best_epoch >= cli.patience

    proc, failures = None, 0
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(f"{cli.name}: driver terminated"))
    try:
        while True:
            epochs = saved_epochs()
            finished = bool(epochs) and epochs[-1] >= last_epoch()
            if (cli.stop_after is not None and finished and proc is not None and proc.poll() is None
                    and time.time() - os.path.getmtime(ckpt(epochs[-1])) > 60):
                stop(proc)
            if proc is not None and proc.poll() is not None:
                if not finished:
                    failures += 1
                    log(f"{cli.name}: training exited with code {proc.returncode} before its last epoch")
                    if failures > 3:
                        sys.exit(f"{cli.name}: giving up after {failures} failures")
                proc = None
            if proc is None and not finished:
                proc = launch(epochs[-1] if epochs else None)
            running = proc is not None and proc.poll() is None
            for e in epochs:
                if str(e) in state["scores"] or (running and time.time() - os.path.getmtime(ckpt(e)) < 60):
                    continue
                try:
                    r = evaluate(e, "val", cli.val_root)
                except subprocess.CalledProcessError:
                    log(f"{cli.name}: evaluating epoch {e + 1} failed (see eval.log); will retry")
                    time.sleep(300)
                    break
                state["scores"][str(e)] = sum(r[m] for m in METRICS) / len(METRICS)
                save()
                log(f"{cli.name}: epoch {e + 1} validation score {state['scores'][str(e)]:.4f} ("
                    + ", ".join(f"{m} {r[m]:.4f}" for m in METRICS) + ")")
                if state["anneal_from"] is None and plateaued(e) and e + cli.anneal_epochs < cli.n_epochs - 1:
                    log(f"{cli.name}: no validation gain of {cli.min_delta} in {cli.patience} epochs; stopping to anneal")
                    if running:
                        stop(proc)
                    for later in (x for x in saved_epochs() if x > e):
                        os.remove(ckpt(later))
                        state["scores"].pop(str(later), None)
                    state["anneal_from"] = e
                    save()
                    proc = launch(e)
                    break
            if finished and proc is None and all(str(e) in state["scores"] for e in epochs):
                break
            time.sleep(30)
    finally:
        if proc is not None and proc.poll() is None:
            stop(proc)

    best = max(state["scores"], key=lambda k: state["scores"][k])
    test = evaluate(int(best), "test", cli.root)
    state.update(best_epoch=int(best), test={m: test[m] for m in METRICS})
    save()
    log(f"{cli.name}: done; best validation checkpoint is epoch {int(best) + 1} (score {state['scores'][best]:.4f}); test "
        + ", ".join(f"{m} {test[m]:.4f}" for m in METRICS))


if __name__ == "__main__":
    main()
