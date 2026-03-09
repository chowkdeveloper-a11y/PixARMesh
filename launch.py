import os
import signal
import psutil
from datetime import datetime
from accelerate.commands.launch import launch_command_parser, launch_command


def handle_sigusr(signum, _):
    print(f"Received SIGUSR: {signum}, forwarding to child processes")
    parent = psutil.Process()
    for child in parent.children(recursive=False):
        try:
            child.send_signal(signal.SIGUSR1)
        except Exception as e:
            print(f"Failed to send SIGUSR1 to process {child.pid}: {e}")


def ensure_run_timestamp():
    if "RUN_TS" in os.environ and os.environ["RUN_TS"]:
        return

    stable_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    if stable_job_id:
        os.environ["RUN_TS"] = stable_job_id
        print(f"[launch.py] Using RUN_TS from job id: {os.environ['RUN_TS']}")
        return

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.environ["RUN_TS"] = ts
    print(f"[launch.py] Generated RUN_TS: {ts}")


def main():
    signal.signal(signal.SIGUSR1, handle_sigusr)
    ensure_run_timestamp()
    parser = launch_command_parser()
    args = parser.parse_args()
    if args.config_file is None and os.path.exists("accelerate.yaml"):
        args.config_file = "accelerate.yaml"
    if args.debug:
        args.num_processes = 1
        args.training_script_args.extend(
            [
                "all.output_dir=debug/runs",
                "~train.train_args.dataloader_num_workers",
                "~train.train_args.dataloader_prefetch_factor",
                "~train.train_args.dataloader_persistent_workers",
            ]
        )
    launch_command(args)


if __name__ == "__main__":
    main()
