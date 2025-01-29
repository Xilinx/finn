import itertools
import sys
import os
import json
import time
import traceback
import onnxruntime as ort

from dut.mvau import bench_mvau
from dut.transformer import bench_transformer
from dut.transformer_radioml import bench_transformer_radioml
from dut.transformer_gpt import bench_transformer_gpt
from dut.fifosizing import bench_fifosizing, bench_metafi_fifosizing, bench_resnet50_fifosizing


def main(config_name):
    exit_code = 0
    # Attempt to work around onnxruntime issue on Slurm-managed clusters:
    # See https://github.com/microsoft/onnxruntime/issues/8313
    # This seems to happen only when assigned CPU cores are not contiguous
    _default_session_options = ort.capi._pybind_state.get_default_session_options()
    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = 1
        _default_session_options.intra_op_num_threads = 1
        return _default_session_options
    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new

    # Gather job array info
    job_id = int(os.environ["SLURM_JOB_ID"])
    #TODO: allow portable execution on any platform by making as many env vars as possible optional
    print("Job launched with ID: %d" % (job_id))
    try:
        array_id = int(os.environ["SLURM_ARRAY_JOB_ID"])
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        print(
            "Launched as job array (Array ID: %d, Task ID: %d, Task count: %d)"
            % (array_id, task_id, task_count)
        )
    except KeyError:
        array_id = job_id
        task_id = 0
        task_count = 1
        print("Launched as single job")

    # Prepare result directory
    # experiment_dir = os.environ.get("EXPERIMENT_DIR") # original experiment dir (before potential copy to ramdisk)
    experiment_dir = os.environ.get("CI_PROJECT_DIR")

    artifacts_dir = os.path.join(experiment_dir, "bench_artifacts")
    print("Collecting results in path: %s" % artifacts_dir)
    os.makedirs(os.path.join(artifacts_dir, "tasks_output"), exist_ok=True)
    log_path = os.path.join(artifacts_dir, "tasks_output", "task_%d.json" % (task_id))
    
    # save dir for saving bitstreams (and optionally full build artifacts for debugging (TODO))
    # TODO: make this more configurable or switch to job/artifact based power measurement
    if job_id == 0:
        #DEBUG mode
        save_dir = experiment_dir + "_save"
    else:
        save_dir = os.path.join("/scratch/hpc-prf-radioml/felix/jobs/",
                            "CI_" + os.environ.get("CI_PIPELINE_IID") + "_" + os.environ.get("CI_PIPELINE_NAME"))
    print("Saving additional artifacts in path: %s" % save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Gather benchmarking configs
    if config_name == "manual":
        configs_path, config_select = os.path.split(os.environ.get("MANUAL_CFG_PATH"))
    else:
        configs_path = os.path.join(os.path.dirname(__file__), "cfg")
        config_select = config_name + ".json"

    # Load config
    config_path = os.path.join(configs_path, config_select)
    print("Loading config %s" % (config_path))
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        print("ERROR: config file not found")
        return

    # Expand all specified config combinations (gridsearch)
    config_expanded = []
    for param_set in config:
        param_set_expanded = list(
            dict(zip(param_set.keys(), x)) for x in itertools.product(*param_set.values())
        )
        config_expanded.extend(param_set_expanded)

    # Save config (only first job of array) for logging purposes
    if task_id == 0:
        with open(os.path.join(artifacts_dir, "bench_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        with open(os.path.join(artifacts_dir, "bench_config_exp.json"), "w") as f:
            json.dump(config_expanded, f, indent=2)

    # Determine which runs this job will work on
    total_runs = len(config_expanded)
    if total_runs <= task_count:
        if task_id < total_runs:
            selected_runs = [task_id]
        else:
            return
    else:
        selected_runs = []
        idx = task_id
        while idx < total_runs:
            selected_runs.append(idx)
            idx = idx + task_count
    print("This job will perform %d out of %d total runs" % (len(selected_runs), total_runs))

    # Run benchmark
    # TODO: integrate this loop (especially status logging) into the bench class
    # TODO: log additional info as artifact or directly into info section of json (e.g. dut, versions, date)
    # TODO: log stdout of individual tasks of the job array into seperate files as artifacts (GitLab web interface is not readable)
    log = []
    for run, run_id in enumerate(selected_runs):
        print(
            "Starting run %d/%d (id %d of %d total runs)"
            % (run + 1, len(selected_runs), run_id, total_runs)
        )

        params = config_expanded[run_id]
        print("Run parameters: %s" % (str(params)))

        log_dict = {"run_id": run_id, "task_id": task_id, "params": params}

        # Determine which DUT to run TODO: do this lookup more generically?
        # give bench subclass name directly in config?
        if config_select.startswith("mvau"):
            bench_object = bench_mvau(params, task_id, run_id, artifacts_dir, save_dir)
        elif config_select.startswith("transformer_radioml"):
            bench_object = bench_transformer_radioml(params, task_id, run_id, artifacts_dir, save_dir)
        elif config_select.startswith("transformer_gpt"):
            bench_object = bench_transformer_gpt(params, task_id, run_id, artifacts_dir, save_dir)
        elif config_select.startswith("transformer"):
            bench_object = bench_transformer(params, task_id, run_id, artifacts_dir, save_dir)
        elif config_select.startswith("fifosizing"):
            bench_object = bench_fifosizing(params, task_id, run_id, artifacts_dir, save_dir)
        elif config_select.startswith("metafi_fifosizing"):
            bench_object = bench_metafi_fifosizing(params, task_id, run_id, artifacts_dir, save_dir)
        elif config_select.startswith("resnet50_fifosizing"):
            bench_object = bench_resnet50_fifosizing(params, task_id, run_id, artifacts_dir, save_dir)
        else:
            print("ERROR: unknown DUT specified")

        start_time = time.time()
        try:
            bench_object.run()
            output_dict = bench_object.output_dict
            if output_dict is None:
                output_dict = {}
                log_dict["status"] = "skipped"
                print("Run skipped")
            else:
                log_dict["status"] = "ok"
                print("Run completed")
        except Exception:
            output_dict = {}
            log_dict["status"] = "failed"
            print("Run failed: " + traceback.format_exc())
            exit_code = 1

        log_dict["total_time"] = int(time.time() - start_time)
        log_dict["output"] = output_dict
        log.append(log_dict)
        # overwrite output log file every time to allow early abort
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        
        # save local artifacts of this run (e.g., detailed debug info)
        bench_object.save_local_artifacts_collection()
    print("Stopping job")
    return exit_code
    #TODO: add additional exit codes (e.g. when some verification within the run failed)?

if __name__ == "__main__":
    exit_code = main(sys.argv[1])
    sys.exit(exit_code)
