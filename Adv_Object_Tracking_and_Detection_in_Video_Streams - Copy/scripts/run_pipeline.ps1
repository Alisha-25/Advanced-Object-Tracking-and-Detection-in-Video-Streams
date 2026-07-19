param(
    [string]$Config = "config.yaml",
    [switch]$Extract,
    [switch]$FullSequence
)

# Run this from PowerShell in the project folder after editing config.yaml.
# It deliberately does not run extraction unless -Extract is supplied, because
# the archive expands to roughly 5.55 GiB.
$ErrorActionPreference = "Stop"

if ($Extract) {
    # dataset_root points to MOT17; prepare_dataset expects its parent folder.
    $extractPaths = @(& python -c "import pathlib,sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); r=pathlib.Path(c['paths']['dataset_root']); print(c['paths']['dataset_zip']); print(r.parent)" $Config)
    python -m src.prepare_dataset --zip $extractPaths[0] --destination $extractPaths[1]
}

python scripts/check_setup.py --config $Config
python -m src.train_detector --config $Config
python -m src.evaluate_detector --config $Config

if ($FullSequence) {
    python -m src.run_tracking --config $Config --max-frames 0
} else {
    python -m src.run_tracking --config $Config
}

# Copy the displayed tracking_results.txt path into the next command.  The
# default validation sequence is MOT17-05-FRCNN.
Write-Host "Run tracking evaluation using:"
Write-Host "python -m src.evaluate_tracking --config $Config --predictions .\results\tracking_MOT17-05-FRCNN\tracking_results.txt"
