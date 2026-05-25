$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir

function Assert-FileExists {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required file: $Path"
    }
}

function Assert-Equal {
    param([string]$Label, $Actual, $Expected)
    if ($Actual -ne $Expected) {
        throw "$Label mismatch: actual=$Actual expected=$Expected"
    }
    Write-Host "[OK] $Label = $Actual"
}

function Assert-Close {
    param([string]$Label, [double]$Actual, [double]$Expected, [double]$Tolerance = 0.0001)
    if ([Math]::Abs($Actual - $Expected) -gt $Tolerance) {
        throw "$Label mismatch: actual=$Actual expected=$Expected tolerance=$Tolerance"
    }
    Write-Host ("[OK] {0} = {1}" -f $Label, $Actual)
}

function Read-RequiredCsv {
    param([string]$Path)
    Assert-FileExists $Path
    return @(Import-Csv -LiteralPath $Path)
}

$Files = @{
    PublicCsv = Join-Path $Root "data\contact_angle_dataset_v3.4_public.csv"
    PublicXlsx = Join-Path $Root "data\contact_angle_dataset_v3.4_public.xlsx"
    PublicManifest = Join-Path $Root "data\split_manifest_v3.4.csv"
    LooFeasibility = Join-Path $Root "results\source_disjoint_external_loo_sfe_feasibility_v3.4_20260519.csv"
    LooMetrics = Join-Path $Root "results\pcrnn_strict_v3.4_loo_sfe_metrics_20260519.csv"
    ForwardCheck = Join-Path $Root "results\pcrnn_strict_v3.4_loo_sfe_forward_check_20260519.csv"
    MainResult = Join-Path $Root "results\main_result_table_v3.4_LOO_corrected_20260519.csv"
    StatTests = Join-Path $Root "results\combined_pcrnn_vs_baselines_stat_tests_v3.4_LOO_corrected_20260519.csv"
}

Write-Host "== Public release reproducibility check =="
foreach ($item in $Files.GetEnumerator()) {
    Assert-FileExists $item.Value
}
Write-Host "[OK] all required files exist"

$publicRows = Read-RequiredCsv $Files.PublicCsv
$manifestRows = Read-RequiredCsv $Files.PublicManifest
Assert-Equal "public dataset rows" $publicRows.Count 616
Assert-Equal "public manifest rows" $manifestRows.Count 616

$expectedSplits = @{
    internal_train = 156
    internal_val = 33
    internal_test = 33
    balanced_holdout = 50
    hard_external = 75
    source_disjoint_external = 148
    excluded_review = 121
}
foreach ($split in $expectedSplits.Keys) {
    $count = @($manifestRows | Where-Object { $_.analysis_split -eq $split }).Count
    Assert-Equal "manifest split $split" $count $expectedSplits[$split]
}

$feasibilityRows = Read-RequiredCsv $Files.LooFeasibility
Assert-Equal "source-disjoint LOO audit rows" $feasibilityRows.Count 148
Assert-Equal "LOO-SFE feasible rows" @($feasibilityRows | Where-Object { $_.loo_sfe_feasible -eq "yes" }).Count 114
Assert-Equal "high-risk diagnostic rows" @($feasibilityRows | Where-Object { $_.loo_sfe_feasible -eq "no" }).Count 34

$mainRows = Read-RequiredCsv $Files.MainResult
$mainPcrnn = @($mainRows | Where-Object { $_.analysis_split -eq "clean_source_disjoint_external_LOO_SFE_114" -and $_.model_name -eq "PCRNN" })
$mainXgb = @($mainRows | Where-Object { $_.analysis_split -eq "clean_source_disjoint_external_LOO_SFE_114" -and $_.model_name -eq "XGBoost" })
Assert-Equal "main PCRNN clean row count" $mainPcrnn.Count 1
Assert-Equal "main XGBoost clean row count" $mainXgb.Count 1
Assert-Close "PCRNN clean external MAE" ([double]$mainPcrnn[0].MAE) 19.0381 0.00001
Assert-Close "PCRNN clean external RMSE" ([double]$mainPcrnn[0].RMSE) 24.8473 0.00001
Assert-Close "XGBoost clean external MAE" ([double]$mainXgb[0].MAE) 17.0573 0.00001

$metrics = Read-RequiredCsv $Files.LooMetrics
$before = @($metrics | Where-Object { $_.scenario -eq "original_all_liquid_loo_feasible_114" -and $_.model_name -eq "PCRNN" })
$after = @($metrics | Where-Object { $_.scenario -eq "loo_sfe_corrected_loo_feasible_114" -and $_.model_name -eq "PCRNN" })
Assert-Equal "PCRNN before-correction row count" $before.Count 1
Assert-Equal "PCRNN after-correction row count" $after.Count 1
Assert-Close "PCRNN before-correction matched MAE" ([double]$before[0].MAE) 7.9098 0.0001
Assert-Close "PCRNN after-correction matched MAE" ([double]$after[0].MAE) 19.0381 0.0001

$forward = Read-RequiredCsv $Files.ForwardCheck
Assert-Equal "forward check rows" $forward.Count 114
Assert-Equal "forward changed rows" @($forward | Where-Object { $_.prediction_changed -eq "yes" }).Count 107
$absShifts = @($forward | ForEach-Object { [Math]::Abs([double]$_.delta_y_pred_loo_minus_original) })
$meanShift = ($absShifts | Measure-Object -Average).Average
$maxShift = ($absShifts | Measure-Object -Maximum).Maximum
Assert-Close "mean absolute prediction shift" $meanShift 12.0957 0.0001
Assert-Close "max absolute prediction shift" $maxShift 55.7162 0.0001

$stats = Read-RequiredCsv $Files.StatTests
$pcrnnVsXgb = @($stats | Where-Object { $_.analysis_split -eq "clean_source_disjoint_external_LOO_SFE_114" -and $_.model_a -eq "PCRNN" -and $_.model_b -eq "XGBoost" })
Assert-Equal "PCRNN vs XGBoost clean stat rows" $pcrnnVsXgb.Count 1
Assert-Close "PCRNN-XGBoost delta MAE" ([double]$pcrnnVsXgb[0].delta_MAE_a_minus_b) 1.980825 0.000001
Assert-Close "PCRNN-XGBoost p value" ([double]$pcrnnVsXgb[0].p_value) 0.0492 0.000001

Write-Host ""
Write-Host "PUBLIC RELEASE REPRODUCIBILITY CHECK PASSED"
