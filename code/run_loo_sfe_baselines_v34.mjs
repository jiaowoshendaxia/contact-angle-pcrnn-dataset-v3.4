import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import {
  enrichRowsForMl,
  fitFeatureEncoder,
  normalizeDataRow,
  owensWendt,
  trainMlp,
  trainRandomForest,
  trainXGBoostStyle,
  writeCsv,
} from "./run_all_baselines.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(scriptDir, "..");
const datasetTag = "v3.4";
const defaultInput = path.join(rootDir, "data", "contact_angle_dataset_v3.4_public.xlsx");
const defaultManifest = path.join(rootDir, "data", "split_manifest_v3.4.csv");
const defaultFeasibility = path.join(rootDir, "results", "source_disjoint_external_loo_sfe_feasibility_v3.4_20260519.csv");
const defaultOut = path.join(rootDir, "results", "_loo_sfe_baseline_predictions_v3.4_20260519.csv");
const seeds = [7, 11, 19, 23, 31, 42, 53, 67, 79, 97];
const evalSplits = new Set(["internal_val", "internal_test", "balanced_holdout", "hard_external", "source_disjoint_external"]);
const modelNames = ["Owens-Wendt", "Random Forest", "XGBoost", "Ordinary MLP"];
const predictionColumns = [
  "record_id",
  "analysis_split",
  "loo_sfe_group",
  "sfe_mode",
  "source_group_id",
  "duplicate_policy",
  "model_name",
  "y_true",
  "y_pred",
  "error",
  "abs_error",
  "dataset_version",
  "prediction_aggregation",
  "n_seeds",
];

function parseArgs(argv) {
  const args = {
    input: defaultInput,
    manifest: defaultManifest,
    feasibility: defaultFeasibility,
    out: defaultOut,
    sheet: "Raw_Data_Public",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--input" && value) {
      args.input = path.resolve(value);
      i += 1;
    } else if (key === "--manifest" && value) {
      args.manifest = path.resolve(value);
      i += 1;
    } else if (key === "--feasibility" && value) {
      args.feasibility = path.resolve(value);
      i += 1;
    } else if (key === "--out" && value) {
      args.out = path.resolve(value);
      i += 1;
    } else if (key === "--sheet" && value) {
      args.sheet = value;
      i += 1;
    }
  }
  return args;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r") {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const headers = rows[0].map((header) => String(header ?? "").trim());
  return rows
    .slice(1)
    .filter((items) => items.some((item) => item !== ""))
    .map((items) => Object.fromEntries(headers.map((header, i) => [header, items[i] ?? ""])));
}

function rowObject(headers, values) {
  return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function asNumber(value) {
  const out = Number(value);
  return Number.isFinite(out) ? out : null;
}

function manifestByRecord(rows) {
  return new Map(rows.map((row) => [String(row.record_id ?? "").trim(), row]));
}

function feasibilityByRecord(rows) {
  return new Map(rows.map((row) => [String(row.record_id ?? "").trim(), row]));
}

function applyLooSfe(raw, auditRow) {
  const out = { ...raw };
  if (String(auditRow?.loo_sfe_feasible ?? "").trim().toLowerCase() !== "yes") return out;
  out.solid_total_surface_energy_mJ_m2 = auditRow.loo_sfe_total_mJ_m2;
  out.solid_dispersion_mJ_m2 = auditRow.loo_sfe_dispersion_mJ_m2;
  out.solid_polar_mJ_m2 = auditRow.loo_sfe_polar_mJ_m2;
  return out;
}

function predictionRow(row, modelName, rawPrediction, aggregation, nSeeds) {
  const yPred = Math.max(0, Math.min(180, Number(rawPrediction)));
  const error = yPred - row.y_true;
  return {
    record_id: row.record_id,
    analysis_split: row.analysis_split,
    loo_sfe_group: row.loo_sfe_group,
    sfe_mode: row.sfe_mode,
    source_group_id: row.source_group_id,
    duplicate_policy: row.duplicate_policy,
    model_name: modelName,
    y_true: row.y_true,
    y_pred: yPred,
    error,
    abs_error: Math.abs(error),
    dataset_version: datasetTag,
    prediction_aggregation: aggregation,
    n_seeds: nSeeds,
  };
}

async function loadRows(args) {
  const manifest = manifestByRecord(parseCsv(await fs.readFile(args.manifest, "utf8")));
  const feasibility = feasibilityByRecord(parseCsv(await fs.readFile(args.feasibility, "utf8")));
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(args.input));
  const sheet = workbook.worksheets.getItem(args.sheet);
  const values = sheet.getRange("A1:BH1200").values;
  const headers = values[0].map((header) => String(header ?? "").trim());
  const rawRows = values
    .slice(1)
    .map((row) => rowObject(headers, row))
    .filter((row) => String(row.record_id ?? "").trim() !== "");
  const rows = [];
  const skipped = [];
  for (const raw0 of rawRows) {
    const recordId = String(raw0.record_id ?? "").trim();
    const manifestRow = manifest.get(recordId);
    if (!manifestRow || manifestRow.analysis_split === "excluded_review") continue;
    const auditRow = feasibility.get(recordId);
    const raw = String(manifestRow.analysis_split).trim() === "source_disjoint_external" ? applyLooSfe(raw0, auditRow) : raw0;
    const normalized = normalizeDataRow(raw);
    if (normalized.skipReason) {
      skipped.push({ record_id: recordId, reason: normalized.skipReason });
      continue;
    }
    const feasible = String(auditRow?.loo_sfe_feasible ?? "").trim().toLowerCase() === "yes";
    rows.push({
      ...normalized,
      dataset_version: datasetTag,
      analysis_split: String(manifestRow.analysis_split ?? "").trim(),
      source_group_id: String(manifestRow.source_group_id ?? "").trim(),
      duplicate_policy: String(manifestRow.duplicate_policy ?? "").trim(),
      loo_sfe_group: String(manifestRow.analysis_split).trim() === "source_disjoint_external" ? (feasible ? "loo_feasible" : "high_risk") : "not_source_disjoint",
      sfe_mode: String(manifestRow.analysis_split).trim() === "source_disjoint_external" && feasible ? "loo_sfe_corrected" : "all_liquid_original",
    });
  }
  if (skipped.length) throw new Error(`Skipped rows after LOO-SFE baseline normalization: ${JSON.stringify(skipped.slice(0, 5))}`);
  return rows;
}

function physicalPredictions(rows) {
  return rows
    .filter((row) => row.analysis_split !== "internal_train" && evalSplits.has(row.analysis_split))
    .map((row) => predictionRow(row, "Owens-Wendt", owensWendt(row).thetaDeg, "deterministic", 1));
}

function mlPredictions(rows) {
  const mlRows = enrichRowsForMl(rows);
  const trainRows = mlRows.filter((row) => row.analysis_split === "internal_train");
  const valRows = mlRows.filter((row) => row.analysis_split === "internal_val");
  const encoder = fitFeatureEncoder(trainRows);
  const xByRecord = new Map(mlRows.map((row) => [row.record_id, encoder.transform(row)]));
  const xTrain = trainRows.map((row) => xByRecord.get(row.record_id));
  const yTrain = trainRows.map((row) => row.y_true);
  const xVal = valRows.map((row) => xByRecord.get(row.record_id));
  const yVal = valRows.map((row) => row.y_true);
  const evalRows = mlRows.filter((row) => row.analysis_split !== "internal_train" && evalSplits.has(row.analysis_split));
  const byModelRecord = new Map();
  for (const seed of seeds) {
    const models = [
      ["Random Forest", trainRandomForest(xTrain, yTrain, seed + 11)],
      ["XGBoost", trainXGBoostStyle(xTrain, yTrain, seed + 23)],
      ["Ordinary MLP", trainMlp(xTrain, yTrain, xVal, yVal, seed + 37)],
    ];
    for (const [modelName, model] of models) {
      for (const row of evalRows) {
        const key = `${modelName}||${row.record_id}`;
        if (!byModelRecord.has(key)) byModelRecord.set(key, []);
        byModelRecord.get(key).push(model.predict(xByRecord.get(row.record_id)));
      }
    }
  }
  const out = [];
  for (const row of evalRows) {
    for (const modelName of modelNames.filter((name) => name !== "Owens-Wendt")) {
      out.push(predictionRow(row, modelName, mean(byModelRecord.get(`${modelName}||${row.record_id}`)), "mean_of_repeated_seeds", seeds.length));
    }
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const rows = await loadRows(args);
  const predictions = [...physicalPredictions(rows), ...mlPredictions(rows)]
    .filter((row) => row.analysis_split === "source_disjoint_external")
    .sort((a, b) => a.record_id.localeCompare(b.record_id) || modelNames.indexOf(a.model_name) - modelNames.indexOf(b.model_name));
  await fs.mkdir(path.dirname(args.out), { recursive: true });
  await writeCsv(args.out, predictionColumns, predictions);
  console.log(JSON.stringify({
    out: args.out,
    predictions: predictions.length,
    feasible: new Set(predictions.filter((row) => row.loo_sfe_group === "loo_feasible").map((row) => row.record_id)).size,
    high_risk: new Set(predictions.filter((row) => row.loo_sfe_group === "high_risk").map((row) => row.record_id)).size,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
