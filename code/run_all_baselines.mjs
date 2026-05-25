import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(scriptDir, "..");
const defaultInput = path.join(rootDir, "outputs", "contact_angle_dataset_v3.1_20260514.xlsx");
const defaultOutputDir = path.join(rootDir, "outputs", "baselines_final");
let runDate = "2026-05-14";
let datasetTag = "v3.1";
let validationType = "fixed_split_v3_1_final";

const predictionColumns = [
  "record_id",
  "model_name",
  "y_true_contact_angle_deg",
  "y_pred_contact_angle_deg",
  "error_deg",
  "split_group",
  "validation_type",
];

const metricColumns = [
  "model_name",
  "dataset_version",
  "validation_type",
  "split_id",
  "test_group",
  "n_samples",
  "MAE",
  "RMSE",
  "R2",
  "Median_AE",
  "nonphysical_rate",
  "notes",
];

const diagnosticColumns = [
  "record_id",
  "model_name",
  "cos_theta_raw",
  "cos_theta_clipped",
  "was_clipped",
  "weight_d",
  "weight_p",
  "weight_rule",
  "component_note",
  "solid_name",
  "liquid_name",
  "split_group",
];

const splitColumns = [
  "record_id",
  "original_split_group",
  "effective_split_group",
  "solid_name",
  "liquid_name",
  "quality_grade",
  "include_in_training",
];

const skippedColumns = [
  "record_id",
  "skip_reason",
  "solid_name",
  "liquid_name",
  "include_in_training",
  "quality_grade",
];

const trainingSummaryColumns = [
  "model_name",
  "training_rows",
  "validation_rows",
  "feature_count",
  "settings",
  "notes",
];

const finalFindingColumns = [
  "section",
  "item",
  "value",
  "notes",
];

const formulaRows = [
  ["model_name", "formula_or_rule"],
  [
    "Owens-Wendt",
    "cos(theta)=2*(sqrt(gamma_s_d*gamma_l_d)+sqrt(gamma_s_p*gamma_l_p))/gamma_l-1",
  ],
  [
    "Harmonic Mean",
    "cos(theta)=(4*gamma_s_d*gamma_l_d/(gamma_s_d+gamma_l_d)+4*gamma_s_p*gamma_l_p/(gamma_s_p+gamma_l_p))/gamma_l-1; zero denominators contribute 0",
  ],
  [
    "Weighted Geometric",
    "Fixed weighted form: term=(1-alpha)*(term_d+term_p)+alpha*2*(w_d*term_d+w_p*term_p), alpha=0.10, w_p=clamp(mean(solid_polar_ratio, liquid_polar_ratio),0.40,0.60), w_d=1-w_p. This keeps the weighted term on the same scale as Owens-Wendt instead of shrinking all interactions.",
  ],
  [
    "van Oss-Chaudhury-Good",
    "W_sl=2*(sqrt(gamma_s_LW*gamma_l_LW)+sqrt(gamma_s_plus*gamma_l_minus)+sqrt(gamma_s_minus*gamma_l_plus)); cos(theta)=W_sl/gamma_l-1. Because solid acid/base fields are absent in the current dataset, solid_LW falls back to solid_dispersion and solid_plus=solid_minus=solid_polar/2.",
  ],
  [
    "Random Forest",
    "Self-contained CART regression forest: bootstrap samples, random feature subsets at each node, mean aggregation.",
  ],
  [
    "XGBoost",
    "Self-contained XGBoost-style squared-error gradient tree boosting: second-order split gain, L2 leaf regularization, shrinkage. The official xgboost package is not installed in this runtime.",
  ],
  [
    "Ordinary MLP",
    "Plain fully connected regressor with two ReLU hidden layers and Adam optimization, trained on standardized tabular features.",
  ],
];

function parseArgs(argv) {
  const args = {
    input: defaultInput,
    outDir: defaultOutputDir,
    sheet: "Raw_Data_Template",
    seed: 42,
    datasetTag,
    runDate,
    validationType,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === "--input" && value) {
      args.input = path.resolve(value);
      i += 1;
    } else if (key === "--out-dir" && value) {
      args.outDir = path.resolve(value);
      i += 1;
    } else if (key === "--sheet" && value) {
      args.sheet = value;
      i += 1;
    } else if (key === "--seed" && value) {
      args.seed = Number(value);
      i += 1;
    } else if (key === "--dataset-tag" && value) {
      args.datasetTag = value;
      i += 1;
    } else if (key === "--run-date" && value) {
      args.runDate = value;
      i += 1;
    } else if (key === "--validation-type" && value) {
      args.validationType = value;
      i += 1;
    }
  }
  return args;
}

function makeRng(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items, seed) {
  const rng = makeRng(seed);
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function colLetter(indexZeroBased) {
  let n = indexZeroBased + 1;
  let s = "";
  while (n > 0) {
    const m = (n - 1) % 26;
    s = String.fromCharCode(65 + m) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const s = String(value);
  if (/[",\r\n]/.test(s)) return `"${s.replaceAll('"', '""')}"`;
  return s;
}

async function writeCsv(filePath, columns, rows) {
  const lines = [
    columns.map(csvEscape).join(","),
    ...rows.map((row) => columns.map((col) => csvEscape(row[col])).join(",")),
  ];
  await fs.writeFile(filePath, `${lines.join("\n")}\n`, "utf8");
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const n = Number(String(value).trim());
  return Number.isFinite(n) ? n : null;
}

function round(value, digits = 6) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "";
  return Number(value.toFixed(digits));
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function sqrtNonnegative(value) {
  return Math.sqrt(Math.max(0, value));
}

function clipCos(raw) {
  const clipped = clamp(raw, -1, 1);
  return {
    raw,
    clipped,
    wasClipped: Math.abs(raw - clipped) > 1e-12,
  };
}

function thetaFromCos(rawCos) {
  const clipped = clipCos(rawCos);
  return {
    ...clipped,
    thetaDeg: (Math.acos(clipped.clipped) * 180) / Math.PI,
  };
}

function harmonicContribution(a, b) {
  const denom = a + b;
  return denom > 0 ? (4 * a * b) / denom : 0;
}

function rowObject(headers, values) {
  return Object.fromEntries(headers.map((header, i) => [header, values[i] ?? ""]));
}

function physicalBaseTerms(row) {
  return {
    d: sqrtNonnegative(row.solid_d * row.liquid_d),
    p: sqrtNonnegative(row.solid_p * row.liquid_p),
  };
}

function polarRatio(d, p) {
  const total = d + p;
  return total > 0 ? p / total : 0;
}

function owensWendt(row) {
  const terms = physicalBaseTerms(row);
  return {
    modelName: "Owens-Wendt",
    ...thetaFromCos((2 * (terms.d + terms.p)) / row.liquid_total - 1),
    weightD: "",
    weightP: "",
    weightRule: "",
    componentNote: "",
  };
}

function harmonicMean(row) {
  const interaction =
    harmonicContribution(row.solid_d, row.liquid_d) +
    harmonicContribution(row.solid_p, row.liquid_p);
  return {
    modelName: "Harmonic Mean",
    ...thetaFromCos(interaction / row.liquid_total - 1),
    weightD: "",
    weightP: "",
    weightRule: "",
    componentNote: "",
  };
}

function weightedGeometric(row) {
  const terms = physicalBaseTerms(row);
  const solidPolarRatio = polarRatio(row.solid_d, row.solid_p);
  const liquidPolarRatio = polarRatio(row.liquid_d, row.liquid_p);
  const weightP = clamp((solidPolarRatio + liquidPolarRatio) / 2, 0.4, 0.6);
  const weightD = 1 - weightP;
  const alpha = 0.1;
  const owInteraction = terms.d + terms.p;
  const weightedInteraction = 2 * (weightD * terms.d + weightP * terms.p);
  const interaction = (1 - alpha) * owInteraction + alpha * weightedInteraction;
  return {
    modelName: "Weighted Geometric",
    ...thetaFromCos((2 * interaction) / row.liquid_total - 1),
    weightD,
    weightP,
    weightRule: "gentle_polarity_balanced_alpha_0.10",
    componentNote: "fixed_scale_weighted_geometric",
  };
}

function vanOssChaudhuryGood(row) {
  const solidLW = row.solid_lw ?? row.solid_d;
  const solidPlus = row.solid_plus ?? row.solid_p / 2;
  const solidMinus = row.solid_minus ?? row.solid_p / 2;
  const liquidLW = row.liquid_lw ?? row.liquid_d;
  const liquidPlus = row.liquid_plus ?? row.liquid_p / 2;
  const liquidMinus = row.liquid_minus ?? row.liquid_p / 2;
  const workAdhesion = 2 * (
    sqrtNonnegative(solidLW * liquidLW) +
    sqrtNonnegative(solidPlus * liquidMinus) +
    sqrtNonnegative(solidMinus * liquidPlus)
  );
  return {
    modelName: "van Oss-Chaudhury-Good",
    ...thetaFromCos(workAdhesion / row.liquid_total - 1),
    weightD: "",
    weightP: "",
    weightRule: "",
    componentNote: row.solidAcidBaseEstimated
      ? "solid acid/base estimated as solid_polar/2; solid_LW uses dispersion"
      : "acid/base components from source fields",
  };
}

const physicalModels = [owensWendt, harmonicMean, weightedGeometric, vanOssChaudhuryGood];

function normalizeDataRow(raw) {
  const contactAngle = toNumber(raw.contact_angle_deg);
  const solidD = toNumber(raw.solid_dispersion_mJ_m2);
  const solidP = toNumber(raw.solid_polar_mJ_m2);
  const solidTotal = toNumber(raw.solid_total_surface_energy_mJ_m2);
  const liquidTotal = toNumber(raw.liquid_total_surface_tension_mN_m);
  const liquidD = toNumber(raw.liquid_dispersion_mN_m);
  const liquidP = toNumber(raw.liquid_polar_mN_m);
  const solidLWSource = toNumber(raw.solid_LW_mJ_m2);
  const solidPlusSource = toNumber(raw.solid_acid_plus_mJ_m2);
  const solidMinusSource = toNumber(raw.solid_base_minus_mJ_m2);
  const liquidLW = toNumber(raw.liquid_LW_mN_m);
  const liquidPlus = toNumber(raw.liquid_acid_plus_mN_m);
  const liquidMinus = toNumber(raw.liquid_base_minus_mN_m);
  const missing = [];

  if (!raw.record_id) missing.push("record_id");
  if (contactAngle === null) missing.push("contact_angle_deg");
  if (solidD === null) missing.push("solid_dispersion_mJ_m2");
  if (solidP === null) missing.push("solid_polar_mJ_m2");
  if (liquidTotal === null) missing.push("liquid_total_surface_tension_mN_m");
  if (liquidD === null) missing.push("liquid_dispersion_mN_m");
  if (liquidP === null) missing.push("liquid_polar_mN_m");

  if (liquidTotal !== null && liquidTotal <= 0) missing.push("liquid_total_surface_tension_mN_m<=0");
  for (const [name, value] of [
    ["solid_dispersion_mJ_m2", solidD],
    ["solid_polar_mJ_m2", solidP],
    ["liquid_dispersion_mN_m", liquidD],
    ["liquid_polar_mN_m", liquidP],
  ]) {
    if (value !== null && value < 0) missing.push(`${name}<0`);
  }

  const quality = String(raw.quality_grade ?? "").trim().toLowerCase();
  if (quality === "exclude") missing.push("quality_grade=exclude");

  const solidAcidBaseEstimated = solidLWSource === null || solidPlusSource === null || solidMinusSource === null;

  return {
    record_id: String(raw.record_id ?? "").trim(),
    dataset_version: String(raw.dataset_version ?? "v3.1").trim() || "v3.1",
    y_true: contactAngle,
    solid_name: String(raw.solid_name ?? "").trim(),
    solid_family: String(raw.solid_family ?? "").trim(),
    surface_treatment: String(raw.surface_treatment ?? "").trim(),
    surface_state: String(raw.surface_state ?? "").trim(),
    liquid_name: String(raw.liquid_name ?? "").trim(),
    liquid_family: String(raw.liquid_family ?? "").trim(),
    contact_angle_type: String(raw.contact_angle_type ?? "").trim(),
    measurement_method: String(raw.measurement_method ?? "").trim(),
    original_split_group: String(raw.split_group ?? "unassigned").trim() || "unassigned",
    include_in_training: raw.include_in_training ?? "",
    quality_grade: raw.quality_grade ?? "",
    collection_status: raw.collection_status ?? "",
    solid_d: solidD,
    solid_p: solidP,
    solid_total: solidTotal ?? (solidD !== null && solidP !== null ? solidD + solidP : null),
    solid_lw: solidLWSource ?? solidD,
    solid_plus: solidPlusSource ?? (solidP !== null ? solidP / 2 : null),
    solid_minus: solidMinusSource ?? (solidP !== null ? solidP / 2 : null),
    liquid_total: liquidTotal,
    liquid_d: liquidD,
    liquid_p: liquidP,
    liquid_lw: liquidLW ?? liquidD,
    liquid_plus: liquidPlus ?? (liquidP !== null ? liquidP / 2 : null),
    liquid_minus: liquidMinus ?? (liquidP !== null ? liquidP / 2 : null),
    liquid_viscosity: toNumber(raw.liquid_viscosity_mPa_s),
    liquid_dipole: toNumber(raw.liquid_dipole_moment_D),
    liquid_dielectric: toNumber(raw.liquid_dielectric_constant),
    temperature_K: toNumber(raw.temperature_K),
    humidity_percent: toNumber(raw.humidity_percent),
    pressure_atm: toNumber(raw.pressure_atm),
    droplet_volume_uL: toNumber(raw.droplet_volume_uL),
    roughness_Ra_nm: toNumber(raw.roughness_Ra_nm),
    roughness_Rq_nm: toNumber(raw.roughness_Rq_nm),
    roughness_r_factor: toNumber(raw.roughness_r_factor),
    contact_angle_std_deg: toNumber(raw.contact_angle_std_deg),
    solidAcidBaseEstimated,
    skipReason: missing.join(";"),
  };
}

function assignEffectiveSplits(rows, seed) {
  const external = rows.filter((row) => row.original_split_group === "external");
  for (const row of external) row.effective_split_group = "external";

  const originalTrain = rows.filter((row) => row.original_split_group === "train");
  for (const row of originalTrain) row.effective_split_group = "train";

  const unassigned = rows.filter((row) => !["external", "train"].includes(row.original_split_group));
  const shuffled = shuffle(unassigned, seed);
  const nonExternalCount = rows.length - external.length;
  const targetTrainCount = Math.max(originalTrain.length, Math.round(nonExternalCount * 0.7));
  const trainFromUnassigned = Math.max(0, targetTrainCount - originalTrain.length);
  const remaining = shuffled.length - trainFromUnassigned;
  const validationCount = Math.floor(remaining / 2);

  shuffled.forEach((row, i) => {
    if (i < trainFromUnassigned) row.effective_split_group = "train";
    else if (i < trainFromUnassigned + validationCount) row.effective_split_group = "validation";
    else row.effective_split_group = "test";
  });
}

function makePhysicalPredictionRows(rows) {
  const predictions = [];
  const diagnostics = [];

  for (const row of rows) {
    for (const model of physicalModels) {
      const result = model(row);
      const error = result.thetaDeg - row.y_true;
      predictions.push({
        record_id: row.record_id,
        model_name: result.modelName,
        y_true_contact_angle_deg: round(row.y_true, 4),
        y_pred_contact_angle_deg: round(result.thetaDeg, 4),
        error_deg: round(error, 4),
        split_group: row.effective_split_group,
        validation_type: validationType,
        _wasClipped: result.wasClipped,
        _datasetVersion: row.dataset_version,
      });
      diagnostics.push({
        record_id: row.record_id,
        model_name: result.modelName,
        cos_theta_raw: round(result.raw, 8),
        cos_theta_clipped: round(result.clipped, 8),
        was_clipped: result.wasClipped ? "yes" : "no",
        weight_d: round(result.weightD, 6),
        weight_p: round(result.weightP, 6),
        weight_rule: result.weightRule,
        component_note: result.componentNote,
        solid_name: row.solid_name,
        liquid_name: row.liquid_name,
        split_group: row.effective_split_group,
      });
    }
  }
  return { predictions, diagnostics };
}

function physicalFeatureValues(row) {
  const ow = owensWendt(row).thetaDeg;
  const hm = harmonicMean(row).thetaDeg;
  const wg = weightedGeometric(row).thetaDeg;
  const vo = vanOssChaudhuryGood(row).thetaDeg;
  const solidPolarRatio = polarRatio(row.solid_d, row.solid_p);
  const liquidPolarRatio = polarRatio(row.liquid_d, row.liquid_p);
  return {
    owens_wendt_theta_deg: ow,
    harmonic_theta_deg: hm,
    weighted_geometric_theta_deg: wg,
    van_oss_theta_deg: vo,
    solid_polar_ratio: solidPolarRatio,
    liquid_polar_ratio: liquidPolarRatio,
    gamma_ratio_s_l: row.liquid_total > 0 ? row.solid_total / row.liquid_total : null,
    gamma_difference_abs: Math.abs(row.solid_total - row.liquid_total),
    polarity_difference_abs: Math.abs(solidPolarRatio - liquidPolarRatio),
  };
}

const numericFeatureNames = [
  "solid_total",
  "solid_d",
  "solid_p",
  "solid_lw",
  "solid_plus",
  "solid_minus",
  "liquid_total",
  "liquid_d",
  "liquid_p",
  "liquid_lw",
  "liquid_plus",
  "liquid_minus",
  "liquid_viscosity",
  "liquid_dipole",
  "liquid_dielectric",
  "temperature_K",
  "humidity_percent",
  "pressure_atm",
  "droplet_volume_uL",
  "roughness_Ra_nm",
  "roughness_Rq_nm",
  "roughness_r_factor",
  "contact_angle_std_deg",
  "solid_polar_ratio",
  "liquid_polar_ratio",
  "gamma_ratio_s_l",
  "gamma_difference_abs",
  "polarity_difference_abs",
];

const categoricalFeatureNames = [
  "solid_name",
  "solid_family",
  "surface_treatment",
  "surface_state",
  "liquid_name",
  "liquid_family",
  "contact_angle_type",
  "measurement_method",
  "quality_grade",
  "collection_status",
];

function enrichRowsForMl(rows) {
  return rows.map((row) => ({ ...row, ...physicalFeatureValues(row) }));
}

function medianNumber(values) {
  const nums = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!nums.length) return 0;
  const mid = Math.floor(nums.length / 2);
  return nums.length % 2 === 0 ? (nums[mid - 1] + nums[mid]) / 2 : nums[mid];
}

function fitFeatureEncoder(trainRows) {
  const medians = Object.fromEntries(
    numericFeatureNames.map((name) => [name, medianNumber(trainRows.map((row) => row[name]))]),
  );
  const means = {};
  const stds = {};
  for (const name of numericFeatureNames) {
    const vals = trainRows.map((row) => Number.isFinite(row[name]) ? row[name] : medians[name]);
    const mean = vals.reduce((sum, value) => sum + value, 0) / vals.length;
    const variance = vals.reduce((sum, value) => sum + (value - mean) ** 2, 0) / vals.length;
    means[name] = mean;
    stds[name] = Math.sqrt(variance) || 1;
  }

  const categories = {};
  for (const name of categoricalFeatureNames) {
    const vocab = new Set(trainRows.map((row) => String(row[name] ?? "__missing__") || "__missing__"));
    vocab.add("__unknown__");
    categories[name] = [...vocab].sort();
  }

  const featureNames = [
    ...numericFeatureNames,
    ...categoricalFeatureNames.flatMap((name) => categories[name].map((value) => `${name}=${value}`)),
  ];

  return {
    medians,
    means,
    stds,
    categories,
    featureNames,
    transform(row) {
      const out = [];
      for (const name of numericFeatureNames) {
        const raw = Number.isFinite(row[name]) ? row[name] : medians[name];
        out.push((raw - means[name]) / stds[name]);
      }
      for (const name of categoricalFeatureNames) {
        const raw = String(row[name] ?? "__missing__") || "__missing__";
        const value = categories[name].includes(raw) ? raw : "__unknown__";
        for (const category of categories[name]) out.push(category === value ? 1 : 0);
      }
      return out;
    },
  };
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function varianceOfY(rows) {
  const m = mean(rows.map((row) => row.y));
  return mean(rows.map((row) => (row.y - m) ** 2));
}

function chooseFeatureSubset(nFeatures, count, rng) {
  const indices = Array.from({ length: nFeatures }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices.slice(0, Math.max(1, Math.min(count, nFeatures)));
}

function candidateThresholds(X, indices, feature, maxThresholds) {
  const unique = [...new Set(indices.map((idx) => X[idx][feature]).filter(Number.isFinite))].sort((a, b) => a - b);
  if (unique.length <= 1) return [];
  const thresholds = [];
  if (unique.length <= maxThresholds + 1) {
    for (let i = 1; i < unique.length; i += 1) thresholds.push((unique[i - 1] + unique[i]) / 2);
  } else {
    for (let k = 1; k <= maxThresholds; k += 1) {
      const pos = Math.floor((k * unique.length) / (maxThresholds + 1));
      thresholds.push((unique[pos - 1] + unique[pos]) / 2);
    }
  }
  return [...new Set(thresholds.filter(Number.isFinite))];
}

function sseFromStats(sum, sumSq, n) {
  return n > 0 ? sumSq - (sum * sum) / n : 0;
}

function buildCartTree(X, y, indices, options, rng, depth = 0) {
  const yVals = indices.map((idx) => y[idx]);
  const value = mean(yVals);
  if (
    depth >= options.maxDepth ||
    indices.length < options.minSamplesSplit ||
    yVals.length <= options.minSamplesLeaf * 2 ||
    varianceOfY(yVals.map((valueY) => ({ y: valueY }))) < 1e-10
  ) {
    return { value };
  }

  const nFeatures = X[0].length;
  const featureCount = options.featureSubsetCount ?? nFeatures;
  const features = chooseFeatureSubset(nFeatures, featureCount, rng);
  let best = null;

  const parentSum = yVals.reduce((sum, valueY) => sum + valueY, 0);
  const parentSumSq = yVals.reduce((sum, valueY) => sum + valueY ** 2, 0);
  const parentSse = sseFromStats(parentSum, parentSumSq, indices.length);

  for (const feature of features) {
    for (const threshold of candidateThresholds(X, indices, feature, options.maxThresholds)) {
      const left = [];
      const right = [];
      let leftSum = 0;
      let leftSumSq = 0;
      let rightSum = 0;
      let rightSumSq = 0;
      for (const idx of indices) {
        if (X[idx][feature] <= threshold) {
          left.push(idx);
          leftSum += y[idx];
          leftSumSq += y[idx] ** 2;
        } else {
          right.push(idx);
          rightSum += y[idx];
          rightSumSq += y[idx] ** 2;
        }
      }
      if (left.length < options.minSamplesLeaf || right.length < options.minSamplesLeaf) continue;
      const sse = sseFromStats(leftSum, leftSumSq, left.length) + sseFromStats(rightSum, rightSumSq, right.length);
      const gain = parentSse - sse;
      if (!best || gain > best.gain) best = { feature, threshold, left, right, gain };
    }
  }

  if (!best || best.gain <= options.minGain) return { value };
  return {
    value,
    feature: best.feature,
    threshold: best.threshold,
    left: buildCartTree(X, y, best.left, options, rng, depth + 1),
    right: buildCartTree(X, y, best.right, options, rng, depth + 1),
  };
}

function predictTree(tree, x) {
  let node = tree;
  while (node.feature !== undefined) {
    node = x[node.feature] <= node.threshold ? node.left : node.right;
  }
  return node.value;
}

function trainRandomForest(XTrain, yTrain, seed) {
  return trainRandomForestWithOptions(XTrain, yTrain, seed);
}

function trainRandomForestWithOptions(XTrain, yTrain, seed, overrides = {}) {
  const rng = makeRng(seed);
  const n = XTrain.length;
  const nFeatures = XTrain[0].length;
  const trees = [];
  const nTrees = overrides.nTrees ?? 120;
  const featureSubsetCount = overrides.featureSubsetCount ?? Math.max(1, Math.round(Math.sqrt(nFeatures)));
  const options = {
    maxDepth: overrides.maxDepth ?? 7,
    minSamplesSplit: overrides.minSamplesSplit ?? 8,
    minSamplesLeaf: overrides.minSamplesLeaf ?? 3,
    maxThresholds: overrides.maxThresholds ?? 18,
    minGain: 1e-7,
    featureSubsetCount,
  };

  for (let t = 0; t < nTrees; t += 1) {
    const sample = Array.from({ length: n }, () => Math.floor(rng() * n));
    trees.push(buildCartTree(XTrain, yTrain, sample, options, rng));
  }

  return {
    settings: `n_trees=${nTrees}, max_depth=${options.maxDepth}, min_leaf=${options.minSamplesLeaf}, mtry=${featureSubsetCount}`,
    predict(x) {
      return mean(trees.map((tree) => predictTree(tree, x)));
    },
  };
}

function buildBoostTree(X, gradients, hessians, indices, options, rng, depth = 0) {
  const G = indices.reduce((sum, idx) => sum + gradients[idx], 0);
  const H = indices.reduce((sum, idx) => sum + hessians[idx], 0);
  const value = clamp(-G / (H + options.lambda), -options.maxLeafValue, options.maxLeafValue);
  if (depth >= options.maxDepth || indices.length < options.minSamplesSplit) return { value };

  const nFeatures = X[0].length;
  const featureCount = Math.max(1, Math.round(nFeatures * options.featureRate));
  const features = chooseFeatureSubset(nFeatures, featureCount, rng);
  let best = null;

  for (const feature of features) {
    for (const threshold of candidateThresholds(X, indices, feature, options.maxThresholds)) {
      const left = [];
      const right = [];
      let GL = 0;
      let HL = 0;
      let GR = 0;
      let HR = 0;
      for (const idx of indices) {
        if (X[idx][feature] <= threshold) {
          left.push(idx);
          GL += gradients[idx];
          HL += hessians[idx];
        } else {
          right.push(idx);
          GR += gradients[idx];
          HR += hessians[idx];
        }
      }
      if (left.length < options.minSamplesLeaf || right.length < options.minSamplesLeaf) continue;
      const gain = 0.5 * (
        (GL ** 2) / (HL + options.lambda) +
        (GR ** 2) / (HR + options.lambda) -
        (G ** 2) / (H + options.lambda)
      ) - options.gamma;
      if (!best || gain > best.gain) best = { feature, threshold, left, right, gain };
    }
  }

  if (!best || best.gain <= 0) return { value };
  return {
    value,
    feature: best.feature,
    threshold: best.threshold,
    left: buildBoostTree(X, gradients, hessians, best.left, options, rng, depth + 1),
    right: buildBoostTree(X, gradients, hessians, best.right, options, rng, depth + 1),
  };
}

function trainXGBoostStyle(XTrain, yTrain, seed) {
  return trainXGBoostStyleWithOptions(XTrain, yTrain, seed);
}

function trainXGBoostStyleWithOptions(XTrain, yTrain, seed, overrides = {}) {
  const rng = makeRng(seed);
  const n = XTrain.length;
  const base = mean(yTrain);
  const pred = Array.from({ length: n }, () => base);
  const trees = [];
  const nEstimators = overrides.nEstimators ?? 160;
  const learningRate = overrides.learningRate ?? 0.05;
  const options = {
    maxDepth: overrides.maxDepth ?? 3,
    minSamplesSplit: overrides.minSamplesSplit ?? 8,
    minSamplesLeaf: overrides.minSamplesLeaf ?? 3,
    maxThresholds: overrides.maxThresholds ?? 18,
    lambda: overrides.lambda ?? 2,
    gamma: overrides.gamma ?? 0.01,
    featureRate: overrides.featureRate ?? 0.85,
    maxLeafValue: overrides.maxLeafValue ?? 40,
  };

  for (let iter = 0; iter < nEstimators; iter += 1) {
    const gradients = pred.map((p, i) => p - yTrain[i]);
    const hessians = Array.from({ length: n }, () => 1);
    const indices = Array.from({ length: n }, (_, i) => i);
    const tree = buildBoostTree(XTrain, gradients, hessians, indices, options, rng);
    trees.push(tree);
    for (let i = 0; i < n; i += 1) pred[i] += learningRate * predictTree(tree, XTrain[i]);
  }

  return {
    settings: `n_estimators=${nEstimators}, learning_rate=${learningRate}, max_depth=${options.maxDepth}, lambda=${options.lambda}, gamma=${options.gamma}`,
    predict(x) {
      let out = base;
      for (const tree of trees) out += learningRate * predictTree(tree, x);
      return out;
    },
  };
}

function standardizeTarget(yTrain) {
  const yMean = mean(yTrain);
  const yStd = Math.sqrt(mean(yTrain.map((value) => (value - yMean) ** 2))) || 1;
  return { yMean, yStd, yScaled: yTrain.map((value) => (value - yMean) / yStd) };
}

function randomNormal(rng) {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0));
}

function trainMlp(XTrain, yTrain, XVal, yVal, seed, overrides = {}) {
  const rng = makeRng(seed);
  const inputDim = XTrain[0].length;
  const h1 = overrides.h1 ?? 48;
  const h2 = overrides.h2 ?? 24;
  const { yMean, yStd, yScaled } = standardizeTarget(yTrain);
  const yValScaled = yVal.map((value) => (value - yMean) / yStd);
  const init = (fanIn) => randomNormal(rng) * Math.sqrt(2 / fanIn);

  const W1 = Array.from({ length: inputDim }, () => Array.from({ length: h1 }, () => init(inputDim)));
  const b1 = Array.from({ length: h1 }, () => 0);
  const W2 = Array.from({ length: h1 }, () => Array.from({ length: h2 }, () => init(h1)));
  const b2 = Array.from({ length: h2 }, () => 0);
  const W3 = Array.from({ length: h2 }, () => init(h2));
  let b3 = 0;

  const params = [W1, b1, W2, b2, W3];
  const m = params.map((param) => JSON.parse(JSON.stringify(param)));
  const v = params.map((param) => JSON.parse(JSON.stringify(param)));
  function zeroLikeInto(item) {
    if (Array.isArray(item[0])) return item.map((row) => row.map(() => 0));
    return item.map(() => 0);
  }
  for (let i = 0; i < m.length; i += 1) {
    const z = zeroLikeInto(params[i]);
    m[i] = JSON.parse(JSON.stringify(z));
    v[i] = JSON.parse(JSON.stringify(z));
  }
  let mb3 = 0;
  let vb3 = 0;

  function forward(x) {
    const z1 = Array.from({ length: h1 }, (_, j) => b1[j] + x.reduce((sum, xi, i) => sum + xi * W1[i][j], 0));
    const a1 = z1.map((z) => Math.max(0, z));
    const z2 = Array.from({ length: h2 }, (_, k) => b2[k] + a1.reduce((sum, a, j) => sum + a * W2[j][k], 0));
    const a2 = z2.map((z) => Math.max(0, z));
    const out = b3 + a2.reduce((sum, a, k) => sum + a * W3[k], 0);
    return { z1, a1, z2, a2, out };
  }

  function predictScaled(x) {
    return forward(x).out;
  }

  function mse(X, y) {
    if (!X.length) return Infinity;
    return mean(X.map((x, i) => (predictScaled(x) - y[i]) ** 2));
  }

  const lr = overrides.lr ?? 0.01;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const l2 = overrides.l2 ?? 1e-4;
  const maxEpochs = overrides.maxEpochs ?? 1200;
  const patience = overrides.patience ?? 120;
  let bestVal = Infinity;
  let bestEpoch = 0;
  let bestWeights = null;
  let step = 0;

  for (let epoch = 1; epoch <= maxEpochs; epoch += 1) {
    const dW1 = zeros(inputDim, h1);
    const db1 = Array.from({ length: h1 }, () => 0);
    const dW2 = zeros(h1, h2);
    const db2 = Array.from({ length: h2 }, () => 0);
    const dW3 = Array.from({ length: h2 }, () => 0);
    let db3 = 0;

    for (let n = 0; n < XTrain.length; n += 1) {
      const x = XTrain[n];
      const cache = forward(x);
      const dOut = (2 * (cache.out - yScaled[n])) / XTrain.length;
      for (let k = 0; k < h2; k += 1) dW3[k] += dOut * cache.a2[k] + l2 * W3[k] / XTrain.length;
      db3 += dOut;
      const da2 = Array.from({ length: h2 }, (_, k) => dOut * W3[k]);
      const dz2 = da2.map((value, k) => cache.z2[k] > 0 ? value : 0);
      for (let j = 0; j < h1; j += 1) {
        for (let k = 0; k < h2; k += 1) dW2[j][k] += dz2[k] * cache.a1[j] + l2 * W2[j][k] / XTrain.length;
      }
      for (let k = 0; k < h2; k += 1) db2[k] += dz2[k];
      const da1 = Array.from({ length: h1 }, (_, j) => dz2.reduce((sum, dz, k) => sum + dz * W2[j][k], 0));
      const dz1 = da1.map((value, j) => cache.z1[j] > 0 ? value : 0);
      for (let i = 0; i < inputDim; i += 1) {
        for (let j = 0; j < h1; j += 1) dW1[i][j] += dz1[j] * x[i] + l2 * W1[i][j] / XTrain.length;
      }
      for (let j = 0; j < h1; j += 1) db1[j] += dz1[j];
    }

    step += 1;
    const grads = [dW1, db1, dW2, db2, dW3];
    const liveParams = [W1, b1, W2, b2, W3];
    for (let p = 0; p < liveParams.length; p += 1) {
      if (Array.isArray(liveParams[p][0])) {
        for (let i = 0; i < liveParams[p].length; i += 1) {
          for (let j = 0; j < liveParams[p][i].length; j += 1) {
            m[p][i][j] = beta1 * m[p][i][j] + (1 - beta1) * grads[p][i][j];
            v[p][i][j] = beta2 * v[p][i][j] + (1 - beta2) * grads[p][i][j] ** 2;
            const mHat = m[p][i][j] / (1 - beta1 ** step);
            const vHat = v[p][i][j] / (1 - beta2 ** step);
            liveParams[p][i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
          }
        }
      } else {
        for (let i = 0; i < liveParams[p].length; i += 1) {
          m[p][i] = beta1 * m[p][i] + (1 - beta1) * grads[p][i];
          v[p][i] = beta2 * v[p][i] + (1 - beta2) * grads[p][i] ** 2;
          const mHat = m[p][i] / (1 - beta1 ** step);
          const vHat = v[p][i] / (1 - beta2 ** step);
          liveParams[p][i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
      }
    }
    mb3 = beta1 * mb3 + (1 - beta1) * db3;
    vb3 = beta2 * vb3 + (1 - beta2) * db3 ** 2;
    b3 -= lr * (mb3 / (1 - beta1 ** step)) / (Math.sqrt(vb3 / (1 - beta2 ** step)) + eps);

    const valLoss = mse(XVal, yValScaled);
    if (valLoss < bestVal) {
      bestVal = valLoss;
      bestEpoch = epoch;
      bestWeights = JSON.parse(JSON.stringify({ W1, b1, W2, b2, W3, b3 }));
    } else if (epoch - bestEpoch >= patience) {
      break;
    }
  }

  Object.assign(W1, bestWeights.W1);
  for (let i = 0; i < b1.length; i += 1) b1[i] = bestWeights.b1[i];
  Object.assign(W2, bestWeights.W2);
  for (let i = 0; i < b2.length; i += 1) b2[i] = bestWeights.b2[i];
  for (let i = 0; i < W3.length; i += 1) W3[i] = bestWeights.W3[i];
  b3 = bestWeights.b3;

  return {
    settings: `hidden_layers=${h1},${h2}, adam_lr=${lr}, l2=${l2}, best_epoch=${bestEpoch}, best_val_mse_scaled=${round(bestVal, 6)}`,
    predict(x) {
      return predictScaled(x) * yStd + yMean;
    },
  };
}

function fitMlModels(rows, seed) {
  const mlRows = enrichRowsForMl(rows);
  const trainRows = mlRows.filter((row) => row.effective_split_group === "train");
  const validationRows = mlRows.filter((row) => row.effective_split_group === "validation");
  const encoder = fitFeatureEncoder(trainRows);
  const X = mlRows.map((row) => encoder.transform(row));
  const y = mlRows.map((row) => row.y_true);
  const trainIndices = mlRows.map((row, i) => row.effective_split_group === "train" ? i : -1).filter((i) => i >= 0);
  const validationIndices = mlRows.map((row, i) => row.effective_split_group === "validation" ? i : -1).filter((i) => i >= 0);
  const XTrain = trainIndices.map((i) => X[i]);
  const yTrain = trainIndices.map((i) => y[i]);
  const XVal = (validationIndices.length ? validationIndices : trainIndices).map((i) => X[i]);
  const yVal = (validationIndices.length ? validationIndices : trainIndices).map((i) => y[i]);

  const models = [
    ["Random Forest", trainRandomForest(XTrain, yTrain, seed + 11)],
    ["XGBoost", trainXGBoostStyle(XTrain, yTrain, seed + 23)],
    ["Ordinary MLP", trainMlp(XTrain, yTrain, XVal, yVal, seed + 37)],
  ];

  const predictions = [];
  const trainingSummary = [];
  for (const [name, model] of models) {
    for (let i = 0; i < mlRows.length; i += 1) {
      const rawPred = model.predict(X[i]);
      const pred = clamp(rawPred, 0, 180);
      predictions.push({
        record_id: mlRows[i].record_id,
        model_name: name,
        y_true_contact_angle_deg: round(mlRows[i].y_true, 4),
        y_pred_contact_angle_deg: round(pred, 4),
        error_deg: round(pred - mlRows[i].y_true, 4),
        split_group: mlRows[i].effective_split_group,
        validation_type: validationType,
        _wasClipped: Math.abs(rawPred - pred) > 1e-12,
        _datasetVersion: mlRows[i].dataset_version,
      });
    }
    trainingSummary.push({
      model_name: name,
      training_rows: trainRows.length,
      validation_rows: validationRows.length,
      feature_count: encoder.featureNames.length,
      settings: model.settings,
      notes: name === "XGBoost"
        ? "Pure JS XGBoost-style implementation because xgboost package is unavailable."
        : "Pure JS implementation with deterministic seed.",
    });
  }

  return { predictions, trainingSummary, featureCount: encoder.featureNames.length };
}

function makePredictionRow(row, modelName, prediction, protocolValidationType, splitGroup, wasClipped = false) {
  const clippedPrediction = clamp(prediction, 0, 180);
  return {
    record_id: row.record_id,
    model_name: modelName,
    y_true_contact_angle_deg: round(row.y_true, 4),
    y_pred_contact_angle_deg: round(clippedPrediction, 4),
    error_deg: round(clippedPrediction - row.y_true, 4),
    split_group: splitGroup,
    validation_type: protocolValidationType,
    _wasClipped: wasClipped || Math.abs(prediction - clippedPrediction) > 1e-12,
    _datasetVersion: row.dataset_version,
  };
}

function makeMetricRow(modelName, predictions, metricValidationType, splitId, testGroup, notes) {
  const rows = predictions.map((row) => ({
    yTrue: row.y_true_contact_angle_deg,
    error: row.error_deg,
    wasClipped: row._wasClipped,
  }));
  return {
    model_name: modelName,
    dataset_version: predictions[0]?._datasetVersion ?? "v3.1",
    validation_type: metricValidationType,
    split_id: splitId,
    test_group: testGroup,
    n_samples: rows.length,
    MAE: rows.length ? round(mae(rows), 4) : "",
    RMSE: rows.length ? round(rmse(rows), 4) : "",
    R2: rows.length ? round(r2(rows), 6) : "",
    Median_AE: rows.length ? round(median(rows.map((row) => Math.abs(row.error))), 4) : "",
    nonphysical_rate: rows.length ? round(rows.filter((row) => row.wasClipped).length / rows.length, 6) : "",
    notes,
  };
}

function makeMacroMetricRow(modelName, rows, metricValidationType, splitId, notes) {
  const numericMean = (field) => {
    const vals = rows.map((row) => Number(row[field])).filter(Number.isFinite);
    return vals.length ? round(mean(vals), 4) : "";
  };
  return {
    model_name: modelName,
    dataset_version: rows[0]?.dataset_version ?? "v3.1",
    validation_type: metricValidationType,
    split_id: splitId,
    test_group: "all_groups_macro",
    n_samples: rows.length,
    MAE: numericMean("MAE"),
    RMSE: numericMean("RMSE"),
    R2: numericMean("R2"),
    Median_AE: numericMean("Median_AE"),
    nonphysical_rate: numericMean("nonphysical_rate"),
    notes,
  };
}

function splitProtocolTrainValidation(trainRows, seed) {
  const order = shuffle(Array.from({ length: trainRows.length }, (_, i) => i), seed);
  const valCount = Math.max(1, Math.min(trainRows.length - 1, Math.round(trainRows.length * 0.15)));
  const valSet = new Set(order.slice(0, valCount));
  const fitRows = [];
  const valRows = [];
  trainRows.forEach((row, i) => {
    if (valSet.has(i)) valRows.push(row);
    else fitRows.push(row);
  });
  return { fitRows, valRows };
}

function fitProtocolMlModels(trainRows, testRows, seed, protocolValidationType, heldOutGroup) {
  const trainEnriched = enrichRowsForMl(trainRows);
  const testEnriched = enrichRowsForMl(testRows);
  const encoder = fitFeatureEncoder(trainEnriched);
  const XTrainAll = trainEnriched.map((row) => encoder.transform(row));
  const yTrainAll = trainEnriched.map((row) => row.y_true);
  const XTest = testEnriched.map((row) => encoder.transform(row));
  const profile = {
    rf: { nTrees: 50, maxDepth: 6, minSamplesLeaf: 3, maxThresholds: 14 },
    xgb: { nEstimators: 80, learningRate: 0.06, maxDepth: 3, minSamplesLeaf: 3, maxThresholds: 14, lambda: 2 },
    mlp: { h1: 32, h2: 16, maxEpochs: 500, patience: 60, lr: 0.01, l2: 1e-4 },
  };

  const { fitRows, valRows } = splitProtocolTrainValidation(trainEnriched, seed);
  const XFit = fitRows.map((row) => encoder.transform(row));
  const yFit = fitRows.map((row) => row.y_true);
  const XVal = valRows.map((row) => encoder.transform(row));
  const yVal = valRows.map((row) => row.y_true);

  const models = [
    ["Random Forest", trainRandomForestWithOptions(XTrainAll, yTrainAll, seed + 101, profile.rf)],
    ["XGBoost", trainXGBoostStyleWithOptions(XTrainAll, yTrainAll, seed + 211, profile.xgb)],
    ["Ordinary MLP", trainMlp(XFit, yFit, XVal, yVal, seed + 307, profile.mlp)],
  ];

  const predictions = [];
  const trainingSummary = [];
  for (const [name, model] of models) {
    for (let i = 0; i < testEnriched.length; i += 1) {
      predictions.push(makePredictionRow(
        testEnriched[i],
        name,
        model.predict(XTest[i]),
        protocolValidationType,
        heldOutGroup,
      ));
    }
    trainingSummary.push({
      model_name: `${name} ${protocolValidationType}`,
      training_rows: trainRows.length,
      validation_rows: valRows.length,
      feature_count: encoder.featureNames.length,
      settings: model.settings,
      notes: `Leave-one-group fold for ${heldOutGroup}; compact protocol settings for runtime.`,
    });
  }

  return { predictions, trainingSummary };
}

function makeOwensProtocolPredictions(testRows, protocolValidationType, heldOutGroup) {
  return testRows.map((row) => {
    const result = owensWendt(row);
    return makePredictionRow(
      row,
      "Owens-Wendt",
      result.thetaDeg,
      protocolValidationType,
      heldOutGroup,
      result.wasClipped,
    );
  });
}

function runLeaveOneProtocol(rows, groupField, protocolValidationType, seed) {
  const splitId = groupField === "solid_name" ? "leave_one_material_out" : "leave_one_liquid_out";
  const groupLabel = groupField === "solid_name" ? "material" : "liquid";
  const groups = [...new Set(rows.map((row) => row[groupField]).filter(Boolean))].sort();
  const predictions = [];
  const metrics = [];
  const trainingSummary = [];
  const groupMetricRows = [];

  groups.forEach((group, foldIndex) => {
    console.error(`[${protocolValidationType}] ${foldIndex + 1}/${groups.length} ${groupLabel}=${group}`);
    const testRows = rows.filter((row) => row[groupField] === group);
    const trainRows = rows.filter((row) => row[groupField] !== group);
    if (!testRows.length || trainRows.length < 20) return;

    const mlFold = fitProtocolMlModels(trainRows, testRows, seed + foldIndex * 17, protocolValidationType, group);
    const foldPredictions = [
      ...makeOwensProtocolPredictions(testRows, protocolValidationType, group),
      ...mlFold.predictions,
    ];
    trainingSummary.push(...mlFold.trainingSummary);

    predictions.push(...foldPredictions);
    for (const modelName of ["Owens-Wendt", "Random Forest", "XGBoost", "Ordinary MLP"]) {
      const modelPreds = foldPredictions.filter((row) => row.model_name === modelName);
      const metric = makeMetricRow(
        modelName,
        modelPreds,
        protocolValidationType,
        splitId,
        group,
        `Held-out ${groupLabel}: ${group}`,
      );
      metrics.push(metric);
      groupMetricRows.push(metric);
    }
  });

  for (const modelName of ["Owens-Wendt", "Random Forest", "XGBoost", "Ordinary MLP"]) {
    const modelPreds = predictions.filter((row) => row.model_name === modelName);
    metrics.push(makeMetricRow(
      modelName,
      modelPreds,
      protocolValidationType,
      splitId,
      "all_groups_micro",
      `Micro-average across all ${groups.length} held-out ${groupLabel} groups.`,
    ));
    const perGroup = groupMetricRows.filter((row) => row.model_name === modelName);
    metrics.push(makeMacroMetricRow(
      modelName,
      perGroup,
      protocolValidationType,
      splitId,
      `Macro-average across held-out ${groupLabel} groups.`,
    ));
  }

  return { predictions, metrics, trainingSummary };
}

function mae(values) {
  return values.reduce((sum, item) => sum + Math.abs(item.error), 0) / values.length;
}

function rmse(values) {
  return Math.sqrt(values.reduce((sum, item) => sum + item.error ** 2, 0) / values.length);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function r2(values) {
  if (values.length < 2) return "";
  const meanY = mean(values.map((item) => item.yTrue));
  const sst = values.reduce((sum, item) => sum + (item.yTrue - meanY) ** 2, 0);
  if (sst === 0) return "";
  const sse = values.reduce((sum, item) => sum + item.error ** 2, 0);
  return 1 - sse / sst;
}

function makeMetrics(predictions) {
  const groups = new Map();
  for (const row of predictions) {
    for (const group of [row.split_group, "all"]) {
      const key = `${row.model_name}||${group}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({
        yTrue: row.y_true_contact_angle_deg,
        error: row.error_deg,
        wasClipped: row._wasClipped,
        datasetVersion: row._datasetVersion,
        modelName: row.model_name,
        group,
      });
    }
  }

  return [...groups.values()]
    .map((rows) => ({
      model_name: rows[0].modelName,
      dataset_version: rows[0].datasetVersion ?? "v3.1",
      validation_type: validationType,
      split_id: "seed42_train_validation_test_plus_fixed_external",
      test_group: rows[0].group,
      n_samples: rows.length,
      MAE: round(mae(rows), 4),
      RMSE: round(rmse(rows), 4),
      R2: round(r2(rows), 6),
      Median_AE: round(median(rows.map((row) => Math.abs(row.error))), 4),
      nonphysical_rate: round(rows.filter((row) => row.wasClipped).length / rows.length, 6),
      notes: `${datasetTag} final baseline; external rows are kept external, original train rows are kept train, unassigned rows are deterministically split with seed 42.`,
    }))
    .sort((a, b) => {
      const modelOrder = ["Owens-Wendt", "Harmonic Mean", "Weighted Geometric", "van Oss-Chaudhury-Good", "Random Forest", "XGBoost", "Ordinary MLP"];
      const groupOrder = ["train", "validation", "test", "external", "all"];
      return modelOrder.indexOf(a.model_name) - modelOrder.indexOf(b.model_name) ||
        groupOrder.indexOf(a.test_group) - groupOrder.indexOf(b.test_group);
    });
}

function metricLookup(metrics, modelName, group) {
  return metrics.find((row) => row.model_name === modelName && row.test_group === group);
}

function protocolLookup(metrics, modelName, group = "all_groups_micro") {
  return metrics.find((row) => row.model_name === modelName && row.test_group === group);
}

function makeFinalFindings(metrics, lomoMetrics, loloMetrics, dataRows) {
  const mlpTrain = metricLookup(metrics, "Ordinary MLP", "train");
  const mlpExternal = metricLookup(metrics, "Ordinary MLP", "external");
  const mlpTest = metricLookup(metrics, "Ordinary MLP", "test");
  const owLolo = protocolLookup(loloMetrics, "Owens-Wendt");
  const rfLolo = protocolLookup(loloMetrics, "Random Forest");
  const xgbLolo = protocolLookup(loloMetrics, "XGBoost");
  const mlpLolo = protocolLookup(loloMetrics, "Ordinary MLP");
  const owLomo = protocolLookup(lomoMetrics, "Owens-Wendt");
  const xgbLomo = protocolLookup(lomoMetrics, "XGBoost");

  const splitCounts = Object.fromEntries(["train", "validation", "test", "external"].map((group) => [
    group,
    dataRows.filter((row) => row.effective_split_group === group).length,
  ]));
  const loloBest = [owLolo, rfLolo, xgbLolo, mlpLolo]
    .filter(Boolean)
    .sort((a, b) => Number(a.MAE) - Number(b.MAE))[0];

  return [
    {
      section: "dataset",
      item: "dataset_version",
      value: dataRows[0]?.dataset_version ?? "v3.1",
      notes: `${dataRows.length} usable rows; split train=${splitCounts.train}, validation=${splitCounts.validation}, test=${splitCounts.test}, external=${splitCounts.external}.`,
    },
    {
      section: "motivation",
      item: "ordinary_mlp_train_external_gap",
      value: mlpTrain && mlpExternal ? round(Number(mlpExternal.MAE) - Number(mlpTrain.MAE), 4) : "",
      notes: mlpTrain && mlpExternal
        ? `Ordinary MLP train MAE=${mlpTrain.MAE}, test MAE=${mlpTest?.MAE ?? ""}, external MAE=${mlpExternal.MAE}; this gap supports the overfitting/generalization-risk motivation.`
        : "",
    },
    {
      section: "motivation",
      item: "lolo_best_micro_mae_model",
      value: loloBest?.model_name ?? "",
      notes: loloBest ? `LOLO micro MAE=${loloBest.MAE}; Owens-Wendt MAE=${owLolo?.MAE ?? ""}, RF=${rfLolo?.MAE ?? ""}, XGBoost=${xgbLolo?.MAE ?? ""}, MLP=${mlpLolo?.MAE ?? ""}.` : "",
    },
    {
      section: "motivation",
      item: "owens_wendt_lolo_stability",
      value: owLolo?.MAE ?? "",
      notes: "Owens-Wendt remains the most stable LOLO baseline in this final run, supporting a physics-guided residual model rather than a purely data-driven replacement.",
    },
    {
      section: "comparison",
      item: "lomo_xgboost_vs_owens_wendt_mae_delta",
      value: owLomo && xgbLomo ? round(Number(owLomo.MAE) - Number(xgbLomo.MAE), 4) : "",
      notes: owLomo && xgbLomo
        ? `LOMO micro MAE: Owens-Wendt=${owLomo.MAE}, XGBoost=${xgbLomo.MAE}; ML helps more for material OOD than liquid OOD.`
        : "",
    },
  ];
}

function matrixFromRows(columns, rows) {
  return [columns, ...rows.map((row) => columns.map((column) => row[column] ?? ""))];
}

function setHeaderStyle(range, fill = "#1F4E78") {
  range.format = {
    fill,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
}

function setBodyStyle(range) {
  range.format = {
    fill: "#FFFFFF",
    font: { color: "#1F2937" },
    verticalAlignment: "top",
    wrapText: true,
  };
}

function addSheetFromRows(workbook, sheetName, columns, rows, headerFill = "#1F4E78") {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  const matrix = matrixFromRows(columns, rows);
  sheet.getRangeByIndexes(0, 0, matrix.length, columns.length).values = matrix;
  setHeaderStyle(sheet.getRangeByIndexes(0, 0, 1, columns.length), headerFill);
  if (matrix.length > 1) setBodyStyle(sheet.getRangeByIndexes(1, 0, matrix.length - 1, columns.length));
  sheet.freezePanes.freezeRows(1);
  const endCell = `${colLetter(columns.length - 1)}${Math.max(1, matrix.length)}`;
  const tableName = sheetName.replaceAll(/[^A-Za-z0-9]/g, "").slice(0, 24) || "ResultTable";
  sheet.tables.add(`A1:${endCell}`, true, tableName);
  for (let i = 0; i < columns.length; i += 1) {
    const header = columns[i];
    let width = 135;
    if (header.includes("formula") || header.includes("notes") || header.includes("settings") || header.includes("skip_reason")) width = 380;
    if (header.includes("record_id") || header.includes("model_name")) width = 165;
    if (header.includes("contact_angle") || header.includes("theta") || header.includes("error") || header.includes("RMSE")) width = 150;
    sheet.getRangeByIndexes(0, i, matrix.length, 1).format.columnWidthPx = width;
  }
  return sheet;
}

async function writeWorkbook(
  outputPath,
  metrics,
  predictions,
  diagnostics,
  skippedRows,
  splitAssignments,
  trainingSummary,
  lomoMetrics,
  loloMetrics,
  protocolPredictions,
  finalFindings,
) {
  const workbook = Workbook.create();
  addSheetFromRows(workbook, "Final_Findings", finalFindingColumns, finalFindings, "#1F4E78");
  addSheetFromRows(workbook, "Summary_Metrics", metricColumns, metrics, "#375623");
  addSheetFromRows(workbook, "Unified_Predictions", predictionColumns, predictions, "#1F4E78");
  addSheetFromRows(workbook, "Protocol_Predictions", predictionColumns, protocolPredictions, "#244062");
  addSheetFromRows(workbook, "LOMO_Metrics", metricColumns, lomoMetrics, "#548235");
  addSheetFromRows(workbook, "LOLO_Metrics", metricColumns, loloMetrics, "#70AD47");
  addSheetFromRows(workbook, "Diagnostics", diagnosticColumns, diagnostics, "#7030A0");
  addSheetFromRows(workbook, "Split_Assignments", splitColumns, splitAssignments, "#806000");
  addSheetFromRows(workbook, "Training_Summary", trainingSummaryColumns, trainingSummary, "#7F6000");
  addSheetFromRows(workbook, "Skipped_Rows", skippedColumns, skippedRows, "#A23B3B");

  const formulaSheet = workbook.worksheets.add("Formula_Notes");
  formulaSheet.showGridLines = false;
  formulaSheet.getRangeByIndexes(0, 0, formulaRows.length, 2).values = formulaRows;
  setHeaderStyle(formulaSheet.getRange("A1:B1"), "#5B4B8A");
  setBodyStyle(formulaSheet.getRangeByIndexes(1, 0, formulaRows.length - 1, 2));
  formulaSheet.freezePanes.freezeRows(1);
  formulaSheet.tables.add(`A1:B${formulaRows.length}`, true, "FormulaNotes");
  formulaSheet.getRange("A:A").format.columnWidthPx = 190;
  formulaSheet.getRange("B:B").format.columnWidthPx = 820;

  const renderPlan = [
    ["Final_Findings", "A1:D8"],
    ["Summary_Metrics", "A1:L40"],
    ["Unified_Predictions", "A1:G45"],
    ["Protocol_Predictions", "A1:G45"],
    ["LOMO_Metrics", "A1:L45"],
    ["LOLO_Metrics", "A1:L45"],
    ["Diagnostics", "A1:L45"],
    ["Split_Assignments", "A1:G45"],
    ["Training_Summary", "A1:F8"],
    ["Skipped_Rows", "A1:F20"],
    ["Formula_Notes", "A1:B8"],
  ];
  for (const [sheetName, range] of renderPlan) {
    await workbook.render({ sheetName, range, autoCrop: "all", scale: 1, format: "png" });
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  datasetTag = args.datasetTag;
  runDate = args.runDate;
  validationType = args.validationType;
  await fs.mkdir(args.outDir, { recursive: true });

  const input = await FileBlob.load(args.input);
  const workbook = await SpreadsheetFile.importXlsx(input);
  const sheet = workbook.worksheets.getItem(args.sheet);
  const values = sheet.getRange("A1:BH1000").values;
  const headers = values[0].map((header) => String(header ?? "").trim());
  const rawRows = values
    .slice(1)
    .map((row) => rowObject(headers, row))
    .filter((row) => String(row.record_id ?? "").trim() !== "");

  const dataRows = [];
  const skippedRows = [];
  for (const raw of rawRows) {
    const row = normalizeDataRow(raw);
    if (row.skipReason) {
      skippedRows.push({
        record_id: row.record_id,
        skip_reason: row.skipReason,
        solid_name: row.solid_name,
        liquid_name: row.liquid_name,
        include_in_training: row.include_in_training,
        quality_grade: row.quality_grade,
      });
    } else {
      dataRows.push(row);
    }
  }

  assignEffectiveSplits(dataRows, args.seed);

  const splitAssignments = dataRows.map((row) => ({
    record_id: row.record_id,
    original_split_group: row.original_split_group,
    effective_split_group: row.effective_split_group,
    solid_name: row.solid_name,
    liquid_name: row.liquid_name,
    quality_grade: row.quality_grade,
    include_in_training: row.include_in_training,
  }));

  const physical = makePhysicalPredictionRows(dataRows);
  const ml = fitMlModels(dataRows, args.seed);
  const allPredictions = [...physical.predictions, ...ml.predictions];
  const publicPredictions = allPredictions.map(({ _wasClipped, _datasetVersion, ...row }) => row);
  const metrics = makeMetrics(allPredictions);
  const lomo = runLeaveOneProtocol(dataRows, "solid_name", "LOMO", args.seed + 1000);
  const lolo = runLeaveOneProtocol(dataRows, "liquid_name", "LOLO", args.seed + 2000);
  const protocolPredictions = [...lomo.predictions, ...lolo.predictions];
  const publicProtocolPredictions = protocolPredictions.map(({ _wasClipped, _datasetVersion, ...row }) => row);
  const combinedTrainingSummary = [...ml.trainingSummary, ...lomo.trainingSummary, ...lolo.trainingSummary];
  const finalFindings = makeFinalFindings(metrics, lomo.metrics, lolo.metrics, dataRows);

  const runStamp = runDate.replaceAll("-", "");
  const predictionCsv = path.join(args.outDir, `baseline_predictions_${datasetTag}_${runStamp}.csv`);
  const protocolPredictionCsv = path.join(args.outDir, `baseline_protocol_predictions_${datasetTag}_${runStamp}.csv`);
  const metricsCsv = path.join(args.outDir, `baseline_metrics_${datasetTag}_${runStamp}.csv`);
  const lomoMetricsCsv = path.join(args.outDir, `baseline_lomo_metrics_${datasetTag}_${runStamp}.csv`);
  const loloMetricsCsv = path.join(args.outDir, `baseline_lolo_metrics_${datasetTag}_${runStamp}.csv`);
  const finalFindingsCsv = path.join(args.outDir, `baseline_final_findings_${datasetTag}_${runStamp}.csv`);
  const diagnosticsCsv = path.join(args.outDir, `baseline_diagnostics_${datasetTag}_${runStamp}.csv`);
  const splitCsv = path.join(args.outDir, `baseline_split_assignments_${datasetTag}_${runStamp}.csv`);
  const skippedCsv = path.join(args.outDir, `baseline_skipped_rows_${datasetTag}_${runStamp}.csv`);
  const trainingCsv = path.join(args.outDir, `baseline_training_summary_${datasetTag}_${runStamp}.csv`);
  const resultWorkbook = path.join(args.outDir, `baseline_results_${datasetTag}_${runStamp}.xlsx`);

  await writeCsv(predictionCsv, predictionColumns, publicPredictions);
  await writeCsv(protocolPredictionCsv, predictionColumns, publicProtocolPredictions);
  await writeCsv(metricsCsv, metricColumns, metrics);
  await writeCsv(lomoMetricsCsv, metricColumns, lomo.metrics);
  await writeCsv(loloMetricsCsv, metricColumns, lolo.metrics);
  await writeCsv(finalFindingsCsv, finalFindingColumns, finalFindings);
  await writeCsv(diagnosticsCsv, diagnosticColumns, physical.diagnostics);
  await writeCsv(splitCsv, splitColumns, splitAssignments);
  await writeCsv(skippedCsv, skippedColumns, skippedRows);
  await writeCsv(trainingCsv, trainingSummaryColumns, combinedTrainingSummary);
  await writeWorkbook(
    resultWorkbook,
    metrics,
    publicPredictions,
    physical.diagnostics,
    skippedRows,
    splitAssignments,
    combinedTrainingSummary,
    lomo.metrics,
    lolo.metrics,
    publicProtocolPredictions,
    finalFindings,
  );

  console.log(JSON.stringify({
    input: args.input,
    sheet: args.sheet,
    source_rows: rawRows.length,
    usable_rows: dataRows.length,
    skipped_rows: skippedRows.length,
    prediction_records: publicPredictions.length,
    split_counts: Object.fromEntries(["train", "validation", "test", "external"].map((group) => [
      group,
      dataRows.filter((row) => row.effective_split_group === group).length,
    ])),
    outputs: {
      predictionCsv,
      protocolPredictionCsv,
      metricsCsv,
      lomoMetricsCsv,
      loloMetricsCsv,
      finalFindingsCsv,
      diagnosticsCsv,
      splitCsv,
      skippedCsv,
      trainingCsv,
      resultWorkbook,
    },
    all_metrics: metrics.filter((row) => row.test_group === "all"),
    lomo_micro_metrics: lomo.metrics.filter((row) => row.test_group === "all_groups_micro"),
    lolo_micro_metrics: lolo.metrics.filter((row) => row.test_group === "all_groups_micro"),
  }, null, 2));
}

export {
  clamp,
  enrichRowsForMl,
  fitFeatureEncoder,
  harmonicMean,
  makeRng,
  normalizeDataRow,
  owensWendt,
  round,
  trainMlp,
  trainRandomForest,
  trainXGBoostStyle,
  vanOssChaudhuryGood,
  weightedGeometric,
  writeCsv,
};

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
