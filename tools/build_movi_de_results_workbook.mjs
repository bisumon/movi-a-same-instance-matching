import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repo = process.argv[2];
const out = process.argv[3];
const previewDir = process.argv[4];
if (!repo || !out || !previewDir) throw new Error("usage: builder <repo> <output.xlsx> <preview-dir>");

const readJson = async (p) => JSON.parse(await fs.readFile(path.join(repo, p), "utf8"));
const parseCsv = (text) => {
  const rows = []; let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i], n = text[i + 1];
    if (c === '"' && quoted && n === '"') { cell += '"'; i++; }
    else if (c === '"') quoted = !quoted;
    else if (c === ',' && !quoted) { row.push(cell); cell = ""; }
    else if ((c === '\n' || c === '\r') && !quoted) {
      if (c === '\r' && n === '\n') i++;
      row.push(cell); if (row.some(v => v !== "")) rows.push(row); row = []; cell = "";
    } else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const headers = rows.shift();
  return rows.map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
};
const typed = (v) => v === "" ? null : (!Number.isNaN(Number(v)) && v.trim() !== "" ? Number(v) : v);

const e = await readJson("results/movi_de_phase8_regime1/movi_e_in_domain_results.json");
const d = await readJson("results/movi_de_phase8_regime2/movi_d_in_domain_results.json");
const transfer = await readJson("results/movi_de_phase8_regime3/movi_d_to_e_transfer_results.json");
const phase9 = await readJson("results/movi_de_phase9/phase9_criteria_evaluation.json");
const presentation = await readJson("results/movi_de_final/presentation_table_metrics.json");
const failure = await readJson("failure_gallery/movi_de_phase10/selection_manifest.json");
const dEmbed = await readJson("runs/movi_de_confirmatory/phase7/movi_d_rgb_embeddings/rgb_embedding_manifest.json");
const eEmbed = await readJson("runs/movi_de_confirmatory/phase7/movi_e_rgb_embeddings/rgb_embedding_manifest.json");
const poseRows = parseCsv(await fs.readFile(path.join(repo, "results/movi_de_phase7_pose_noise/phase7_pose_noise_results_table.csv"), "utf8"));
const reviewRows = parseCsv(await fs.readFile(path.join(repo, "failure_gallery/movi_de_phase10/failure_review.csv"), "utf8"));
const strataRows = parseCsv(await fs.readFile(path.join(repo, "results/movi_de_final/pair_strata_results.csv"), "utf8"));
const strataCiRows = parseCsv(await fs.readFile(path.join(repo, "results/movi_de_final/pair_strata_d_minus_c_intervals.csv"), "utf8"));

const wb = Workbook.create();
const navy = "#17324D", blue = "#2E75B6", light = "#EAF2F8", gray = "#F3F5F7", ink = "#17212B", green = "#DFF2E1", amber = "#FFF1CC";
function title(sheet, text, subtitle, width) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${width}1`).merge(); sheet.getRange("A1").values = [[text]];
  sheet.getRange(`A1:${width}1`).format = { fill: navy, font: { bold: true, color: "#FFFFFF", size: 16 }, rowHeight: 30, verticalAlignment: "center" };
  sheet.getRange(`A2:${width}2`).merge(); sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${width}2`).format = { fill: light, font: { italic: true, color: ink }, wrapText: true, rowHeight: 34, verticalAlignment: "center" };
}
function table(sheet, startRow, headers, rows, widths = []) {
  const endCol = String.fromCharCode(64 + headers.length);
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).values = [headers];
  sheet.getRange(`A${startRow}:${endCol}${startRow}`).format = { fill: blue, font: { bold: true, color: "#FFFFFF" }, wrapText: true, rowHeight: 28, verticalAlignment: "center" };
  if (rows.length) sheet.getRange(`A${startRow + 1}:${endCol}${startRow + rows.length}`).values = rows;
  const used = sheet.getRange(`A${startRow}:${endCol}${startRow + rows.length}`);
  used.format.borders = { insideHorizontal: { style: "thin", color: "#D9E2EA" }, bottom: { style: "thin", color: "#AAB8C5" } };
  used.format.wrapText = true;
  widths.forEach((w, i) => sheet.getRange(`${String.fromCharCode(65 + i)}:${String.fromCharCode(65 + i)}`).format.columnWidth = w);
  sheet.freezePanes.freezeRows(startRow);
  return startRow + rows.length;
}
function section(sheet, row, text, width = "I") {
  sheet.getRange(`A${row}:${width}${row}`).merge();
  sheet.getRange(`A${row}`).values = [[text]];
  sheet.getRange(`A${row}:${width}${row}`).format = { fill: light, font: { bold: true, color: ink }, rowHeight: 22, verticalAlignment: "center" };
}

{
  const s = wb.worksheets.add("Presentation Table");
  title(s, "MOVi-D/E Experiment - Presentation Result Table", "Study design, all eight clean systems, paired confidence intervals, operational counts, difficult-condition results, controls, and interpretation context", "I");
  const systemInfo = {
    A_rgb: ["A - RGB only", "Object-crop appearance only"],
    B_rgb_2d: ["B - RGB + 2D", "Appearance plus ordinary 2D controls"],
    C_camera_geometry: ["C - Camera geometry", "Appearance, 2D controls, depth, and camera-space geometry"],
    D_pose_aligned_geometry: ["D - Pose-aligned geometry", "Same feature budget as C, with geometry aligned to world coordinates"],
    G_camera_geometry_only: ["G-C - Camera geometry only", "2D, depth, and camera-space geometry without RGB appearance"],
    G_pose_aligned_geometry_only: ["G-D - Aligned geometry only", "2D, depth, and world-aligned geometry without RGB appearance"],
    P_pose_only: ["P - Pose only", "Camera displacement and rotation only; shortcut control"],
    S_shuffled_pose: ["S - Shuffled pose", "D inputs transformed with deliberately mismatched camera poses"],
  };
  const systemRows = (data) => Object.entries(data.aggregate).map(([system, splits]) => {
    const m = splits.test;
    return [systemInfo[system][0], systemInfo[system][1], m.auroc, m.pr_auc, m.f1_at_locked_max_f1_threshold, m.false_match_rate_at_locked_90_recall_threshold, m.recall_at_locked_90_recall_threshold, Math.round(1000 * m.recall_at_locked_90_recall_threshold), Math.round(1000 * m.false_match_rate_at_locked_90_recall_threshold)];
  });
  const pairedRows = (data, derived, moving) => {
    const c = data.aggregate.C_camera_geometry.test;
    const dd = data.aggregate.D_pose_aligned_geometry.test;
    const paired = data.paired_differences_vs_C.D_pose_aligned_geometry;
    const specs = [
      ["AUROC", c.auroc, dd.auroc, paired.auroc.system_minus_C ?? paired.auroc.system_minus_reference, paired.auroc, moving ? "0.31 percentage points better ranking performance" : "Small fixed-camera ranking difference; interval crosses zero"],
      ["PR-AUC", c.pr_auc, dd.pr_auc, derived.pr_auc.comparison_minus_reference, derived.pr_auc, moving ? "0.29 percentage points better precision-recall performance" : "Small fixed-camera difference; interval crosses zero"],
      ["F1 at locked max-F1 threshold", c.f1_at_locked_max_f1_threshold, dd.f1_at_locked_max_f1_threshold, derived.f1_at_locked_max_f1_threshold.comparison_minus_reference, derived.f1_at_locked_max_f1_threshold, moving ? "1.16 percentage points higher fixed-threshold F1" : "No resolved fixed-camera F1 advantage"],
      ["False-match rate at locked 90%-recall rule", c.false_match_rate_at_locked_90_recall_threshold, dd.false_match_rate_at_locked_90_recall_threshold, paired.false_match_rate.comparison_minus_reference, paired.false_match_rate, moving ? "2 fewer nonmatches accepted per 1,000; interval crosses zero" : "2 fewer nonmatches accepted per 1,000; interval crosses zero"],
      ["Achieved genuine-match recall", c.recall_at_locked_90_recall_threshold, dd.recall_at_locked_90_recall_threshold, paired.recall.comparison_minus_reference, paired.recall, moving ? "10 additional genuine matches retained per 1,000; interval crosses zero" : "6 fewer genuine matches retained per 1,000; interval crosses zero"],
    ];
    return specs.map(([metric, cv, dv, delta, ci, reading]) => [metric, cv, dv, delta, ci.paired_video_cluster_ci_low, ci.paired_video_cluster_ci_high, (ci.paired_video_cluster_ci_low > 0 || ci.paired_video_cluster_ci_high < 0) ? "Yes" : "No", reading]);
  };

  section(s, 4, "Study design and evaluation basis");
  table(s, 5, ["Design item", "MOVi-D", "Design item", "MOVi-E"], [
    ["Selected videos", 150, "Selected videos", 150],
    ["Locked video split", "90 train / 30 dev / 30 test", "Locked video split", "90 train / 30 dev / 30 test"],
    ["Pair split", "6,000 train / 2,000 dev / 2,000 test", "Pair split", "6,000 train / 2,000 dev / 2,000 test"],
    ["Test pair balance", "1,000 genuine / 1,000 nonmatch", "Test pair balance", "1,000 genuine / 1,000 nonmatch"],
    ["Cross-video pairs", 0, "Cross-video pairs", 0],
    ["Bootstrap", "10,000 paired video-cluster resamples", "Bootstrap unit", "30 test videos"],
  ], [24, 34, 24, 34]);

  section(s, 13, "All eight clean system versions on the locked 2,000-pair MOVi-E moving-camera test set");
  table(s, 14, ["Version", "Information used", "AUROC", "PR-AUC", "F1 locked", "False-match rate", "Genuine-match recall", "Genuine retained (of 1,000)", "False accepted (of 1,000)"], systemRows(e), [29, 62, 12, 12, 12, 16, 18, 22, 21]);
  s.getRange("C15:G22").format.numberFormat = "0.0000";

  section(s, 24, "All eight clean system versions on the locked 2,000-pair MOVi-D fixed-camera test set");
  table(s, 25, ["Version", "Information used", "AUROC", "PR-AUC", "F1 locked", "False-match rate", "Genuine-match recall", "Genuine retained (of 1,000)", "False accepted (of 1,000)"], systemRows(d), [29, 62, 12, 12, 12, 16, 18, 22, 21]);
  s.getRange("C26:G33").format.numberFormat = "0.0000";

  section(s, 35, "Primary MOVi-E C-to-D change with paired 95% video-cluster confidence intervals");
  table(s, 36, ["Metric", "C", "D", "D - C", "CI lower", "CI upper", "Excludes zero", "Plain-language reading"], pairedRows(e, presentation.datasets.movi_e, true), [38, 13, 13, 13, 13, 13, 16, 62]);
  s.getRange("B37:F41").format.numberFormat = "0.0000";

  section(s, 43, "Operational impact on the balanced MOVi-E test set");
  const ec = e.aggregate.C_camera_geometry.test, ed = e.aggregate.D_pose_aligned_geometry.test;
  const cFalse = Math.round(1000 * ec.false_match_rate_at_locked_90_recall_threshold), dFalse = Math.round(1000 * ed.false_match_rate_at_locked_90_recall_threshold);
  const cRetained = Math.round(1000 * ec.recall_at_locked_90_recall_threshold), dRetained = Math.round(1000 * ed.recall_at_locked_90_recall_threshold);
  table(s, 44, ["Test outcome", "C count", "D count", "Count change", "Relative change", "Meaning"], [
    ["False matches accepted", cFalse, dFalse, dFalse - cFalse, (dFalse - cFalse) / cFalse, "2 fewer false matches; about 22.2% relative reduction"],
    ["Genuine matches retained", cRetained, dRetained, dRetained - cRetained, (dRetained - cRetained) / cRetained, "10 additional genuine matches retained"],
    ["Genuine matches missed", 1000 - cRetained, 1000 - dRetained, cRetained - dRetained, (cRetained - dRetained) / (1000 - cRetained), "10 fewer genuine matches missed"],
    ["Nonmatches correctly rejected", 1000 - cFalse, 1000 - dFalse, cFalse - dFalse, (cFalse - dFalse) / (1000 - cFalse), "2 additional nonmatches correctly rejected"],
  ], [38, 14, 14, 16, 18, 65]);
  s.getRange("E45:E48").format.numberFormat = "0.0%";

  section(s, 50, "Fixed-camera MOVi-D C-to-D change with paired 95% video-cluster confidence intervals");
  table(s, 51, ["Metric", "C", "D", "D - C", "CI lower", "CI upper", "Excludes zero", "Plain-language reading"], pairedRows(d, presentation.datasets.movi_d, false), [38, 13, 13, 13, 13, 13, 16, 62]);
  s.getRange("B52:F56").format.numberFormat = "0.0000";

  const strataIndex = new Map(strataRows.map(row => [[row.dataset, row.stratum_variable, row.stratum_level, row.system].join("|"), row]));
  const difficultRows = [];
  const addStratum = (label, variable, level, metric = "auroc") => {
    const cRow = strataIndex.get(["movi_e", variable, level, "C_camera_geometry"].join("|"));
    const dRow = strataIndex.get(["movi_e", variable, level, "D_pose_aligned_geometry"].join("|"));
    const cv = typed(cRow[metric]), dv = typed(dRow[metric]);
    difficultRows.push([label, metric === "false_match_rate" ? "False-match rate" : "AUROC", cv, dv, dv - cv, metric === "false_match_rate" ? (dv < cv ? "Yes" : "No") : (dv > cv ? "Yes" : "No")]);
  };
  for (const level of ["low", "medium", "high"]) addStratum(`${level[0].toUpperCase() + level.slice(1)} temporal gap`, "temporal_gap", level);
  for (const level of ["low", "medium", "high"]) addStratum(`${level[0].toUpperCase() + level.slice(1)} minimum visibility`, "minimum_visibility", level);
  addStratum("Dynamic-dynamic pairs", "dynamic_status", "dynamic-dynamic");
  addStratum("Static-static pairs", "dynamic_status", "static-static");
  for (const level of ["low", "medium", "high"]) addStratum(`${level[0].toUpperCase() + level.slice(1)} camera displacement`, "camera_displacement_scene_units", level);
  for (const level of ["low", "medium", "high"]) addStratum(`${level[0].toUpperCase() + level.slice(1)} camera rotation`, "relative_camera_rotation_degrees", level);
  addStratum("Easy negative pairs", "negative_difficulty", "easy", "false_match_rate");
  addStratum("Hard negative pairs", "negative_difficulty", "hard", "false_match_rate");
  section(s, 58, "Where pose alignment helped: MOVi-E performance by difficult condition");
  table(s, 59, ["Condition", "Metric", "C", "D", "D - C", "Improved"], difficultRows, [40, 20, 14, 14, 14, 14]);
  s.getRange(`C60:E${59 + difficultRows.length}`).format.numberFormat = "0.0000";

  const h = Object.fromEntries(phase9.criteria.map(x => [x.id, x]));
  const mechanismRows = [
    ["Primary moving-camera D - C AUROC", h.H1.evidence.D_minus_C_AUROC, h.H1.evidence.ci_low, h.H1.evidence.ci_high, "Yes", "Primary rule passed"],
    ["High-minus-low translation benefit", h.H3.evidence.MOVi_E_motion_strata.camera_displacement_scene_units.high_minus_low_D_minus_C_AUROC, h.H3.evidence.MOVi_E_motion_strata.camera_displacement_scene_units.paired_video_cluster_ci_low, h.H3.evidence.MOVi_E_motion_strata.camera_displacement_scene_units.paired_video_cluster_ci_high, "Yes", "Alignment helps more when translation is larger"],
    ["High-minus-low rotation benefit", h.H3.evidence.MOVi_E_motion_strata.relative_camera_rotation_degrees.high_minus_low_D_minus_C_AUROC, h.H3.evidence.MOVi_E_motion_strata.relative_camera_rotation_degrees.paired_video_cluster_ci_low, h.H3.evidence.MOVi_E_motion_strata.relative_camera_rotation_degrees.paired_video_cluster_ci_high, "Yes", "Alignment helps more when rotation is larger"],
    ["Shuffled pose S - D AUROC", h.H5.evidence.S_vs_D.auroc.system_minus_D, h.H5.evidence.S_vs_D.auroc.paired_video_cluster_ci_low, h.H5.evidence.S_vs_D.auroc.paired_video_cluster_ci_high, "Yes", "Wrong correspondence removes the clean-D advantage"],
    ["Maximum combined pose-noise change on MOVi-E", h.H6.evidence.movi_e.maximum_combined.noise_minus_clean_auroc, h.H6.evidence.movi_e.maximum_combined.paired_video_cluster_ci_low, h.H6.evidence.movi_e.maximum_combined.paired_video_cluster_ci_high, "Yes", "Large pose errors degrade performance"],
    ["Transfer minus in-domain D AUROC", transfer.paired_transfer_differences.in_domain_D.auroc.transfer_minus_reference, transfer.paired_transfer_differences.in_domain_D.auroc.paired_video_cluster_ci_low, transfer.paired_transfer_differences.in_domain_D.auroc.paired_video_cluster_ci_high, "No", "Ranking transfers nearly unchanged"],
    ["Transfer minus in-domain D false-match rate", transfer.paired_transfer_differences.in_domain_D.false_match_rate.comparison_minus_reference, transfer.paired_transfer_differences.in_domain_D.false_match_rate.paired_video_cluster_ci_low, transfer.paired_transfer_differences.in_domain_D.false_match_rate.paired_video_cluster_ci_high, "Yes", "The MOVi-D threshold accepts 6 more false matches per 1,000"],
  ];
  const mechanismStart = 62 + difficultRows.length;
  section(s, mechanismStart, "Mechanism, robustness, and transfer checks");
  table(s, mechanismStart + 1, ["Finding", "Estimate", "CI lower", "CI upper", "Excludes zero", "Interpretation"], mechanismRows, [48, 14, 14, 14, 16, 68]);
  s.getRange(`B${mechanismStart + 2}:D${mechanismStart + 1 + mechanismRows.length}`).format.numberFormat = "0.0000";

  const boundaryStart = mechanismStart + mechanismRows.length + 3;
  section(s, boundaryStart, "Failure review and interpretation boundary");
  const boundary = [
    "Failure gallery: 24 unique locked-threshold errors—12 MOVi-D, 12 MOVi-E; 12 false positives and 12 false negatives; 12 each for C and D.",
    "Recurring diagnoses: matched-category distractors, long-gap motion, occlusion/truncated masks, biased depth, camera misalignment, and pose-transform errors.",
    "Oracle geometry: simulator-provided masks, depth, intrinsics, and camera poses are privileged, unusually clean inputs.",
    "Fixed-camera limitation: MOVi-D is a synthetic control, not a paired counterfactual rendering or evidence about every static-camera setting.",
    "Scope: synthetic within-video matching only; real pose estimation, sensor noise, and cross-video re-identification remain untested.",
  ];
  boundary.forEach((text, index) => {
    const row = boundaryStart + 1 + index;
    s.getRange(`A${row}:I${row}`).merge(); s.getRange(`A${row}`).values = [[text]];
    s.getRange(`A${row}:I${row}`).format = { fill: gray, font: { color: ink }, wrapText: true, rowHeight: 24, verticalAlignment: "center" };
  });
  [40, 62, 14, 14, 14, 26, 18, 48, 22].forEach((width, index) => {
    const column = String.fromCharCode(65 + index);
    s.getRange(`${column}:${column}`).format.columnWidth = width;
  });
  s.getRange("A6:D11").format.rowHeight = 24;
  s.getRange("B5:B11").format.borders = { right: { style: "medium", color: "#AAB8C5" } };
  s.getRange("A15:I22").format.rowHeight = 30;
  s.getRange("A26:I33").format.rowHeight = 30;
  s.getRange("A37:I41").format.rowHeight = 26;
  s.getRange("A45:F48").format.rowHeight = 26;
  s.getRange("A52:I56").format.rowHeight = 26;
  s.getRange(`A${mechanismStart + 2}:F${mechanismStart + 1 + mechanismRows.length}`).format.rowHeight = 26;
  s.freezePanes.freezeRows(14);
}

{
  const s = wb.worksheets.add("Executive Summary"); title(s, "MOVi-D/E Camera-Pose Extension", "Locked confirmatory results, fixed-camera control, pose-noise study, transfer, and failure review", "H");
  const h = Object.fromEntries(phase9.criteria.map(x => [x.id, x]));
  const rows = [
    ["Primary: D - C on MOVi-E", h.H1.evidence.D_minus_C_AUROC, h.H1.evidence.ci_low, h.H1.evidence.ci_high, h.H1.status, "Pose alignment helps under moving camera"],
    ["Translation: high - low benefit", h.H3.evidence.MOVi_E_motion_strata.camera_displacement_scene_units.high_minus_low_D_minus_C_AUROC, h.H3.evidence.MOVi_E_motion_strata.camera_displacement_scene_units.paired_video_cluster_ci_low, h.H3.evidence.MOVi_E_motion_strata.camera_displacement_scene_units.paired_video_cluster_ci_high, h.H3.status, "Benefit rises with displacement"],
    ["Rotation: high - low benefit", h.H3.evidence.MOVi_E_motion_strata.relative_camera_rotation_degrees.high_minus_low_D_minus_C_AUROC, h.H3.evidence.MOVi_E_motion_strata.relative_camera_rotation_degrees.paired_video_cluster_ci_low, h.H3.evidence.MOVi_E_motion_strata.relative_camera_rotation_degrees.paired_video_cluster_ci_high, h.H3.status, "Benefit rises with rotation"],
    ["Fixed camera: D - C on MOVi-D", h.H4.evidence.D_minus_C, h.H4.evidence.paired_video_cluster_ci_low, h.H4.evidence.paired_video_cluster_ci_high, h.H4.status, "Consistent with falsification; not equivalence"],
    ["Transfer AUROC vs in-domain D", transfer.paired_transfer_differences.in_domain_D.auroc.transfer_minus_reference, transfer.paired_transfer_differences.in_domain_D.auroc.paired_video_cluster_ci_low, transfer.paired_transfer_differences.in_domain_D.auroc.paired_video_cluster_ci_high, "ranking_transfers", "Threshold calibration still degrades"],
  ];
  table(s, 4, ["Finding", "Estimate", "CI Low", "CI High", "Status", "Interpretation"], rows, [34, 14, 14, 14, 24, 42]);
  s.getRange("B5:D9").format.numberFormat = "0.000000";
  s.getRange("A11:H11").merge(); s.getRange("A11").values = [["Scope: oracle masks, depth, intrinsics, and camera pose on synthetic within-video pairs. These results do not establish deployment performance with estimated geometry or real cameras."]];
  s.getRange("A11:H11").format = { fill: amber, font: { bold: true, color: ink }, wrapText: true, rowHeight: 44 };
}

{
  const s = wb.worksheets.add("System Metrics"); title(s, "All Clean Systems", "Development and locked-test metrics for baselines A/B, geometry systems C/D, and diagnostic controls G/P/S", "N");
  const rows = [];
  for (const [dataset, data] of [["MOVi-E", e], ["MOVi-D", d]]) for (const [system, splits] of Object.entries(data.aggregate)) for (const split of ["dev", "test"]) {
    const m = splits[split]; rows.push([dataset, system, split, m.auroc, m.pr_auc, m.false_match_rate_at_locked_90_recall_threshold, m.recall_at_locked_90_recall_threshold, m.f1_at_locked_max_f1_threshold, m.precision_at_locked_max_f1_threshold, m.recall_at_locked_max_f1_threshold, m.calibration.brier_score, m.calibration.log_loss, m.calibration.expected_calibration_error_10_bins]);
  }
  table(s, 4, ["Dataset", "System", "Split", "AUROC", "PR-AUC", "FMR @ locked 90% recall", "Recall @ locked threshold", "F1 @ locked max-F1", "Precision @ max-F1", "Recall @ max-F1", "Brier", "Log loss", "ECE (10 bins)"], rows, [12, 30, 10, 12, 12, 18, 18, 17, 17, 17, 12, 12, 14]);
  s.getRange(`D5:M${4 + rows.length}`).format.numberFormat = "0.000000";
}

{
  const s = wb.worksheets.add("Paired Confidence Intervals"); title(s, "Paired Test Differences", "System minus C and system minus D on identical test pairs; 10,000 video-cluster bootstrap replicates", "I");
  const rows = [];
  for (const [dataset, data] of [["MOVi-E", e], ["MOVi-D", d]]) for (const [ref, group] of [["C", data.paired_differences_vs_C], ["D", data.paired_differences_vs_D]]) for (const [system, metrics] of Object.entries(group)) for (const [metric, obj] of Object.entries(metrics)) {
    if (typeof obj !== "object" || obj === null) continue;
    const est = obj.system_minus_C ?? obj.system_minus_D ?? obj.comparison_minus_reference ?? null;
    rows.push([dataset, system, ref, metric, est, obj.paired_video_cluster_ci_low ?? null, obj.paired_video_cluster_ci_high ?? null, obj.bootstrap_replicates ?? 10000]);
  }
  table(s, 4, ["Dataset", "System", "Reference", "Metric", "Difference", "CI Low", "CI High", "Replicates"], rows, [12, 30, 12, 20, 14, 14, 14, 12]);
  s.getRange(`E5:G${4 + rows.length}`).format.numberFormat = "0.000000";
}

{
  const s = wb.worksheets.add("Motion Strata"); title(s, "MOVi-E Motion Strata", "D minus C AUROC using tertiles frozen from the MOVi-E training pool", "J");
  const motion = phase9.criteria.find(x => x.id === "H3").evidence.MOVi_E_motion_strata; const rows = [];
  for (const [measure, obj] of Object.entries(motion)) for (const [stratum, v] of Object.entries(obj.strata)) rows.push([measure, stratum, v.pairs, v.positives, v.negatives, v.D_minus_C_AUROC, v.paired_video_cluster_ci_low, v.paired_video_cluster_ci_high, obj.train_tertile_cutoffs[0], obj.train_tertile_cutoffs[1]]);
  table(s, 4, ["Motion measure", "Stratum", "Pairs", "Positive", "Negative", "D - C AUROC", "CI Low", "CI High", "Train cut 1", "Train cut 2"], rows, [35, 12, 10, 10, 10, 16, 14, 14, 14, 14]);
  s.getRange(`F5:J${4 + rows.length}`).format.numberFormat = "0.000000";
}

{
  const s = wb.worksheets.add("Pose Noise"); title(s, "Phase 7 Pose-Noise Study", "All frozen noisy-pose conditions and reported test metrics", "Z");
  const headers = Object.keys(poseRows[0]); const rows = poseRows.map(r => headers.map(h => typed(r[h])));
  table(s, 4, headers, rows, headers.map(h => Math.min(28, Math.max(12, h.length + 2))));
}

{
  const s = wb.worksheets.add("Transfer"); title(s, "MOVi-D to MOVi-E Transfer", "Unchanged MOVi-D clean-D scaler, model, and development thresholds applied to locked MOVi-E test pairs", "L");
  const blocks = [["Transfer D", transfer.transfer_test], ["In-domain E C", transfer.in_domain_MOVi_E_benchmarks.in_domain_C], ["In-domain E D", transfer.in_domain_MOVi_E_benchmarks.in_domain_D]];
  const rows = blocks.map(([name,m]) => [name,m.auroc,m.pr_auc,m.false_match_rate_at_locked_90_recall_threshold,m.recall_at_locked_90_recall_threshold,m.f1_at_locked_max_f1_threshold,m.precision_at_locked_max_f1_threshold,m.recall_at_locked_max_f1_threshold,m.calibration.brier_score,m.calibration.log_loss,m.calibration.expected_calibration_error_10_bins]);
  table(s, 4, ["Regime", "AUROC", "PR-AUC", "FMR", "Recall", "F1", "Precision", "Recall @ max-F1", "Brier", "Log loss", "ECE"], rows, [22,12,12,12,12,12,12,16,12,12,12]);
  s.getRange("B5:K7").format.numberFormat = "0.000000";
  const diffs=[]; for (const [ref,metrics] of Object.entries(transfer.paired_transfer_differences)) for (const [metric,o] of Object.entries(metrics)) diffs.push([ref,metric,o.transfer_minus_reference ?? o.comparison_minus_reference,o.paired_video_cluster_ci_low,o.paired_video_cluster_ci_high,o.bootstrap_replicates]);
  table(s, 10, ["Reference", "Metric", "Transfer - reference", "CI Low", "CI High", "Replicates"], diffs, [20,18,20,14,14,12]);
  s.getRange(`C11:E${10+diffs.length}`).format.numberFormat = "0.000000";
}

{
  const s = wb.worksheets.add("Latency"); title(s, "Measured Latency Components", "Observed timings; hardware/load dependent. Scoring excludes feature extraction and RGB encoding.", "H");
  const rows=[]; for (const [dataset,data] of [["MOVi-E",e],["MOVi-D",d]]) for (const [system,l] of Object.entries(data.scoring_latency)) rows.push([dataset,"logistic scoring",system,"microseconds/pair",l.microseconds_per_pair_p50,l.microseconds_per_pair_p95,l.batch_size,l.repetitions]);
  for (const [dataset,m] of [["MOVi-E",eEmbed.latency],["MOVi-D",dEmbed.latency]]) rows.push([dataset,"ResNet-18 forward","all crops","milliseconds/crop",m.forward_ms_per_crop_p50,m.forward_ms_per_crop_p95,null,null],[dataset,"decode + preprocess + forward","all crops","milliseconds/crop",m.wall_ms_per_crop,null,null,null]);
  rows.push(["MOVi-E","transfer D scoring","D","microseconds/pair",transfer.scoring_latency.microseconds_per_pair_single_pass,null,transfer.scoring_latency.batch_size,1]);
  table(s,4,["Dataset","Component","System/scope","Unit","P50 / observed","P95","Batch","Repetitions"],rows,[12,28,28,20,18,16,12,12]);
  s.getRange(`E5:F${4+rows.length}`).format.numberFormat = "0.000000";
}

{
  const s = wb.worksheets.add("Failure Analysis"); title(s, "Phase 10 Failure Review", "24 unique fixed-threshold misclassifications balanced across datasets, systems, error types, and feasible motion strata", "J");
  const headers = Object.keys(reviewRows[0]); const rows = reviewRows.map(r => headers.map(h => typed(r[h])));
  table(s,4,headers,rows,headers.map(h => h.includes("diagnosis") ? 46 : Math.min(24,Math.max(12,h.length+2))));
}

{
  const s = wb.worksheets.add("All Pair Strata"); title(s, "All Predeclared Pair Strata", "All eight clean systems on locked test pairs; continuous boundaries derived from training pairs only", "N");
  const headers = Object.keys(strataRows[0]); const rows = strataRows.map(r => headers.map(h => typed(r[h])));
  table(s,4,headers,rows,[12,36,25,36,10,10,10,14,14,18,14,27]);
  s.getRange(`A5:L${4 + rows.length}`).format.wrapText = false;
}

{
  const s = wb.worksheets.add("Strata D-C Intervals"); title(s, "Paired D-minus-C Stratum Intervals", "AUROC for mixed-label strata; recall for positive-only strata; false-match rate for negative-only strata", "M");
  const headers = Object.keys(strataCiRows[0]); const rows = strataCiRows.map(r => headers.map(h => typed(r[h])));
  table(s,4,headers,rows,[12,38,25,20,16,29,30,10,10,10,18,18]);
  s.getRange(`A5:L${4 + rows.length}`).format.wrapText = false;
}

{
  const s = wb.worksheets.add("Provenance"); title(s, "Workbook Provenance", "Canonical machine-readable sources used to build this workbook", "D");
  const rows = [
    ["In-domain MOVi-E", "results/movi_de_phase8_regime1/movi_e_in_domain_results.json", "Aggregate metrics, paired CIs, scoring latency", "locked"],
    ["In-domain MOVi-D", "results/movi_de_phase8_regime2/movi_d_in_domain_results.json", "Aggregate metrics, paired CIs, scoring latency", "locked"],
    ["D to E transfer", "results/movi_de_phase8_regime3/movi_d_to_e_transfer_results.json", "Transfer metrics and paired differences", "locked"],
    ["Phase 9 evaluation", "results/movi_de_phase9/phase9_criteria_evaluation.json", "Hypotheses and motion strata", "locked"],
    ["Pose noise", "results/movi_de_phase7_pose_noise/phase7_pose_noise_results_table.csv", "All noise conditions", "locked"],
    ["Failure review", "failure_gallery/movi_de_phase10/failure_review.csv", "24 selected errors", "locked"],
    ["All pair strata", "results/movi_de_final/pair_strata_results.csv", "Eight systems across every predeclared stratum", "descriptive release reporting"],
    ["Stratum intervals", "results/movi_de_final/pair_strata_d_minus_c_intervals.csv", "Paired D-minus-C video-cluster intervals", "descriptive release reporting"],
    ["Presentation intervals", "results/movi_de_final/presentation_table_metrics.json", "Paired PR-AUC and locked-F1 intervals for presentation parity", "descriptive release reporting"],
    ["Dependency/source terms", "docs/SOURCES_AND_LICENSES.md", "Data, model, dependency sources and licenses", "release documentation"],
  ];
  table(s,4,["Source block","Repository path","Use","Status"],rows,[24,65,48,22]);
}

for (let i=0;i<wb.worksheets.items.length;i++) {
  const s=wb.worksheets.getItemAt(i); const used=s.getUsedRange(); used.format.font = { name: "Aptos", size: 10 };
  s.getRange("A1:Z1").format.font = { name: "Aptos", bold: true, color: "#FFFFFF", size: 16 };
  s.getRange("A2:Z2").format.font = { name: "Aptos", italic: true, color: ink, size: 10 };
}

await fs.mkdir(path.dirname(out), {recursive:true}); await fs.mkdir(previewDir,{recursive:true});
for (let i=0;i<wb.worksheets.items.length;i++) {
  const s=wb.worksheets.getItemAt(i); const blob=await wb.render({sheetName:s.name,autoCrop:"all",scale:1,format:"png"});
  await fs.writeFile(path.join(previewDir,`${String(i+1).padStart(2,"0")}_${s.name.replaceAll(" ","_")}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const check=await wb.inspect({kind:"table",range:"'Presentation Table'!A1:I95",include:"values,formulas",tableMaxRows:100,tableMaxCols:10,maxChars:22000});
console.log(check.ndjson);
const errors=await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",options:{useRegex:true,maxResults:100},summary:"final formula error scan"});
console.log(errors.ndjson);
const file=await SpreadsheetFile.exportXlsx(wb); await file.save(out);
