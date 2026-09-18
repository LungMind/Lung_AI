"""MIMIC-IV Dataset Inspection Script.

Systematically inspects all available tabular files in the root and data directories:
- admissions.csv
- diagnoses_icd.csv
- microbiologyevents.csv
- procedures_icd.csv
- services.csv

Determines:
1. File sizes, row counts, columns, missing values, sample rows
2. Unique subject_id (patients) and hadm_id (admissions)
3. Whether free-text clinical notes (discharge summaries, radiology reports, physician notes) exist
4. Whether diagnosis information (ICD-9 / ICD-10) exists
5. Whether note-to-diagnosis linkage is possible
6. Data provenance: MIMIC-IV clinical modules vs MIMIC-IV-Note
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Project root anchor
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Files to inspect
TARGET_FILES = [
    "admissions.csv",
    "diagnoses_icd.csv",
    "microbiologyevents.csv",
    "procedures_icd.csv",
    "services.csv",
]


def format_size(bytes_size):
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def inspect_file(filepath):
    """Inspect a single CSV file with memory-efficient chunking."""
    path = Path(filepath)
    if not path.exists():
        return None

    file_size = path.stat().st_size
    formatted_size = format_size(file_size)
    print("\n" + "=" * 80)
    print(f"INSPECTING: {path.name} ({formatted_size})")
    print("=" * 80)

    # Read header and first few rows
    df_sample = pd.read_csv(path, nrows=3)
    columns = list(df_sample.columns)
    print(f"Columns ({len(columns)}): {columns}")

    # Chunked pass for total rows, missing values, and unique IDs
    total_rows = 0
    missing_counts = {col: 0 for col in columns}
    unique_subjects = set()
    unique_hadms = set()
    max_text_lengths = {col: 0 for col in columns}

    has_subject_id = "subject_id" in columns
    has_hadm_id = "hadm_id" in columns

    # Process in chunks of 100,000 to remain memory-safe
    chunk_size = 100000
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)
        for col in columns:
            missing_counts[col] += int(chunk[col].isna().sum())
            if chunk[col].dtype == object:
                non_null_series = chunk[col].dropna().astype(str)
                if len(non_null_series) > 0:
                    chunk_max_len = non_null_series.str.len().max()
                    if chunk_max_len > max_text_lengths[col]:
                        max_text_lengths[col] = int(chunk_max_len)

        if has_subject_id:
            unique_subjects.update(chunk["subject_id"].dropna().unique())
        if has_hadm_id:
            unique_hadms.update(chunk["hadm_id"].dropna().unique())

    print(f"Total Rows: {total_rows:,}")
    if has_subject_id:
        print(f"Unique subject_id (patients): {len(unique_subjects):,}")
    if has_hadm_id:
        print(f"Unique hadm_id (admissions): {len(unique_hadms):,}")

    print("\nMissing Values per Column:")
    for col in columns:
        pct = (missing_counts[col] / total_rows * 100) if total_rows > 0 else 0
        max_len = max_text_lengths.get(col, 0)
        print(f"  - {col:<25s}: {missing_counts[col]:>10,d} missing ({pct:>5.1f}%) | Max Str Len: {max_len}")

    print("\nFirst 3 Sample Rows (preview):")
    # De-identify / mask if any string looks like narrative
    for idx, row in df_sample.iterrows():
        print(f"  Row {idx + 1}:")
        for col in columns:
            val_str = str(row[col])
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            print(f"    {col}: {val_str}")

    return {
        "filename": path.name,
        "file_size_bytes": file_size,
        "file_size_formatted": formatted_size,
        "total_rows": total_rows,
        "columns": columns,
        "missing_counts": missing_counts,
        "unique_patients": len(unique_subjects) if has_subject_id else 0,
        "unique_admissions": len(unique_hadms) if has_hadm_id else 0,
        "max_text_lengths": max_text_lengths,
    }


def search_for_clinical_text(file_summaries):
    """Analyze all columns across all tables to identify potential free-text clinical notes."""
    text_indicators = [
        "text", "note", "clinical", "report", "discharge", "radiology",
        "impression", "history", "findings", "narrative", "free_text", "long_text"
    ]

    print("\n" + "=" * 80)
    print("PHASE 2 ANALYSIS: SEARCHING FOR CLINICAL FREE-TEXT NOTES")
    print("=" * 80)

    found_text_columns = []
    found_diagnosis = False

    for summary in file_summaries:
        fname = summary["filename"]
        cols = summary["columns"]
        max_lens = summary["max_text_lengths"]

        if "icd_code" in cols or "icd_version" in cols:
            found_diagnosis = True

        for col in cols:
            col_lower = col.lower()
            max_len = max_lens.get(col, 0)

            # Check for name match or long string length
            is_text_name = any(ind in col_lower for ind in text_indicators)
            is_long_text = max_len > 250  # Narrative reports typically have hundreds/thousands of chars

            if is_text_name or is_long_text:
                found_text_columns.append(
                    {
                        "file": fname,
                        "column": col,
                        "max_length": max_len,
                        "is_name_match": is_text_name,
                        "is_long_text": is_long_text,
                    }
                )

    if found_text_columns:
        print("Candidate text fields found:")
        for c in found_text_columns:
            print(f"  - File: {c['file']}, Column: {c['column']}, Max Length: {c['max_length']} chars")
    else:
        print("No candidate free-text fields or long text columns found in any table.")

    # Clinical notes in MIMIC-IV are housed in MIMIC-IV-Note (discharge.csv.gz, radiology.csv.gz)
    # Structured tables like admissions, diagnoses_icd, etc. contain short categorical codes.
    has_free_text = any(c["max_length"] > 300 for c in found_text_columns)

    return has_free_text, found_diagnosis, found_text_columns


def main():
    print("=" * 80)
    print("MIMIC-IV DATASET INSPECTION & PROVENANCE AUDIT")
    print("=" * 80)
    print(f"Inspecting directory: {PROJECT_ROOT}\n")

    results = []
    for fname in TARGET_FILES:
        fpath = PROJECT_ROOT / fname
        if fpath.exists():
            summary = inspect_file(fpath)
            if summary:
                results.append(summary)
        else:
            print(f"\n[WARNING] Expected file not found: {fname}")

    # Check for any other files in project root or data/
    all_root_files = [p.name for p in PROJECT_ROOT.glob("*.csv")]
    extra_files = [f for f in all_root_files if f not in TARGET_FILES]
    if extra_files:
        print(f"\nAdditional CSV files found in workspace: {extra_files}")
        for ef in extra_files:
            summary = inspect_file(PROJECT_ROOT / ef)
            if summary:
                results.append(summary)

    # Check for clinical text
    has_text, has_diag, candidate_cols = search_for_clinical_text(results)

    # Summary determinations
    has_notes = has_text
    note_to_diag_possible = has_notes and has_diag

    print("\n" + "=" * 80)
    print("DETERMINATION SUMMARY:")
    print("=" * 80)
    print(f"CLINICAL TEXT AVAILABLE:       {'YES' if has_notes else 'NO'}")
    print(f"DIAGNOSIS DATA AVAILABLE:      {'YES' if has_diag else 'NO'}")
    print(f"NOTE-TO-DIAGNOSIS LINK POSSIBLE: {'YES' if note_to_diag_possible else 'NO'}")
    print("=" * 80)

    if not has_notes:
        print("\n" + "!" * 80)
        print("CRITICAL FINDING: CLINICAL FREE-TEXT NOTES ARE NOT PRESENT")
        print("!" * 80)
        print("Reason:")
        print("- The available tables (admissions, diagnoses_icd, microbiologyevents,")
        print("  procedures_icd, services) represent the STRUCTURED clinical modules of MIMIC-IV.")
        print("- In the MIMIC-IV schema:")
        print("  * 'hosp' module contains structured tabular records (admissions, diagnoses, etc.)")
        print("  * 'note' module (MIMIC-IV-Note) is a SEPARATE dataset containing:")
        print("    1. discharge.csv (Discharge summaries - full free-text doctor reports)")
        print("    2. radiology.csv (Radiology reports - free-text imaging impressions)")
        print("- Free-text clinical notes are required by the reference paper architecture")
        print("  to extract text embeddings, 1D convolutions, and sequence attention.")
        print("- None of the currently available tables contain clinical narrative reports.")
        print("!" * 80)

    # Save inspection output
    output_dir = PROJECT_ROOT / "text_model" / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "dataset_inspection.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "files_inspected": results,
                "clinical_text_available": has_notes,
                "diagnosis_data_available": has_diag,
                "note_to_diagnosis_link_possible": note_to_diag_possible,
                "candidate_text_columns": candidate_cols,
            },
            f,
            indent=2,
        )
    print(f"\n[OUTPUT] Inspection results serialized to: {out_file}")


if __name__ == "__main__":
    main()
