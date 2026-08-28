"""
Task 1: Repository Inventory Scanner for IO-VNBD.
Recursively inspects all files, folders, LFS metadata, categories, drivers, and synchronization status.
Outputs inventory to CSV and JSON.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_lfs_info(file_path: Path) -> Dict[str, Any]:
    """Reads LFS metadata if file is an LFS pointer."""
    info = {"is_lfs": False, "lfs_oid": None, "lfs_size_bytes": None}
    if not file_path.exists() or file_path.stat().st_size > 1024:
        return info

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        oid_m = re.search(r"oid sha256:([a-f0-9]+)", content)
        size_m = re.search(r"size (\d+)", content)
        if oid_m and size_m:
            info["is_lfs"] = True
            info["lfs_oid"] = oid_m.group(1)
            info["lfs_size_bytes"] = int(size_m.group(1))
    except Exception:
        pass
    return info


def classify_file(rel_path: str, filename: str) -> Dict[str, str]:
    """Classifies dataset domain, driver, sync status, and category."""
    path_lower = rel_path.lower()
    fn_lower = filename.lower()

    # Domain
    if fn_lower.startswith("s-") or "/s-dataset/" in path_lower or "\\s-dataset\\" in path_lower:
        domain = "Smartphone (S)"
    elif fn_lower.startswith("v-") or "/v-dataset/" in path_lower or "\\v-dataset\\" in path_lower:
        domain = "Vehicle (V)"
    elif filename.endswith(".pdf"):
        domain = "Documentation (Paper)"
    elif filename.endswith(".md"):
        domain = "Documentation (Readme)"
    elif filename.endswith(".zip"):
        domain = "Archive"
    else:
        domain = "Other / Metadata"

    # Synchronization
    if "synchronised" in path_lower or "synchronized" in path_lower:
        sync_status = "Synchronised"
    elif "unsynchronised" in path_lower or "unsynchronized" in path_lower:
        sync_status = "Unsynchronised"
    else:
        sync_status = "N/A"

    # Categorization
    if "uncategorised" in path_lower or "uncategorized" in path_lower:
        cat_status = "Uncategorised"
    elif "categorised" in path_lower or "categorized" in path_lower:
        cat_status = "Categorised"
    else:
        cat_status = "N/A"

    # Driver identification
    driver = "Unknown"
    driver_match = re.search(r"driver\s+([a-h])", path_lower)
    if driver_match:
        driver = f"Driver {driver_match.group(1).upper()}"
    else:
        # Check based on paper prefix mapping
        if fn_lower.startswith("s-s") or fn_lower.startswith("v-s"):
            driver = "Driver A (UK)"
        elif fn_lower.startswith("s-m") or fn_lower.startswith("v-m"):
            driver = "Driver B (UK)"
        elif fn_lower.startswith("s-y") or fn_lower.startswith("v-y"):
            driver = "Driver D (UK)"
        elif any(fn_lower.startswith(p) for p in ["s-vta", "v-vta", "s-vtb", "v-vtb", "s-vw", "v-vw", "s-vfa", "v-vfa"]):
            driver = "Driver E (UK)"
        elif fn_lower.startswith("s-t"):
            driver = "Driver F/G/H (France/Nigeria)"

    return {
        "domain": domain,
        "sync_status": sync_status,
        "cat_status": cat_status,
        "driver": driver,
    }


def scan_repository(repo_root: str = "IO-VNBD-master") -> pd.DataFrame:
    """Recursively scans repository and produces inventory DataFrame."""
    root_path = Path(repo_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")

    records = []
    for file_path in root_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(root_path).as_posix()
            filename = file_path.name
            ext = file_path.suffix.lower()
            disk_size = file_path.stat().st_size

            lfs_info = extract_lfs_info(file_path)
            classification = classify_file(rel_path, filename)
            real_size_bytes = lfs_info["lfs_size_bytes"] if lfs_info["is_lfs"] else disk_size

            records.append({
                "relative_path": rel_path,
                "filename": filename,
                "extension": ext,
                "parent_dir": file_path.parent.name,
                "disk_size_bytes": disk_size,
                "real_size_bytes": real_size_bytes,
                "is_lfs_pointer": lfs_info["is_lfs"],
                "lfs_oid": lfs_info["lfs_oid"],
                "domain": classification["domain"],
                "sync_status": classification["sync_status"],
                "cat_status": classification["cat_status"],
                "driver": classification["driver"],
            })

    df = pd.DataFrame(records)
    return df


def generate_inventory(
    repo_root: str = "IO-VNBD-master",
    output_dir: str = "Data_details/outputs/inventory",
) -> pd.DataFrame:
    """Generates and saves inventory reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_repository(repo_root)

    csv_path = out_dir / "dataset_inventory.csv"
    json_path = out_dir / "dataset_inventory.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    logger.info(f"Inventory saved to {csv_path} and {json_path}")

    # Summary statistics
    total_files = len(df)
    csv_files = len(df[df["extension"] == ".csv"])
    s_files = len(df[df["domain"] == "Smartphone (S)"])
    v_files = len(df[df["domain"] == "Vehicle (V)"])
    sync_files = len(df[df["sync_status"] == "Synchronised"])
    total_data_gb = df["real_size_bytes"].sum() / (1024**3)

    print("\n=======================================================")
    print("           IO-VNBD DATASET REPOSITORY INVENTORY        ")
    print("=======================================================")
    print(f"Total files scanned:        {total_files}")
    print(f"Total CSV data files:       {csv_files}")
    print(f"Smartphone (S) datasets:    {s_files}")
    print(f"Vehicle (V) datasets:       {v_files}")
    print(f"Synchronised data files:    {sync_files}")
    print(f"Total uncompressed volume:  ~{total_data_gb:.2f} GB")
    print("-------------------------------------------------------")
    print("Files by Domain:")
    print(df["domain"].value_counts().to_string())
    print("-------------------------------------------------------")
    print("Files by Driver / Region:")
    print(df[df["extension"] == ".csv"]["driver"].value_counts().to_string())
    print("=======================================================\n")

    return df


if __name__ == "__main__":
    generate_inventory()
