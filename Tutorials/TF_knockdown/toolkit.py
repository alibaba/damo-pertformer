import os
import io
import json
import math
import random
import warnings
import tempfile
import subprocess
import contextlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import pyBigWig
from einops import rearrange

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# General helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_tf_deg_filename(csv_path: str):
    """
    Example:
      /path/K562_RELA_up.csv -> (K562, RELA, up)
    """
    base = os.path.basename(csv_path)
    if not base.endswith(".csv"):
        return None

    name = base[:-4]
    parts = name.split("_")
    if len(parts) < 3:
        return None

    cell_line = parts[0]
    direction = parts[-1]
    tf_name = "_".join(parts[1:-1])
    return cell_line, tf_name, direction


def load_deg_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def sample_deg_rows(
    up_df: pd.DataFrame,
    down_df: pd.DataFrame,
    max_up: Optional[int] = None,
    max_down: Optional[int] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    def _sample_df(df, n):
        if n is None or n >= len(df):
            return df.copy()
        idx = rng.choice(len(df), size=n, replace=False)
        idx = sorted(idx.tolist())
        return df.iloc[idx].copy()

    up_sub = _sample_df(up_df, max_up)
    down_sub = _sample_df(down_df, max_down)

    merged = pd.concat([up_sub, down_sub], axis=0, ignore_index=True)
    return merged


def build_gene_to_deg_info(df: pd.DataFrame) -> Dict[str, dict]:
    gene_to_info = {}
    for _, row in df.iterrows():
        gene = str(row.get("Gene", "")).strip()
        if not gene:
            continue
        if gene not in gene_to_info:
            gene_to_info[gene] = row.to_dict()
    return gene_to_info


def txt_contains_n(txt_path: str) -> bool:
    with open(txt_path, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    for i in range(0, len(lines), 3):
        if i + 1 < len(lines):
            seq = lines[i + 1].strip().upper()
            if "N" in seq:
                return True
    return False


def load_txt_entries(txt_path: str) -> List[Tuple[str, str, str]]:
    with open(txt_path, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    entries = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            entries.append((lines[i], lines[i + 1], lines[i + 2]))
    return entries


# ============================================================
# Gene / sequence / signal preprocessing
# ============================================================
class GeneATACPreprocessor:
    def __init__(self, gtf_path: str, genome_dir: str, peak_path: str, bigwig_path: str):
        print("[1/4] Loading GTF index...")
        self.gene_index = self._parse_gtf(gtf_path)
        print(f"  Loaded {len(self.gene_index)} genes from GTF.")

        print("[2/4] Loading chromosome tensors...")
        self.genome = [None] * 24
        for i in range(22):
            self.genome[i] = torch.load(f"{genome_dir}/chr{i+1}.pkl")
        self.genome[22] = torch.load(f"{genome_dir}/chrX.pkl")
        self.genome[23] = torch.load(f"{genome_dir}/chrY.pkl")

        self.chr_to_idx = {f"chr{i+1}": i for i in range(22)}
        self.chr_to_idx["chrX"] = 22
        self.chr_to_idx["chrY"] = 23
        print("  Genome tensors loaded.")

        print("[3/4] Loading and deduplicating peaks...")
        self.peaks = self._load_and_dedup_peaks(peak_path)
        total_peaks = sum(len(v) for v in self.peaks.values())
        print(f"  Loaded {total_peaks} unique peaks.")

        print("[4/4] Opening bigWig...")
        self.bw = pyBigWig.open(bigwig_path)
        print("  bigWig opened successfully.\n")

    def _parse_gtf(self, gtf_path: str) -> Dict[str, dict]:
        gene_info = {}
        with open(gtf_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9 or fields[2] != "gene":
                    continue

                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]

                gene_name = None
                for attr in fields[8].split(";"):
                    attr = attr.strip()
                    if attr.startswith("gene_name"):
                        gene_name = attr.split('"')[1]
                        break

                if gene_name is None:
                    continue

                tss = start if strand == "+" else end
                gene_info[gene_name] = {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "tss": tss
                }
        return gene_info

    def _load_and_dedup_peaks(self, peak_path: str) -> Dict[str, List[Tuple[int, int, str]]]:
        raw_peaks = {}
        seen = set()
        with open(peak_path, "r") as f:
            for line in f:
                fields = line.strip().split("\t")
                chrom, start, end, peak_id = fields[0], int(fields[1]), int(fields[2]), fields[3]
                key = (chrom, start, end)
                if key not in seen:
                    seen.add(key)
                    raw_peaks.setdefault(chrom, []).append((start, end, peak_id))

        for chrom in raw_peaks:
            raw_peaks[chrom].sort(key=lambda x: x[0])
        return raw_peaks

    def _get_chr_tensor(self, chrom: str) -> Optional[torch.Tensor]:
        idx = self.chr_to_idx.get(chrom)
        return self.genome[idx] if idx is not None else None

    def _extract_sequence(self, chrom: str, center: int, length: int = 1029) -> Optional[str]:
        chr_tensor = self._get_chr_tensor(chrom)
        if chr_tensor is None:
            return None

        half = length // 2
        start_0 = center - 1 - half
        end_0 = start_0 + length
        chr_len = len(chr_tensor)

        if start_0 < 0 or end_0 > chr_len:
            return None

        seq_tensor = chr_tensor[start_0:end_0]
        base_map = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

        try:
            seq = "".join([base_map[int(b)] for b in seq_tensor]).upper()
        except Exception:
            seq = "".join([str(b) for b in seq_tensor]).upper()

        return seq

    def _extract_signal(self, chrom: str, center: int, length: int = 1029) -> Optional[List[float]]:
        half = length // 2
        start_0 = center - 1 - half
        end_0 = start_0 + length
        try:
            values = self.bw.values(chrom, start_0, end_0)
            if values is None:
                return [0.0] * length
            return [v if v is not None and not np.isnan(v) else 0.0 for v in values]
        except Exception:
            return [0.0] * length

    def process_gene(self, gene_name: str, output_txt: Optional[str] = None, window: int = 150000):
        if gene_name not in self.gene_index:
            return None

        gene = self.gene_index[gene_name]
        chrom, tss, strand = gene["chrom"], gene["tss"], gene["strand"]
        region_start, region_end = tss - window, tss + window

        chr_peaks = self.peaks.get(chrom, [])
        selected_peaks = [(s, e, pid) for s, e, pid in chr_peaks if s >= region_start and e <= region_end]

        if len(selected_peaks) == 0:
            return None

        min_dist, closest_idx = float("inf"), 0
        for i, (s, e, _) in enumerate(selected_peaks):
            dist = abs((s + e) / 2.0 - tss)
            if dist < min_dist:
                min_dist, closest_idx = dist, i

        sequences, signals, peak_coords, ids = [], [], [], []
        for p_start, p_end, p_id in selected_peaks:
            center = (p_start + p_end) // 2
            seq = self._extract_sequence(chrom, center, 1029)
            sig = self._extract_signal(chrom, center, 1029)
            if seq is None or sig is None:
                continue
            sequences.append(seq)
            signals.append(sig)
            peak_coords.append((p_start, p_end))
            ids.append(p_id)

        valid_n = len(ids)
        if valid_n == 0:
            return None

        if output_txt:
            with open(output_txt, "w") as f:
                for idx in range(valid_n):
                    p_start, p_end = peak_coords[idx]
                    f.write(f">{chrom}_{p_start}_{p_end}\n")
                    f.write(f"{sequences[idx]}\n")
                    f.write("\t".join([str(v) for v in signals[idx]]) + "\n")

        return {
            "gene_name": gene_name,
            "chrom": chrom,
            "tss": tss,
            "strand": strand,
            "closest_peak_index": closest_idx,
            "n_peaks": valid_n,
            "ids": ids,
            "peak_coords": peak_coords,
            "sequences": sequences,
            "signals": signals
        }

    def close(self):
        if self.bw is not None:
            self.bw.close()


# ============================================================
# Motif scanning
# ============================================================
class FIMOMotifScanner:
    def __init__(
        self,
        motif_dir: str,
        strong_pvalue_thresh: float = 1e-5,
        fallback_pvalue_thresh: float = 1e-4,
        fimo_path: str = "fimo"
    ):
        self.motif_dir = motif_dir
        self.strong_pvalue_thresh = strong_pvalue_thresh
        self.fallback_pvalue_thresh = fallback_pvalue_thresh
        self.fimo_path = fimo_path

    def get_motif_path(self, tf_name: str) -> Optional[str]:
        motif_path = os.path.join(self.motif_dir, f"{tf_name}.meme")
        if os.path.exists(motif_path):
            return motif_path
        return None

    def scan_sequences(self, tf_name: str, sequence_dict: Dict[str, str]) -> Optional[Dict[str, List[dict]]]:
        motif_path = self.get_motif_path(tf_name)
        if motif_path is None:
            return None

        if len(sequence_dict) == 0:
            return {}

        with tempfile.TemporaryDirectory(prefix="fimo_tmp_") as tmpdir:
            fasta_path = os.path.join(tmpdir, "input.fa")
            out_dir = os.path.join(tmpdir, "fimo_out")

            with open(fasta_path, "w") as f:
                for seq_name, seq in sequence_dict.items():
                    f.write(f">{seq_name}\n{seq}\n")

            cmd = [
                self.fimo_path,
                "--verbosity", "1",
                "--thresh", str(self.fallback_pvalue_thresh),
                "--oc", out_dir,
                motif_path,
                fasta_path
            ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True
                )
            except FileNotFoundError:
                raise RuntimeError(f"FIMO executable not found: {self.fimo_path}")
            except subprocess.CalledProcessError:
                return {}

            fimo_tsv = os.path.join(out_dir, "fimo.tsv")
            if not os.path.exists(fimo_tsv):
                return {}

            hits_by_seq = {k: [] for k in sequence_dict.keys()}

            with open(fimo_tsv, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    line = line.rstrip("\n")
                    if not line:
                        continue

                    fields = line.split("\t")
                    if len(fields) < 10:
                        continue

                    try:
                        sequence_name = fields[2]
                        start = int(fields[3])
                        stop = int(fields[4])
                        strand = fields[5]
                        p_value = float(fields[7])
                        matched_sequence = fields[9]
                    except Exception:
                        continue

                    if p_value <= self.fallback_pvalue_thresh:
                        hits_by_seq.setdefault(sequence_name, []).append({
                            "start": start,
                            "stop": stop,
                            "strand": strand,
                            "p_value": p_value,
                            "matched_sequence": matched_sequence
                        })

            return hits_by_seq


# ============================================================
# WT / KO txt generation
# ============================================================
def build_wild_type_inputs_for_genes(
    preprocessor: GeneATACPreprocessor,
    gene_list: List[str],
    wild_type_dir: str,
    verbose_every: int = 20,
    suppress_inner_print: bool = True
) -> str:
    ensure_dir(wild_type_dir)
    wt_meta_path = os.path.join(wild_type_dir, "meta_info.json")
    wt_meta_records = []

    print(f"Building WT inputs for {len(gene_list)} genes...")
    success_n = 0
    failed_n = 0
    existing_n = 0

    for idx, gene_name in enumerate(gene_list):
        output_txt = os.path.join(wild_type_dir, f"{gene_name}.txt")

        if os.path.exists(output_txt):
            existing_n += 1
            wt_meta_records.append({
                "gene_name": gene_name,
                "chrom": None,
                "tss": None,
                "strand": None,
                "closest_peak_index": None,
                "n_peaks": None,
                "status": "already_exists"
            })
            continue

        try:
            if suppress_inner_print:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = preprocessor.process_gene(gene_name=gene_name, output_txt=output_txt)
            else:
                result = preprocessor.process_gene(gene_name=gene_name, output_txt=output_txt)

            if result is not None:
                success_n += 1
                wt_meta_records.append({
                    "gene_name": result["gene_name"],
                    "chrom": result["chrom"],
                    "tss": result["tss"],
                    "strand": result["strand"],
                    "closest_peak_index": result["closest_peak_index"],
                    "n_peaks": result["n_peaks"],
                    "status": "success"
                })
            else:
                failed_n += 1
                wt_meta_records.append({
                    "gene_name": gene_name,
                    "chrom": None,
                    "tss": None,
                    "strand": None,
                    "closest_peak_index": None,
                    "n_peaks": None,
                    "status": "failed"
                })

        except Exception as e:
            failed_n += 1
            wt_meta_records.append({
                "gene_name": gene_name,
                "status": "exception",
                "error": str(e)
            })

        if (idx + 1) % verbose_every == 0 or idx == 0 or idx == len(gene_list) - 1:
            print(f"  WT progress: {idx+1}/{len(gene_list)} | success={success_n} failed={failed_n} existing={existing_n}")

    with open(wt_meta_path, "w") as f:
        json.dump(wt_meta_records, f, indent=2)

    print(f"WT generation finished. success={success_n}, failed={failed_n}, existing={existing_n}")
    print(f"WT meta saved to: {wt_meta_path}")
    return wt_meta_path


class TFKOPeakMasker:
    def __init__(
        self,
        preprocessor: GeneATACPreprocessor,
        wild_type_dir: str,
        ko_base_dir: str,
        motif_scanner: FIMOMotifScanner
    ):
        self.preprocessor = preprocessor
        self.wild_type_dir = wild_type_dir
        self.ko_base_dir = ko_base_dir
        self.motif_scanner = motif_scanner
        ensure_dir(ko_base_dir)
        self._peak_coords_cache = {}

    def _load_wt_entries(self, gene_name: str) -> Optional[List[Tuple[str, str, str]]]:
        wt_path = os.path.join(self.wild_type_dir, f"{gene_name}.txt")
        if not os.path.exists(wt_path):
            return None
        return load_txt_entries(wt_path)

    def _get_peak_coords(self, gene_name: str) -> Optional[List[Optional[Tuple[str, int, int, str]]]]:
        """
        Return peak coordinates aligned with WT txt entry order:
        [(chrom, start, end, peak_name), ...]
        """
        if gene_name in self._peak_coords_cache:
            return self._peak_coords_cache[gene_name]

        if gene_name not in self.preprocessor.gene_index:
            self._peak_coords_cache[gene_name] = None
            return None

        gene = self.preprocessor.gene_index[gene_name]
        chrom, tss, window = gene["chrom"], gene["tss"], 150000
        region_start, region_end = tss - window, tss + window
        chr_peaks = self.preprocessor.peaks.get(chrom, [])
        selected = [(s, e, pid) for s, e, pid in chr_peaks if s >= region_start and e <= region_end]

        wt_path = os.path.join(self.wild_type_dir, f"{gene_name}.txt")
        if os.path.exists(wt_path):
            txt_headers = []
            with open(wt_path, "r") as f:
                for line in f:
                    if line.startswith(">"):
                        txt_headers.append(line.strip().lstrip(">"))

            id_to_coord = {
                f"{chrom}_{s}_{e}": (chrom, s, e, f"{chrom}_{s}_{e}")
                for s, e, _ in selected
            }
            matched = [id_to_coord.get(h) for h in txt_headers]
            selected = matched
        else:
            selected = [(chrom, s, e, f"{chrom}_{s}_{e}") for s, e, _ in selected]

        self._peak_coords_cache[gene_name] = selected
        return selected

    def mask_single_gene_for_tf(
        self,
        tf_name: str,
        gene_name: str,
        ko_output_dir: str,
        cell_line: str,
        deg_direction: str,
        deg_row: dict
    ) -> Tuple[bool, dict]:
        """
        Strategy:
        1. If any peak has best motif p <= strong threshold, mask all such peaks.
        2. Otherwise choose the globally best peak by minimum p-value.
        3. If the best peak is weaker than fallback threshold, skip.
        """
        ensure_dir(ko_output_dir)
        out_path = os.path.join(ko_output_dir, f"{gene_name}.txt")

        if os.path.exists(out_path):
            return True, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "cell_line": cell_line,
                "deg_direction": deg_direction,
                "status": "already_exists",
                "output_path": out_path
            }

        motif_path = self.motif_scanner.get_motif_path(tf_name)
        if motif_path is None:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "cell_line": cell_line,
                "deg_direction": deg_direction,
                "status": "no_motif_file"
            }

        entries = self._load_wt_entries(gene_name)
        if entries is None:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "cell_line": cell_line,
                "deg_direction": deg_direction,
                "status": "wt_missing"
            }

        peak_coords = self._get_peak_coords(gene_name)
        if peak_coords is None or len(peak_coords) == 0:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "cell_line": cell_line,
                "deg_direction": deg_direction,
                "status": "no_peak_coords"
            }

        if len(entries) != len(peak_coords):
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "cell_line": cell_line,
                "deg_direction": deg_direction,
                "status": "entry_coord_mismatch",
                "n_entries": len(entries),
                "n_coords": len(peak_coords)
            }

        seq_dict = {}
        for header, seq, _ in entries:
            seq_name = header.lstrip(">")
            seq_dict[seq_name] = seq

        hits_by_seq = self.motif_scanner.scan_sequences(tf_name, seq_dict)
        if hits_by_seq is None:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "cell_line": cell_line,
                "deg_direction": deg_direction,
                "status": "no_motif_file"
            }

        strong_thresh = self.motif_scanner.strong_pvalue_thresh
        fallback_thresh = self.motif_scanner.fallback_pvalue_thresh

        peak_best_pvals = []
        motif_hits_summary = []

        for i, (header, seq, sig_str) in enumerate(entries):
            seq_name = header.lstrip(">")
            hits = hits_by_seq.get(seq_name, [])

            if len(hits) == 0:
                peak_best_pvals.append((i, seq_name, None))
                continue

            best_p = min(h["p_value"] for h in hits)
            peak_best_pvals.append((i, seq_name, best_p))
            motif_hits_summary.append({
                "peak_index": i,
                "peak_id": seq_name,
                "n_hits": len(hits),
                "best_p_value": best_p
            })

        strong_peak_indices = [
            i for i, _, best_p in peak_best_pvals
            if best_p is not None and best_p <= strong_thresh
        ]

        selected_indices = []
        selection_mode = None
        fallback_selected_best_p = None

        if len(strong_peak_indices) > 0:
            selected_indices = strong_peak_indices
            selection_mode = "strong_all"
        else:
            candidate_peaks = [(i, seq_name, best_p) for i, seq_name, best_p in peak_best_pvals if best_p is not None]
            if len(candidate_peaks) == 0:
                return False, {
                    "gene_name": gene_name,
                    "tf_name": tf_name,
                    "cell_line": cell_line,
                    "deg_direction": deg_direction,
                    "status": "no_motif_hit_peak",
                    "log2FC": deg_row.get("log2FC")
                }

            best_i, best_seq_name, best_p = min(candidate_peaks, key=lambda x: x[2])
            fallback_selected_best_p = best_p

            if best_p > fallback_thresh:
                return False, {
                    "gene_name": gene_name,
                    "tf_name": tf_name,
                    "cell_line": cell_line,
                    "deg_direction": deg_direction,
                    "status": "best_motif_too_weak_skip",
                    "best_peak_index": best_i,
                    "best_peak_id": best_seq_name,
                    "best_peak_p_value": best_p,
                    "fallback_threshold": fallback_thresh,
                    "log2FC": deg_row.get("log2FC")
                }

            selected_indices = [best_i]
            selection_mode = "fallback_best_one"

        selected_index_set = set(selected_indices)
        masked_peaks = []

        for i in selected_indices:
            coord = peak_coords[i]
            seq_name = entries[i][0].lstrip(">")
            if coord is not None:
                chrom, start, end, peak_name = coord
                masked_peaks.append({
                    "peak_index": i,
                    "peak_chr": chrom,
                    "peak_start": start,
                    "peak_end": end,
                    "peak_id": peak_name
                })
            else:
                masked_peaks.append({
                    "peak_index": i,
                    "peak_chr": None,
                    "peak_start": None,
                    "peak_end": None,
                    "peak_id": seq_name
                })

        output_blocks = []
        for i, (header, seq, sig_str) in enumerate(entries):
            if i in selected_index_set:
                n_sig = len(sig_str.split("\t"))
                zero_sig = "\t".join(["0.0"] * n_sig)
                output_blocks.append((header, seq, zero_sig))
            else:
                output_blocks.append((header, seq, sig_str))

        with open(out_path, "w") as f:
            for header, seq, sig_str in output_blocks:
                f.write(f"{header}\n{seq}\n{sig_str}\n")

        meta = {
            "gene_name": gene_name,
            "tf_name": tf_name,
            "cell_line": cell_line,
            "deg_direction": deg_direction,
            "status": "success",
            "selection_mode": selection_mode,
            "strong_pvalue_threshold": strong_thresh,
            "fallback_pvalue_threshold": fallback_thresh,
            "fallback_selected_best_p": fallback_selected_best_p,
            "output_path": out_path,
            "n_total_peaks": len(entries),
            "n_masked_peaks": len(selected_indices),
            "masked_peak_indices": selected_indices,
            "masked_peaks": masked_peaks,
            "motif_hits_summary": motif_hits_summary,
            "log2FC": deg_row.get("log2FC")
        }

        return True, meta

    def process_gene_list_for_tf(
        self,
        tf_name: str,
        cell_line: str,
        deg_direction: str,
        gene_to_deg_info: Dict[str, dict],
        verbose_every: int = 20
    ) -> Tuple[str, List[dict]]:
        subdir_name = f"{cell_line}_{tf_name}_{deg_direction}"
        ko_output_dir = os.path.join(self.ko_base_dir, subdir_name)
        ensure_dir(ko_output_dir)

        print(f"Generating KO inputs: {subdir_name}")
        print(f"  Number of genes: {len(gene_to_deg_info)}")

        meta_records = []
        success_n = 0
        skipped_n = 0

        gene_list = list(gene_to_deg_info.keys())

        for idx, gene_name in enumerate(gene_list):
            if (idx + 1) % verbose_every == 0 or idx == 0 or idx == len(gene_list) - 1:
                print(f"  KO progress: {idx+1}/{len(gene_list)} | success={success_n} skipped={skipped_n}")

            deg_row = gene_to_deg_info[gene_name]

            try:
                ok, meta = self.mask_single_gene_for_tf(
                    tf_name=tf_name,
                    gene_name=gene_name,
                    ko_output_dir=ko_output_dir,
                    cell_line=cell_line,
                    deg_direction=deg_direction,
                    deg_row=deg_row
                )
            except Exception as e:
                ok = False
                meta = {
                    "gene_name": gene_name,
                    "tf_name": tf_name,
                    "cell_line": cell_line,
                    "deg_direction": deg_direction,
                    "status": "exception",
                    "error": str(e)
                }

            meta_records.append(meta)
            if ok:
                success_n += 1
            else:
                skipped_n += 1

        meta_path = os.path.join(ko_output_dir, "ko_meta_info.json")
        with open(meta_path, "w") as f:
            json.dump(meta_records, f, indent=2)

        print(f"KO generation finished: success={success_n}, skipped={skipped_n}")
        print(f"KO meta saved to: {meta_path}\n")
        return ko_output_dir, meta_records


# ============================================================
# Model input preparation
# ============================================================
def prepare_model_input_from_txt(
    txt_path,
    tss_loc,
    direction,
    narrowPeak_Reader,
    tokenizer,
    pre_processing,
    pre_pro,
    genome_dict
):
    raw_input_data = narrowPeak_Reader(txt_path)
    data_len = int(len(raw_input_data) / 3)

    dna_in_list, atac_in_list = [], []

    for i_data in range(data_len):
        seq = raw_input_data[i_data * 3 + 1][0].upper()
        if "N" in seq:
            raise ValueError(f"Sequence contains N in file: {txt_path}")

        dna_seq = pre_processing(tokenizer(seq, 6), genome_dict)
        dna_in_list.append(dna_seq)

        atac_vals = [float(v) for v in raw_input_data[i_data * 3 + 2]]
        atac_in_list.append(pre_pro(atac_vals, 6))

    return (
        torch.tensor(dna_in_list),
        torch.tensor(atac_in_list),
        int(tss_loc),
        direction
    )


def load_wt_meta(wt_meta_path: str) -> Dict[str, dict]:
    with open(wt_meta_path, "r") as f:
        records = json.load(f)

    gene_meta = {}
    for rec in records:
        gene = rec.get("gene_name")
        status = rec.get("status")
        if gene is None:
            continue
        if status not in {"success", "already_exists"}:
            continue

        # If already_exists was recorded without metadata, it cannot be used
        if rec.get("closest_peak_index") is None or rec.get("strand") is None:
            continue

        gene_meta[gene] = {
            "tss_loc": rec["closest_peak_index"],
            "direction": rec["strand"]
        }

    return gene_meta


def rebuild_wt_meta_from_txt_and_gtf(
    wild_type_dir: str,
    preprocessor: GeneATACPreprocessor,
    output_meta_path: str
) -> str:
    """
    Optional utility:
    reconstruct WT metadata from WT txt files and preprocessor gene index.
    Useful if previous meta file contains many 'already_exists' records without fields.
    """
    records = []

    txt_files = sorted([
        f for f in os.listdir(wild_type_dir)
        if f.endswith(".txt")
    ])

    for fname in txt_files:
        gene_name = fname[:-4]
        if gene_name not in preprocessor.gene_index:
            records.append({
                "gene_name": gene_name,
                "status": "failed"
            })
            continue

        gene = preprocessor.gene_index[gene_name]
        chrom = gene["chrom"]
        tss = gene["tss"]
        strand = gene["strand"]

        chr_peaks = preprocessor.peaks.get(chrom, [])
        selected = [(s, e, pid) for s, e, pid in chr_peaks if s >= tss - 150000 and e <= tss + 150000]
        headers = []
        txt_path = os.path.join(wild_type_dir, fname)
        with open(txt_path, "r") as f:
            for line in f:
                if line.startswith(">"):
                    headers.append(line.strip().lstrip(">"))

        aligned_coords = []
        id_to_coord = {f"{chrom}_{s}_{e}": (s, e) for s, e, _ in selected}
        for h in headers:
            aligned_coords.append(id_to_coord.get(h, None))

        closest_idx = None
        best_dist = float("inf")
        for i, coord in enumerate(aligned_coords):
            if coord is None:
                continue
            s, e = coord
            d = abs((s + e) / 2.0 - tss)
            if d < best_dist:
                best_dist = d
                closest_idx = i

        if closest_idx is None:
            records.append({
                "gene_name": gene_name,
                "status": "failed"
            })
            continue

        records.append({
            "gene_name": gene_name,
            "chrom": chrom,
            "tss": tss,
            "strand": strand,
            "closest_peak_index": closest_idx,
            "n_peaks": len(headers),
            "status": "success"
        })

    with open(output_meta_path, "w") as f:
        json.dump(records, f, indent=2)

    return output_meta_path


def build_gene_samples_from_txt_dir(
    txt_dir: str,
    gene_list: List[str],
    gene_meta: Dict[str, dict],
    narrowPeak_Reader,
    tokenizer,
    pre_processing,
    pre_pro,
    genome_dict,
    verbose_every: int = 20
) -> Dict[str, list]:
    gene_to_sample = {}
    valid_genes = [g for g in gene_list if g in gene_meta]

    print(f"Preparing model-ready tensors from: {txt_dir}")
    print(f"  Candidate genes with meta: {len(valid_genes)}")

    success_n = 0
    skipped_n = 0

    for idx, gene in enumerate(valid_genes):
        txt_path = os.path.join(txt_dir, f"{gene}.txt")
        if not os.path.exists(txt_path):
            skipped_n += 1
            continue

        if txt_contains_n(txt_path):
            skipped_n += 1
            continue

        meta = gene_meta[gene]

        try:
            dna_in, atac_in, tss_loc, direction = prepare_model_input_from_txt(
                txt_path=txt_path,
                tss_loc=meta["tss_loc"],
                direction=meta["direction"],
                narrowPeak_Reader=narrowPeak_Reader,
                tokenizer=tokenizer,
                pre_processing=pre_processing,
                pre_pro=pre_pro,
                genome_dict=genome_dict
            )

            gene_to_sample[gene] = [dna_in, atac_in, tss_loc, None, direction, {"gene_name": gene}]
            success_n += 1

        except Exception:
            skipped_n += 1

        if (idx + 1) % verbose_every == 0 or idx == 0 or idx == len(valid_genes) - 1:
            print(f"  Tensor prep progress: {idx+1}/{len(valid_genes)} | success={success_n} skipped={skipped_n}")

    print(f"Tensor preparation finished. success={success_n}, skipped={skipped_n}\n")
    return gene_to_sample


# ============================================================
# Model loading and inference
# ============================================================
def load_model_components(checkpoint_dir: str, device: torch.device) -> Dict[str, torch.nn.Module]:
    """
    Load all pretrained model components from checkpoint directory.
    """
    print("Loading model components...")

    components = {
        "dna_embed": torch.load(os.path.join(checkpoint_dir, "dna_embed.pkl"), map_location="cpu", weights_only=False).to(device),
        "atac_embed": torch.load(os.path.join(checkpoint_dir, "atac_embed.pkl"), map_location="cpu", weights_only=False).to(device),
        "pos1_embed": torch.load(os.path.join(checkpoint_dir, "pos1_embed.pkl"), map_location="cpu", weights_only=False).to(device),
        "pos2_embed": torch.load(os.path.join(checkpoint_dir, "pos2_embed.pkl"), map_location="cpu", weights_only=False).to(device),
        "tss_embed": torch.load(os.path.join(checkpoint_dir, "tss_embed.pkl"), map_location="cpu", weights_only=False).to(device),
        "pad_embed": torch.load(os.path.join(checkpoint_dir, "pad_embed.pkl"), map_location="cpu", weights_only=False).to(device),
        "encoder_1": torch.load(os.path.join(checkpoint_dir, "transformer_1.pkl"), map_location="cpu", weights_only=False).to(device),
        "encoder_2": torch.load(os.path.join(checkpoint_dir, "transformer_2.pkl"), map_location="cpu", weights_only=False).to(device),
        "encoder_3": torch.load(os.path.join(checkpoint_dir, "transformer_3.pkl"), map_location="cpu", weights_only=False).to(device),
        "atten_pool": torch.load(os.path.join(checkpoint_dir, "atten_pool.pkl"), map_location="cpu", weights_only=False).to(device),
        "ff_net": torch.load(os.path.join(checkpoint_dir, "feedforward.pkl"), map_location="cpu", weights_only=False).to(device),
    }

    for module in components.values():
        module.eval()

    print("Model components loaded.\n")
    return components


def infer_expression_from_prepared_sample(
    sample: list,
    model_parts: Dict[str, torch.nn.Module],
    device: torch.device,
    use_autocast: bool = True
) -> float:
    """
    sample format:
      [dna_in, atac_in, tss_loc, None_or_label, direction, meta]
    """
    dna_in = sample[0]
    atac_in = sample[1]
    tss_loc = int(sample[2])
    direction = sample[4]

    if not torch.is_tensor(dna_in):
        dna_in = torch.tensor(dna_in)
    if not torch.is_tensor(atac_in):
        atac_in = torch.tensor(atac_in)

    if dna_in.shape[0] > 150:
        raise ValueError(f"Number of peaks exceeds model limit 150: {dna_in.shape[0]}")

    if tss_loc < 0 or tss_loc >= dna_in.shape[0]:
        raise ValueError(f"Invalid tss_loc={tss_loc} for n_peaks={dna_in.shape[0]}")

    dna_in = dna_in.to(device)
    sig_in = atac_in.to(device)

    pos1 = torch.ones(129, dtype=int, device=device)
    for i in range(len(pos1)):
        pos1[i] += i

    pos2 = torch.ones(8, dtype=int, device=device)
    for i in range(len(pos2)):
        pos2[i] += i

    pos3 = torch.zeros(150, dtype=int, device=device)
    pos3[tss_loc] = 0

    if direction == "+":
        if tss_loc - 1 >= 0:
            pos3[tss_loc - 1] = 1
        if tss_loc + 1 < 150:
            pos3[tss_loc + 1] = 2

        for tss_i in range(max(0, tss_loc - 1)):
            pos3[tss_loc - 1 - tss_i - 1] = pos3[tss_loc - 1 - tss_i] + 2
        for tss_i in range(max(0, dna_in.shape[0] - tss_loc - 2)):
            pos3[tss_loc + 1 + tss_i + 1] = pos3[tss_loc + 1 + tss_i] + 2

    elif direction == "-":
        if tss_loc - 1 >= 0:
            pos3[tss_loc - 1] = 2
        if tss_loc + 1 < 150:
            pos3[tss_loc + 1] = 1

        for tss_i in range(max(0, tss_loc - 1)):
            pos3[tss_loc - 1 - tss_i - 1] = pos3[tss_loc - 1 - tss_i] + 2
        for tss_i in range(max(0, dna_in.shape[0] - tss_loc - 2)):
            pos3[tss_loc + 1 + tss_i + 1] = pos3[tss_loc + 1 + tss_i] + 2

    else:
        raise ValueError(f"Invalid direction: {direction}")

    dna_embed = model_parts["dna_embed"]
    atac_embed = model_parts["atac_embed"]
    pos1_embed = model_parts["pos1_embed"]
    pos2_embed = model_parts["pos2_embed"]
    tss_embed = model_parts["tss_embed"]
    pad_embed = model_parts["pad_embed"]
    encoder_1 = model_parts["encoder_1"]
    encoder_2 = model_parts["encoder_2"]
    encoder_3 = model_parts["encoder_3"]
    atten_pool = model_parts["atten_pool"]
    ff_net = model_parts["ff_net"]

    with torch.no_grad():
        if use_autocast and device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=True)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            cls_token = dna_embed(torch.ones(dna_in.shape[0] * 8, 1, dtype=int, device=device))
            x_pos_1 = pos1_embed(pos1)

            x_mul = (
                dna_embed(dna_in.int().reshape(dna_in.shape[0] * 8, 128)) +
                atac_embed(sig_in.int().reshape(dna_in.shape[0] * 8, 128))
            )

            x_embed = torch.cat((cls_token, x_mul), dim=1)
            x_enc_1 = encoder_1(x_embed + x_pos_1)[:, 0, :].reshape(dna_in.shape[0], 8, 2048)

            x_pos_2 = pos2_embed(pos2)
            x_enc_2 = encoder_2(x_enc_1 + x_pos_2)
            x_enc_2 = rearrange(x_enc_2, "b n d -> b d n")
            x_enc_2 = atten_pool(x_enc_2)
            x_enc_2 = rearrange(x_enc_2, "b d n -> b n d").squeeze(1)

            x_pad = pad_embed(torch.zeros(150 - x_enc_2.shape[0], dtype=int, device=device))
            x_eb3 = torch.cat((x_enc_2, x_pad), dim=0)

            x_pos_3 = tss_embed(pos3)
            x_enc_3 = encoder_3(x_eb3 + x_pos_3)
            result = ff_net(x_enc_3).squeeze(1)

    return result.float().item()


def predict_delta_for_gene(
    gene_name: str,
    wt_gene_to_sample: Dict[str, list],
    ko_gene_to_sample: Dict[str, list],
    model_parts: Dict[str, torch.nn.Module],
    device: torch.device,
    wt_pred_cache: Optional[Dict[str, float]] = None
) -> Optional[float]:
    if gene_name not in wt_gene_to_sample or gene_name not in ko_gene_to_sample:
        return None

    if wt_pred_cache is None:
        wt_pred_cache = {}

    if gene_name not in wt_pred_cache:
        wt_pred_cache[gene_name] = infer_expression_from_prepared_sample(
            wt_gene_to_sample[gene_name],
            model_parts=model_parts,
            device=device
        )

    wt_pred = wt_pred_cache[gene_name]

    ko_pred = infer_expression_from_prepared_sample(
        ko_gene_to_sample[gene_name],
        model_parts=model_parts,
        device=device
    )

    return float(ko_pred - wt_pred)


def run_delta_inference(
    genes_with_labels: pd.DataFrame,
    wt_gene_to_sample: Dict[str, list],
    ko_up_gene_to_sample: Dict[str, list],
    ko_down_gene_to_sample: Dict[str, list],
    model_parts: Dict[str, torch.nn.Module],
    device: torch.device,
    verbose_every: int = 20
) -> pd.DataFrame:
    """
    genes_with_labels must contain:
      - Gene
      - expected_direction
      - log2FC
    """
    records = []
    wt_pred_cache = {}

    total = len(genes_with_labels)
    print(f"Running inference for {total} genes...")

    for idx, row in genes_with_labels.reset_index(drop=True).iterrows():
        gene = str(row["Gene"]).strip()
        expected_direction = str(row["expected_direction"]).strip()
        log2fc = float(row["log2FC"])

        if expected_direction == "up":
            ko_map = ko_up_gene_to_sample
        elif expected_direction == "down":
            ko_map = ko_down_gene_to_sample
        else:
            records.append({
                "Gene": gene,
                "expected_direction": expected_direction,
                "log2FC": log2fc,
                "pred_delta": np.nan,
                "pred_direction": None,
                "status": "invalid_expected_direction"
            })
            continue

        try:
            pred_delta = predict_delta_for_gene(
                gene_name=gene,
                wt_gene_to_sample=wt_gene_to_sample,
                ko_gene_to_sample=ko_map,
                model_parts=model_parts,
                device=device,
                wt_pred_cache=wt_pred_cache
            )

            if pred_delta is None:
                records.append({
                    "Gene": gene,
                    "expected_direction": expected_direction,
                    "log2FC": log2fc,
                    "pred_delta": np.nan,
                    "pred_direction": None,
                    "status": "missing_sample"
                })
            else:
                pred_direction = "up" if pred_delta > 0 else ("down" if pred_delta < 0 else "flat")
                records.append({
                    "Gene": gene,
                    "expected_direction": expected_direction,
                    "log2FC": log2fc,
                    "pred_delta": float(pred_delta),
                    "pred_direction": pred_direction,
                    "status": "success"
                })

        except Exception as e:
            records.append({
                "Gene": gene,
                "expected_direction": expected_direction,
                "log2FC": log2fc,
                "pred_delta": np.nan,
                "pred_direction": None,
                "status": f"exception: {str(e)}"
            })

        if (idx + 1) % verbose_every == 0 or idx == 0 or idx == total - 1:
            done = idx + 1
            success_n = sum(1 for r in records if r["status"] == "success")
            print(f"  Inference progress: {done}/{total} | success={success_n}")

    print("Inference finished.\n")
    return pd.DataFrame(records)


# ============================================================
# Evaluation
# ============================================================
def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _f1_binary_for_label(y_true: List[str], y_pred: List[str], positive_label: str) -> float:
    tp = sum((t == positive_label and p == positive_label) for t, p in zip(y_true, y_pred))
    fp = sum((t != positive_label and p == positive_label) for t, p in zip(y_true, y_pred))
    fn = sum((t == positive_label and p != positive_label) for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_prediction_table(result_df: pd.DataFrame) -> Dict[str, float]:
    valid = result_df[result_df["status"] == "success"].copy()

    if len(valid) == 0:
        return {
            "n_valid": 0,
            "up_accuracy": float("nan"),
            "down_accuracy": float("nan"),
            "overall_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "pcc_delta_vs_log2fc": float("nan")
        }

    valid["correct"] = valid["expected_direction"] == valid["pred_direction"]

    up_df = valid[valid["expected_direction"] == "up"]
    down_df = valid[valid["expected_direction"] == "down"]

    up_accuracy = float(up_df["correct"].mean()) if len(up_df) > 0 else float("nan")
    down_accuracy = float(down_df["correct"].mean()) if len(down_df) > 0 else float("nan")
    overall_accuracy = float(valid["correct"].mean())

    y_true = valid["expected_direction"].tolist()
    y_pred = valid["pred_direction"].tolist()

    f1_up = _f1_binary_for_label(y_true, y_pred, "up")
    f1_down = _f1_binary_for_label(y_true, y_pred, "down")
    macro_f1 = (f1_up + f1_down) / 2.0

    pcc = _safe_pearsonr(valid["pred_delta"].values.astype(float), valid["log2FC"].values.astype(float))

    return {
        "n_valid": int(len(valid)),
        "up_accuracy": up_accuracy,
        "down_accuracy": down_accuracy,
        "overall_accuracy": overall_accuracy,
        "macro_f1": macro_f1,
        "pcc_delta_vs_log2fc": pcc
    }


def print_evaluation_summary(metrics: Dict[str, float]):
    print("Evaluation summary")
    print("------------------")
    print(f"Valid genes            : {metrics['n_valid']}")
    print(f"Up accuracy            : {metrics['up_accuracy']:.4f}" if not math.isnan(metrics["up_accuracy"]) else "Up accuracy            : nan")
    print(f"Down accuracy          : {metrics['down_accuracy']:.4f}" if not math.isnan(metrics["down_accuracy"]) else "Down accuracy          : nan")
    print(f"Overall accuracy       : {metrics['overall_accuracy']:.4f}" if not math.isnan(metrics["overall_accuracy"]) else "Overall accuracy       : nan")
    print(f"Macro-F1               : {metrics['macro_f1']:.4f}" if not math.isnan(metrics["macro_f1"]) else "Macro-F1               : nan")
    print(f"PCC(pred_delta, log2FC): {metrics['pcc_delta_vs_log2fc']:.4f}" if not math.isnan(metrics["pcc_delta_vs_log2fc"]) else "PCC(pred_delta, log2FC): nan")
    print("")


# ============================================================
# Tutorial-level pipeline helpers
# ============================================================
def build_tutorial_gene_tables(
    up_csv_path: str,
    down_csv_path: str,
    max_up: Optional[int] = None,
    max_down: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return:
      up_df   : sampled up DEG table with expected_direction column
      down_df : sampled down DEG table with expected_direction column
      merged  : concatenated table
    """
    up_df = load_deg_csv(up_csv_path).copy()
    down_df = load_deg_csv(down_csv_path).copy()

    if "Gene" not in up_df.columns or "log2FC" not in up_df.columns:
        raise ValueError(f"Up CSV must contain columns: Gene, log2FC | got: {list(up_df.columns)}")
    if "Gene" not in down_df.columns or "log2FC" not in down_df.columns:
        raise ValueError(f"Down CSV must contain columns: Gene, log2FC | got: {list(down_df.columns)}")

    up_df["expected_direction"] = "up"
    down_df["expected_direction"] = "down"

    merged = sample_deg_rows(
        up_df=up_df,
        down_df=down_df,
        max_up=max_up,
        max_down=max_down,
        random_seed=random_seed
    ).copy()

    merged["Gene"] = merged["Gene"].astype(str).str.strip()
    merged = merged[merged["Gene"] != ""].drop_duplicates(subset=["Gene", "expected_direction"]).reset_index(drop=True)

    up_sub = merged[merged["expected_direction"] == "up"].reset_index(drop=True)
    down_sub = merged[merged["expected_direction"] == "down"].reset_index(drop=True)

    return up_sub, down_sub, merged


def collect_union_gene_list(up_df: pd.DataFrame, down_df: pd.DataFrame) -> List[str]:
    genes = sorted(set(up_df["Gene"].astype(str).tolist()) | set(down_df["Gene"].astype(str).tolist()))
    return genes


def save_results_table(result_df: pd.DataFrame, output_csv_path: str):
    result_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")


def save_metrics_json(metrics: Dict[str, float], output_json_path: str):
    with open(output_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_json_path}")


def build_ko_gene_to_sample_map_from_dir(
    ko_dir: str,
    gene_list: List[str],
    gene_meta: Dict[str, dict],
    narrowPeak_Reader,
    tokenizer,
    pre_processing,
    pre_pro,
    genome_dict,
    verbose_every: int = 20
) -> Dict[str, list]:
    return build_gene_samples_from_txt_dir(
        txt_dir=ko_dir,
        gene_list=gene_list,
        gene_meta=gene_meta,
        narrowPeak_Reader=narrowPeak_Reader,
        tokenizer=tokenizer,
        pre_processing=pre_processing,
        pre_pro=pre_pro,
        genome_dict=genome_dict,
        verbose_every=verbose_every
    )


def prepare_tutorial_outputs(
    output_root: str,
    tf_name: str,
    cell_line: str
) -> Dict[str, str]:
    """
    Create and return all tutorial output paths.
    """
    ensure_dir(output_root)

    wt_dir = os.path.join(output_root, "wild_type_txt")
    ko_root = os.path.join(output_root, "tf_ko_txt")
    results_dir = os.path.join(output_root, "results")
    cache_dir = os.path.join(output_root, "cache")

    ensure_dir(wt_dir)
    ensure_dir(ko_root)
    ensure_dir(results_dir)
    ensure_dir(cache_dir)

    return {
        "wt_dir": wt_dir,
        "ko_root": ko_root,
        "results_dir": results_dir,
        "cache_dir": cache_dir,
        "wt_meta_path": os.path.join(wt_dir, "meta_info.json"),
        "result_csv": os.path.join(results_dir, f"{cell_line}_{tf_name}_tutorial_predictions.csv"),
        "metric_json": os.path.join(results_dir, f"{cell_line}_{tf_name}_tutorial_metrics.json"),
    }

