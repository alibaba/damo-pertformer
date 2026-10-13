import os
import csv
import json
import math
import glob
import random
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional

import torch
import pyBigWig
import numpy as np
import pandas as pd
from einops import rearrange


# ============================================================
# Basic file / csv utils
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def discover_csvs(csv_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(csv_dir, "*.csv")))


def parse_tf_deg_filename(csv_path: str) -> Optional[Tuple[str, str, str]]:
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


def load_deg_csv_rows(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def split_tf_combo(tf_name: str) -> List[str]:
    tf_name = str(tf_name).strip()
    if not tf_name:
        return []
    return [tf_name]


def sample_gene_rows(rows: List[dict], max_samples: Optional[int] = None, random_seed: int = 42) -> List[dict]:
    if max_samples is None or max_samples >= len(rows):
        return rows
    rng = random.Random(random_seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:max_samples])
    return [rows[i] for i in idxs]


def merge_up_down_rows(
    up_csv_path: str,
    down_csv_path: str,
    max_samples_per_direction: Optional[int] = None,
    random_seed: int = 42
) -> List[dict]:
    up_rows = load_deg_csv_rows(up_csv_path)
    down_rows = load_deg_csv_rows(down_csv_path)

    up_rows = sample_gene_rows(up_rows, max_samples_per_direction, random_seed)
    down_rows = sample_gene_rows(down_rows, max_samples_per_direction, random_seed)

    merged = []
    for row in up_rows:
        x = dict(row)
        x["_tutorial_direction"] = "up"
        merged.append(x)

    for row in down_rows:
        x = dict(row)
        x["_tutorial_direction"] = "down"
        merged.append(x)

    return merged


# ============================================================
# Genome / preprocessing
# ============================================================
class GeneATACPreprocessor:
    def __init__(self, gtf_path: str, genome_dir: str, peak_path: str, bigwig_path: str):
        print("[Preprocessor] Loading static resources...")
        self.gene_index = self._parse_gtf(gtf_path)
        print(f"[Preprocessor] GTF loaded: {len(self.gene_index)} genes")

        self.genome = [None] * 24
        for i in range(22):
            self.genome[i] = torch.load(f"{genome_dir}/chr{i+1}.pkl")
        self.genome[22] = torch.load(f"{genome_dir}/chrX.pkl")
        self.genome[23] = torch.load(f"{genome_dir}/chrY.pkl")

        self.chr_to_idx = {f"chr{i+1}": i for i in range(22)}
        self.chr_to_idx["chrX"] = 22
        self.chr_to_idx["chrY"] = 23
        print("[Preprocessor] Genome tensors loaded")

        self.peaks = self._load_and_dedup_peaks(peak_path)
        print(f"[Preprocessor] Peaks loaded: {sum(len(v) for v in self.peaks.values())} unique peaks")

        self.bw = pyBigWig.open(bigwig_path)
        print("[Preprocessor] BigWig opened")
        print("[Preprocessor] Ready.\n")

    def _parse_gtf(self, gtf_path: str) -> Dict[str, dict]:
        gene_info = {}
        with open(gtf_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9 or fields[2] != "gene":
                    continue
                chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
                gene_name = None
                for attr in fields[8].split(";"):
                    attr = attr.strip()
                    if attr.startswith("gene_name"):
                        try:
                            gene_name = attr.split('"')[1]
                        except Exception:
                            pass
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

    def _load_and_dedup_peaks(self, peak_path: str) -> Dict[str, List[Tuple]]:
        raw_peaks = {}
        seen = set()
        with open(peak_path, "r") as f:
            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < 4:
                    continue
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

        if start_0 < 0 or end_0 > len(chr_tensor):
            return None

        seq_tensor = chr_tensor[start_0:end_0]
        seq = decode_genome_slice_to_sequence(seq_tensor)
        return seq

    def _extract_signal(self, chrom: str, center: int, length: int = 1029) -> Optional[List[float]]:
        half = length // 2
        start_0 = center - 1 - half
        end_0 = start_0 + length
        try:
            values = self.bw.values(chrom, start_0, end_0)
            if values is None:
                return [0.0] * length
            return [float(v) if v is not None and not np.isnan(v) else 0.0 for v in values]
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

        if output_txt is not None:
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


def build_wild_type_inputs_for_genes(
    preprocessor: GeneATACPreprocessor,
    gene_names: List[str],
    wild_type_dir: str
) -> str:
    ensure_dir(wild_type_dir)
    meta_path = os.path.join(wild_type_dir, "meta_info.json")

    print(f"[WT] Building WT inputs for {len(gene_names)} genes...")

    meta_records = []
    success_n = 0

    for i, gene_name in enumerate(gene_names):
        if (i + 1) % 10 == 0 or i == 0 or i == len(gene_names) - 1:
            print(f"[WT] Progress: {i+1}/{len(gene_names)} | success={success_n}")

        output_txt = os.path.join(wild_type_dir, f"{gene_name}.txt")
        result = preprocessor.process_gene(gene_name=gene_name, output_txt=output_txt)

        if result is None:
            meta_records.append({
                "gene_name": gene_name,
                "status": "failed"
            })
            continue

        success_n += 1
        meta_records.append({
            "gene_name": result["gene_name"],
            "chrom": result["chrom"],
            "tss": result["tss"],
            "strand": result["strand"],
            "closest_peak_index": result["closest_peak_index"],
            "n_peaks": result["n_peaks"],
            "status": "success"
        })

    with open(meta_path, "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"[WT] Done. Success={success_n}/{len(gene_names)}")
    return meta_path


# ============================================================
# Fixed donor signal
# ============================================================
def load_fixed_donor_signal(saved_path: str) -> List[float]:
    obj = torch.load(saved_path, map_location="cpu")
    if isinstance(obj, dict):
        signal = obj.get("signal", None)
        if signal is None:
            raise ValueError("Saved donor object does not contain key 'signal'")
        signal = [float(x) for x in signal]
    else:
        signal = [float(x) for x in obj]

    if len(signal) == 0:
        raise ValueError("Fixed donor signal is empty")

    print(f"[Donor] Loaded fixed donor signal: length={len(signal)}")
    return signal

def decode_genome_slice_to_sequence(seq_tensor) -> str:
    """
    Robustly decode a genome slice into an uppercase DNA string.

    Supported element types:
    - integer-coded bases: 0/1/2/3/4 -> A/C/G/T/N
    - character-coded bases: 'a', 'c', 'g', 't', 'n'
    """
    base_map = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

    seq_chars = []
    for b in seq_tensor:
        try:
            seq_chars.append(base_map.get(int(b), "N"))
        except Exception:
            s = str(b).strip().upper()
            if s in {"A", "C", "G", "T", "N"}:
                seq_chars.append(s)
            else:
                seq_chars.append("N")

    return "".join(seq_chars)

# ============================================================
# FIMO motif scan
# ============================================================
class FIMOMotifScanner:
    def __init__(self, motif_dir: str, fimo_path: str = "fimo"):
        self.motif_dir = motif_dir
        self.fimo_path = fimo_path

    def get_motif_path(self, tf_name: str) -> Optional[str]:
        motif_path = os.path.join(self.motif_dir, f"{tf_name}.meme")
        return motif_path if os.path.exists(motif_path) else None

    def scan_search_segments(
        self,
        tf_name: str,
        segments: List[dict],
        pvalue_thresh: float = 1e-4
    ) -> Optional[List[dict]]:
        motif_path = self.get_motif_path(tf_name)
        if motif_path is None:
            return None

        if len(segments) == 0:
            return []

        seg_map = {seg["segment_name"]: seg for seg in segments}

        with tempfile.TemporaryDirectory(prefix="fimo_tutorial_") as tmpdir:
            fasta_path = os.path.join(tmpdir, "input.fa")
            out_dir = os.path.join(tmpdir, "fimo_out")

            with open(fasta_path, "w") as f:
                for seg in segments:
                    f.write(f">{seg['segment_name']}\n{seg['sequence']}\n")

            cmd = [
                self.fimo_path,
                "--verbosity", "1",
                "--thresh", str(pvalue_thresh),
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
                return []

            fimo_tsv = os.path.join(out_dir, "fimo.tsv")
            if not os.path.exists(fimo_tsv):
                return []

            hits = []
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
                        start_1 = int(fields[3])
                        stop_1 = int(fields[4])
                        strand = fields[5]
                        p_value = float(fields[7])
                        matched_sequence = fields[9]
                    except Exception:
                        continue

                    seg = seg_map.get(sequence_name)
                    if seg is None:
                        continue

                    genomic_start_0 = seg["start"] + (start_1 - 1)
                    genomic_end_0 = seg["start"] + stop_1
                    genomic_center_1 = ((genomic_start_0 + genomic_end_0 - 1) // 2) + 1

                    hits.append({
                        "segment_name": sequence_name,
                        "chrom": seg["chrom"],
                        "segment_start": seg["start"],
                        "segment_end": seg["end"],
                        "hit_start_1based": start_1,
                        "hit_stop_1based": stop_1,
                        "strand": strand,
                        "p_value": p_value,
                        "matched_sequence": matched_sequence,
                        "genomic_start_0based": genomic_start_0,
                        "genomic_end_0based": genomic_end_0,
                        "genomic_center_1based": genomic_center_1
                    })

            return hits


# ============================================================
# KI generator with fixed donor signal
# ============================================================
class TFKIPeakInserterFixedSignal:
    def __init__(
        self,
        preprocessor: GeneATACPreprocessor,
        wild_type_dir: str,
        ki_dir: str,
        motif_scanner: FIMOMotifScanner,
        fixed_donor_signal: List[float],
        motif_pvalue_thresh: float = 1e-4,
        window: int = 150000,
        seq_len: int = 1029
    ):
        self.preprocessor = preprocessor
        self.wild_type_dir = wild_type_dir
        self.ki_dir = ki_dir
        self.motif_scanner = motif_scanner
        self.fixed_donor_signal = fixed_donor_signal
        self.fixed_donor_signal_str = "\t".join([str(v) for v in fixed_donor_signal])
        self.motif_pvalue_thresh = motif_pvalue_thresh
        self.window = window
        self.seq_len = seq_len
        self._peak_coords_cache = {}

        ensure_dir(self.ki_dir)

        if len(self.fixed_donor_signal) != self.seq_len:
            raise ValueError(
                f"fixed_donor_signal length must equal seq_len: "
                f"{len(self.fixed_donor_signal)} vs {self.seq_len}"
            )

    def _load_wt_entries(self, gene_name: str) -> Optional[List[Tuple[str, str, str]]]:
        wt_path = os.path.join(self.wild_type_dir, f"{gene_name}.txt")
        if not os.path.exists(wt_path):
            return None

        with open(wt_path, "r") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]

        entries = []
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                entries.append((lines[i], lines[i + 1], lines[i + 2]))
        return entries

    def _get_peak_coords(self, gene_name: str) -> Optional[List[Optional[Tuple[str, int, int, str]]]]:
        if gene_name in self._peak_coords_cache:
            return self._peak_coords_cache[gene_name]

        if gene_name not in self.preprocessor.gene_index:
            self._peak_coords_cache[gene_name] = None
            return None

        gene = self.preprocessor.gene_index[gene_name]
        chrom, tss = gene["chrom"], gene["tss"]
        region_start, region_end = tss - self.window, tss + self.window

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
                for s, e, pid in selected
            }
            selected = [id_to_coord.get(h) for h in txt_headers]
        else:
            selected = [(chrom, s, e, f"{chrom}_{s}_{e}") for s, e, pid in selected]

        self._peak_coords_cache[gene_name] = selected
        return selected

    def _parse_signal_str(self, sig_str: str) -> List[float]:
        if not sig_str:
            return []
        return [float(x) for x in sig_str.split("\t")]

    def _signal_strength(self, sig: List[float]) -> float:
        return float(np.sum(sig))

    def _find_donor_peak(self, entries, peak_coords):
        best_idx = None
        best_strength = -1.0
        best_sig = None
        best_coord = None
        best_header = None

        for i, (header, seq, sig_str) in enumerate(entries):
            sig = self._parse_signal_str(sig_str)
            strength = self._signal_strength(sig)
            if strength > best_strength:
                best_strength = strength
                best_idx = i
                best_sig = sig
                best_coord = peak_coords[i]
                best_header = header

        if best_idx is None or best_coord is None:
            return None

        chrom, start, end, peak_name = best_coord
        return {
            "donor_peak_index": best_idx,
            "donor_peak_header": best_header,
            "donor_peak_chr": chrom,
            "donor_peak_start": start,
            "donor_peak_end": end,
            "donor_peak_id": peak_name,
            "donor_peak_width": end - start,
            "donor_peak_signal": best_sig,
            "donor_peak_signal_strength": best_strength
        }

    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))
        return merged

    def _subtract_intervals(
        self,
        region_start: int,
        region_end: int,
        blocked: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        if region_end <= region_start:
            return []

        blocked = self._merge_intervals([
            (max(region_start, s), min(region_end, e))
            for s, e in blocked
            if e > region_start and s < region_end
        ])

        result = []
        cur = region_start
        for s, e in blocked:
            if cur < s:
                result.append((cur, s))
            cur = max(cur, e)
        if cur < region_end:
            result.append((cur, region_end))
        return result

    def _build_search_segments(
        self,
        gene_name: str,
        donor_width: int
    ) -> Tuple[Optional[List[dict]], dict]:
        gene = self.preprocessor.gene_index.get(gene_name)
        if gene is None:
            return None, {"status": "gene_not_found"}

        chrom = gene["chrom"]
        tss = gene["tss"]
        region_start_1 = tss - self.window
        region_end_1 = tss + self.window

        chr_tensor = self.preprocessor._get_chr_tensor(chrom)
        if chr_tensor is None:
            return None, {"status": "chrom_tensor_missing"}

        chr_len = len(chr_tensor)
        region_start_0 = max(0, region_start_1 - 1)
        region_end_0 = min(chr_len, region_end_1)

        peak_coords = self._get_peak_coords(gene_name)
        if peak_coords is None:
            return None, {"status": "no_peak_coords"}

        donor_half = donor_width // 2

        blocked = []
        for coord in peak_coords:
            if coord is None:
                continue
            _, s, e, _ = coord
            blocked.append((max(0, s - donor_half), min(chr_len, e + donor_half)))

        blocked = self._merge_intervals(blocked)
        allowed = self._subtract_intervals(region_start_0, region_end_0, blocked)

        segments = []
        for idx, (s0, e0) in enumerate(allowed):
            if e0 <= s0:
                continue
            seq_tensor = self.preprocessor._get_chr_tensor(chrom)[s0:e0]
            seq = decode_genome_slice_to_sequence(seq_tensor)
            if len(seq) == 0:
                continue

            segments.append({
                "segment_name": f"{gene_name}__seg{idx}__{chrom}_{s0}_{e0}",
                "chrom": chrom,
                "start": s0,
                "end": e0,
                "sequence": seq
            })

        meta = {
            "status": "success",
            "chrom": chrom,
            "tss": tss,
            "region_start_0based": region_start_0,
            "region_end_0based": region_end_0,
            "donor_width": donor_width,
            "n_allowed_segments": len(segments)
        }
        return segments, meta

    def _select_best_hits_for_single_tf(self, tf_name: str, segments: List[dict]) -> Tuple[bool, dict]:
        motif_path = self.motif_scanner.get_motif_path(tf_name)
        if motif_path is None:
            return False, {
                "tf_name": tf_name,
                "status": "no_motif_file"
            }

        hits = self.motif_scanner.scan_search_segments(
            tf_name=tf_name,
            segments=segments,
            pvalue_thresh=self.motif_pvalue_thresh
        )
        if hits is None:
            return False, {
                "tf_name": tf_name,
                "status": "no_motif_file"
            }

        if len(hits) == 0:
            return False, {
                "tf_name": tf_name,
                "status": "no_hit_under_threshold"
            }

        best_p = min(h["p_value"] for h in hits)
        best_hits = [h for h in hits if h["p_value"] == best_p]

        return True, {
            "tf_name": tf_name,
            "status": "success",
            "best_p_value": best_p,
            "best_hits": best_hits
        }

    def _merge_close_centers(self, centers_1based: List[int], donor_width: int) -> List[int]:
        if not centers_1based:
            return []

        centers = sorted(centers_1based)
        changed = True

        while changed:
            changed = False
            groups = []
            cur = [centers[0]]

            for c in centers[1:]:
                if c - cur[-1] < donor_width:
                    cur.append(c)
                else:
                    groups.append(cur)
                    cur = [c]
            groups.append(cur)

            new_centers = []
            for g in groups:
                if len(g) == 1:
                    new_centers.append(g[0])
                else:
                    changed = True
                    new_centers.append(int(round(sum(g) / len(g))))

            centers = sorted(new_centers)

        return centers

    def _find_tss_closest_peak_index_from_records(self, records: List[dict], tss: int) -> Optional[int]:
        best_idx, best_dist = None, float("inf")
        for i, rec in enumerate(records):
            center = (rec["start"] + rec["end"]) / 2.0
            d = abs(center - tss)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def knock_in_single_gene_for_tf(
        self,
        tf_name: str,
        gene_name: str
    ) -> Tuple[bool, dict]:
        out_path = os.path.join(self.ki_dir, f"{gene_name}.txt")
        if os.path.exists(out_path):
            return True, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "already_exists",
                "output_path": out_path
            }

        tf_list = split_tf_combo(tf_name)
        if len(tf_list) == 0:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "empty_tf_name"
            }

        single_tf = tf_list[0]

        entries = self._load_wt_entries(gene_name)
        if entries is None:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "wt_missing"
            }

        peak_coords = self._get_peak_coords(gene_name)
        if peak_coords is None or len(peak_coords) == 0:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "no_peak_coords"
            }

        if len(entries) != len(peak_coords):
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "entry_coord_mismatch",
                "n_entries": len(entries),
                "n_coords": len(peak_coords)
            }

        donor = self._find_donor_peak(entries, peak_coords)
        if donor is None:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "donor_peak_not_found"
            }

        donor_width = donor["donor_peak_width"]
        if donor_width <= 0:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "invalid_donor_width",
                "donor_width": donor_width
            }

        segments, search_meta = self._build_search_segments(
            gene_name=gene_name,
            donor_width=donor_width
        )
        if segments is None:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": search_meta.get("status", "build_search_segments_failed")
            }

        if len(segments) == 0:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "no_search_space_after_peak_exclusion",
                "donor_width": donor_width
            }

        ok, tf_scan_meta = self._select_best_hits_for_single_tf(
            tf_name=single_tf,
            segments=segments
        )

        per_tf_scan_results = [tf_scan_meta]
        raw_selected_centers = []
        raw_selected_hits = []

        if ok:
            best_hits = tf_scan_meta.get("best_hits", [])
            for hit in best_hits:
                center_1based = hit["genomic_center_1based"]
                raw_selected_centers.append(center_1based)
                raw_selected_hits.append({
                    "source_tf": single_tf,
                    "chrom": hit["chrom"],
                    "segment_name": hit["segment_name"],
                    "p_value": hit["p_value"],
                    "strand": hit["strand"],
                    "matched_sequence": hit["matched_sequence"],
                    "genomic_start_0based": hit["genomic_start_0based"],
                    "genomic_end_0based": hit["genomic_end_0based"],
                    "genomic_center_1based": center_1based
                })

        if len(raw_selected_centers) == 0:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "no_hit_under_threshold",
                "motif_pvalue_threshold": self.motif_pvalue_thresh,
                "per_tf_scan_results": per_tf_scan_results
            }

        final_centers_1based = self._merge_close_centers(
            centers_1based=raw_selected_centers,
            donor_width=donor_width
        )

        gene_info = self.preprocessor.gene_index.get(gene_name, {})
        chrom = gene_info.get("chrom")
        tss = gene_info.get("tss")
        strand = gene_info.get("strand")

        wt_records = []
        for i, ((header, seq, sig_str), coord) in enumerate(zip(entries, peak_coords)):
            if coord is None:
                return False, {
                    "gene_name": gene_name,
                    "tf_name": tf_name,
                    "status": "wt_coord_missing_at_some_peak",
                    "peak_index": i
                }

            cchrom, start, end, peak_name = coord
            wt_records.append({
                "header": header,
                "seq": seq,
                "sig_str": sig_str,
                "chrom": cchrom,
                "start": start,
                "end": end,
                "peak_type": "WT",
                "source_tf": None,
                "source_peak_id": peak_name
            })

        ki_records = []
        final_centers_1based_sorted = sorted(final_centers_1based)

        for center_1based in final_centers_1based_sorted:
            seq = self.preprocessor._extract_sequence(chrom, center_1based, self.seq_len)
            if seq is None:
                continue

            new_start = center_1based - 1 - (donor_width // 2)
            new_end = new_start + donor_width

            chr_tensor = self.preprocessor._get_chr_tensor(chrom)
            if chr_tensor is None:
                continue

            chr_len = len(chr_tensor)
            if new_start < 0 or new_end > chr_len or new_end <= new_start:
                continue

            linked_hits = []
            for h in raw_selected_hits:
                if h["genomic_center_1based"] == center_1based:
                    linked_hits.append(h)

            if len(linked_hits) == 0:
                linked_hits = [
                    h for h in raw_selected_hits
                    if abs(h["genomic_center_1based"] - center_1based) < donor_width
                ]

            source_tf_name = single_tf
            header = f">KI_{chrom}_{new_start}_{new_end}_tf_{source_tf_name}"

            ki_records.append({
                "header": header,
                "seq": seq,
                "sig_str": self.fixed_donor_signal_str,
                "chrom": chrom,
                "start": new_start,
                "end": new_end,
                "peak_type": "KI",
                "source_tf": source_tf_name,
                "source_peak_id": "fixed_signal_donor",
                "ki_center_1based": center_1based,
                "linked_hits": linked_hits
            })

        if len(ki_records) == 0:
            return False, {
                "gene_name": gene_name,
                "tf_name": tf_name,
                "status": "no_valid_ki_peak_after_coordinate_check",
                "raw_selected_centers_1based": raw_selected_centers,
                "merged_centers_1based": final_centers_1based
            }

        all_records = wt_records + ki_records
        all_records = sorted(all_records, key=lambda x: (x["chrom"], x["start"], x["end"], x["peak_type"]))

        tss_closest_peak_index = self._find_tss_closest_peak_index_from_records(all_records, tss) if tss is not None else None

        with open(out_path, "w") as f:
            for rec in all_records:
                f.write(f"{rec['header']}\n")
                f.write(f"{rec['seq']}\n")
                f.write(f"{rec['sig_str']}\n")

        meta = {
            "gene_name": gene_name,
            "tf_name": tf_name,
            "tf_list": [single_tf],
            "status": "success",
            "output_path": out_path,
            "strand": strand,
            "tss_position": tss,
            "tss_closest_peak_index": tss_closest_peak_index,
            "n_wt_peaks": len(wt_records),
            "n_added_peaks": len(ki_records),
            "n_total_peaks_after_ki": len(all_records),
            "donor_peak_index": donor["donor_peak_index"],
            "donor_peak_header": donor["donor_peak_header"],
            "donor_peak_chr": donor["donor_peak_chr"],
            "donor_peak_start": donor["donor_peak_start"],
            "donor_peak_end": donor["donor_peak_end"],
            "donor_peak_id": donor["donor_peak_id"],
            "donor_peak_width": donor["donor_peak_width"],
            "donor_peak_signal_strength": donor["donor_peak_signal_strength"],
            "ki_signal_mode": "fixed_external_saved_signal",
            "fixed_signal_length": len(self.fixed_donor_signal),
            "motif_pvalue_threshold": self.motif_pvalue_thresh,
            "search_space_meta": search_meta,
            "per_tf_scan_results": per_tf_scan_results,
            "raw_selected_centers_1based": sorted(raw_selected_centers),
            "merged_final_centers_1based": final_centers_1based_sorted,
            "added_peaks": [
                {
                    "peak_index_after_sort": i,
                    "header": rec["header"],
                    "chrom": rec["chrom"],
                    "start": rec["start"],
                    "end": rec["end"],
                    "peak_type": rec["peak_type"],
                    "source_tf": rec.get("source_tf"),
                    "source_peak_id": rec.get("source_peak_id"),
                    "ki_center_1based": rec.get("ki_center_1based"),
                    "linked_hits": rec.get("linked_hits", [])
                }
                for i, rec in enumerate(all_records)
                if rec["peak_type"] == "KI"
            ]
        }

        return True, meta


    def batch_build_for_gene_rows(
        self,
        tf_name: str,
        gene_rows: List[dict],
        meta_output_path: str
    ) -> List[dict]:
        print(f"[KI] Building KI inputs for TF={tf_name} on {len(gene_rows)} genes...")

        meta_records = []
        success_n = 0

        for i, row in enumerate(gene_rows):
            gene_name = str(row.get("Gene", "")).strip()

            if (i + 1) % 10 == 0 or i == 0 or i == len(gene_rows) - 1:
                print(f"[KI] Progress: {i+1}/{len(gene_rows)} | success={success_n}")

            if not gene_name:
                meta_records.append({
                    "gene_name": None,
                    "tf_name": tf_name,
                    "status": "empty_gene_name"
                })
                continue

            try:
                ok, meta = self.knock_in_single_gene_for_tf(tf_name=tf_name, gene_name=gene_name)
            except Exception as e:
                ok = False
                meta = {
                    "gene_name": gene_name,
                    "tf_name": tf_name,
                    "status": "exception",
                    "error": str(e)
                }

            if ok:
                success_n += 1

            if row.get("log2FC") is not None:
                meta["log2FC"] = row.get("log2FC")
            if row.get("_tutorial_direction") is not None:
                meta["tutorial_direction"] = row.get("_tutorial_direction")

            meta_records.append(meta)

        with open(meta_output_path, "w") as f:
            json.dump(meta_records, f, indent=2)

        print(f"[KI] Done. Success={success_n}/{len(gene_rows)}")
        print(f"[KI] Meta saved to: {meta_output_path}")
        return meta_records


# ============================================================
# Dataset preparation
# ============================================================
def txt_contains_N(txt_path: str) -> bool:
    with open(txt_path, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    for i in range(0, len(lines), 3):
        if i + 1 < len(lines):
            seq = lines[i + 1].strip().upper()
            if "N" in seq:
                return True
    return False


def load_wt_gene_meta(wt_meta_path: str) -> Dict[str, dict]:
    gene_meta = {}
    with open(wt_meta_path, "r") as f:
        records = json.load(f)

    for rec in records:
        gene = rec.get("gene_name")
        if gene is None:
            continue
        if rec.get("status") != "success":
            continue

        gene_meta[gene] = {
            "tss_loc": rec["closest_peak_index"],
            "direction": rec["strand"]
        }

    return gene_meta


def load_ki_gene_meta(ki_meta_path: str) -> Dict[str, dict]:
    gene_meta = {}
    with open(ki_meta_path, "r") as f:
        records = json.load(f)

    for rec in records:
        gene = rec.get("gene_name")
        if gene is None:
            continue
        if rec.get("status") != "success":
            continue

        gene_meta[gene] = {
            "tss_loc": rec["tss_closest_peak_index"],
            "direction": rec["strand"],
            "n_wt_peaks": rec.get("n_wt_peaks"),
            "n_added_peaks": rec.get("n_added_peaks"),
            "n_total_peaks_after_ki": rec.get("n_total_peaks_after_ki"),
            "log2FC": rec.get("log2FC"),
            "tutorial_direction": rec.get("tutorial_direction")
        }

    return gene_meta


def prepare_model_input(txt_path, tss_loc, direction, genome_dict, narrowPeak_Reader, pre_processing, tokenizer, pre_pro):
    raw_input_data = narrowPeak_Reader(txt_path)
    data_len = int(len(raw_input_data) / 3)

    dna_in_list, atac_in_list = [], []
    for i_data in range(data_len):
        seq = raw_input_data[i_data * 3 + 1][0].upper()

        if "N" in seq:
            raise ValueError(f"Sequence contains N in file: {txt_path}")

        dna_seq = pre_processing(
            tokenizer(seq, 6), genome_dict
        )
        dna_in_list.append(dna_seq)

        atac_vals = [float(v) for v in raw_input_data[i_data * 3 + 2]]
        atac_in_list.append(pre_pro(atac_vals, 6))

    return (
        torch.tensor(dna_in_list),
        torch.tensor(atac_in_list),
        int(tss_loc),
        direction
    )


def build_wt_dataset_for_tutorial(
    wild_type_dir: str,
    wt_meta_path: str,
    gene_names: List[str],
    genome_dict,
    narrowPeak_Reader,
    pre_processing,
    tokenizer,
    pre_pro
):
    print(f"[Dataset] Building WT dataset for {len(gene_names)} genes...")

    gene_meta = load_wt_gene_meta(wt_meta_path)
    dataset = []
    gene_to_sample = {}

    for i, gene in enumerate(gene_names):
        if (i + 1) % 10 == 0 or i == 0 or i == len(gene_names) - 1:
            print(f"[Dataset][WT] Progress: {i+1}/{len(gene_names)} | kept={len(dataset)}")

        txt_path = os.path.join(wild_type_dir, f"{gene}.txt")
        if gene not in gene_meta:
            continue
        if not os.path.exists(txt_path):
            continue
        if txt_contains_N(txt_path):
            continue

        try:
            meta = gene_meta[gene]
            dna_in, atac_in, tss_loc, direction = prepare_model_input(
                txt_path, meta["tss_loc"], meta["direction"],
                genome_dict, narrowPeak_Reader, pre_processing, tokenizer, pre_pro
            )

            sample_meta = {
                "gene_name": gene,
                "source": "WT"
            }
            sample = [dna_in, atac_in, tss_loc, 0.0, direction, sample_meta]
            dataset.append(sample)
            gene_to_sample[gene] = sample
        except Exception:
            continue

    print(f"[Dataset][WT] Done. Samples={len(dataset)}")
    return dataset, gene_to_sample


def build_ki_dataset_for_tutorial(
    ki_dir: str,
    ki_meta_path: str,
    gene_to_wt_sample: Dict[str, list],
    genome_dict,
    narrowPeak_Reader,
    pre_processing,
    tokenizer,
    pre_pro
):
    print("[Dataset] Building KI dataset...")

    ki_gene_meta = load_ki_gene_meta(ki_meta_path)
    dataset = []

    txt_files = sorted([
        f for f in os.listdir(ki_dir)
        if f.endswith(".txt") and f != os.path.basename(ki_meta_path)
    ])

    for i, fname in enumerate(txt_files):
        gene = fname.replace(".txt", "")
        txt_path = os.path.join(ki_dir, fname)

        if (i + 1) % 10 == 0 or i == 0 or i == len(txt_files) - 1:
            print(f"[Dataset][KI] Progress: {i+1}/{len(txt_files)} | kept={len(dataset)}")

        if gene not in gene_to_wt_sample:
            continue
        if gene not in ki_gene_meta:
            continue
        if txt_contains_N(txt_path):
            continue

        try:
            meta = ki_gene_meta[gene]
            dna_in, atac_in, tss_loc, direction = prepare_model_input(
                txt_path,
                meta["tss_loc"],
                meta["direction"],
                genome_dict,
                narrowPeak_Reader,
                pre_processing,
                tokenizer,
                pre_pro
            )

            sample_meta = {
                "gene_name": gene,
                "source": "KI",
                "log2FC": float(meta["log2FC"]) if meta.get("log2FC") is not None else None,
                "true_direction": meta.get("tutorial_direction"),
                "tss_loc": meta["tss_loc"]
            }
            sample = [dna_in, atac_in, tss_loc, 0.0, direction, sample_meta]
            dataset.append(sample)

        except Exception:
            continue

    print(f"[Dataset][KI] Done. Samples={len(dataset)}")
    return dataset


# ============================================================
# Model loading / inference
# ============================================================
def load_model_components(checkpoint_dir: str, cuda_device: int = 0) -> dict:
    device = torch.device("cuda", cuda_device)

    print("[Model] Loading model components...")

    model_parts = {
        "device": device,
        "dna_embed":  torch.load(os.path.join(checkpoint_dir, "dna_embed.pkl"),     map_location="cpu", weights_only=False).to(device),
        "atac_embed": torch.load(os.path.join(checkpoint_dir, "atac_embed.pkl"),    map_location="cpu", weights_only=False).to(device),
        "pos1_embed": torch.load(os.path.join(checkpoint_dir, "pos1_embed.pkl"),    map_location="cpu", weights_only=False).to(device),
        "pos2_embed": torch.load(os.path.join(checkpoint_dir, "pos2_embed.pkl"),    map_location="cpu", weights_only=False).to(device),
        "tss_embed":  torch.load(os.path.join(checkpoint_dir, "tss_embed.pkl"),     map_location="cpu", weights_only=False).to(device),
        "pad_embed":  torch.load(os.path.join(checkpoint_dir, "pad_embed.pkl"),     map_location="cpu", weights_only=False).to(device),
        "encoder_1":  torch.load(os.path.join(checkpoint_dir, "transformer_1.pkl"), map_location="cpu", weights_only=False).to(device),
        "encoder_2":  torch.load(os.path.join(checkpoint_dir, "transformer_2.pkl"), map_location="cpu", weights_only=False).to(device),
        "encoder_3":  torch.load(os.path.join(checkpoint_dir, "transformer_3.pkl"), map_location="cpu", weights_only=False).to(device),
        "atten_pool": torch.load(os.path.join(checkpoint_dir, "atten_pool.pkl"),    map_location="cpu", weights_only=False).to(device),
        "ff_net":     torch.load(os.path.join(checkpoint_dir, "feedforward.pkl"),   map_location="cpu", weights_only=False).to(device),
    }

    print("[Model] Model loaded.\n")
    return model_parts


def infer_expression_from_prepared_sample(sample, model_parts: dict) -> float:
    device = model_parts["device"]

    dna_embed  = model_parts["dna_embed"]
    atac_embed = model_parts["atac_embed"]
    pos1_embed = model_parts["pos1_embed"]
    pos2_embed = model_parts["pos2_embed"]
    tss_embed  = model_parts["tss_embed"]
    pad_embed  = model_parts["pad_embed"]
    encoder_1  = model_parts["encoder_1"]
    encoder_2  = model_parts["encoder_2"]
    encoder_3  = model_parts["encoder_3"]
    atten_pool = model_parts["atten_pool"]
    ff_net     = model_parts["ff_net"]

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

    pos1 = torch.ones(129, dtype=int).to(device)
    for i in range(len(pos1)):
        pos1[i] += i

    pos2 = torch.ones(8, dtype=int).to(device)
    for i in range(len(pos2)):
        pos2[i] += i

    pos3 = torch.zeros(150, dtype=int).to(device)
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

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=True):
            CLS = dna_embed(torch.ones(dna_in.shape[0] * 8, 1, dtype=int).to(device))
            x_POS_1 = pos1_embed(pos1)

            x_mul = dna_embed(dna_in.int().reshape(dna_in.shape[0] * 8, 128)) + \
                    atac_embed(sig_in.int().reshape(dna_in.shape[0] * 8, 128))

            x_embed = torch.cat((CLS, x_mul), dim=1)
            x_enc_1 = encoder_1(x_embed + x_POS_1)[:, 0, :].reshape(dna_in.shape[0], 8, 2048)

            x_POS_2 = pos2_embed(pos2)
            x_enc_2 = encoder_2(x_enc_1 + x_POS_2)
            x_enc_2 = rearrange(x_enc_2, "b n d -> b d n")
            x_enc_2 = atten_pool(x_enc_2)
            x_enc_2 = rearrange(x_enc_2, "b d n -> b n d").squeeze(1)

            x_pad = pad_embed(torch.zeros(150 - x_enc_2.shape[0], dtype=int).to(device))
            x_eb3 = torch.cat((x_enc_2, x_pad), dim=0)
            x_POS_3 = tss_embed(pos3)
            x_enc_3 = encoder_3(x_eb3 + x_POS_3)
            result = ff_net(x_enc_3).squeeze(1)

    return result.float().item()


def build_gene_to_sample_map(dataset: List[list]) -> Dict[str, list]:
    gene_to_sample = {}
    for sample in dataset:
        if len(sample) < 6:
            continue
        meta = sample[5]
        gene_name = meta.get("gene_name", None)
        if gene_name is not None and gene_name not in gene_to_sample:
            gene_to_sample[gene_name] = sample
    return gene_to_sample


def predict_delta_for_common_genes(
    wt_dataset: List[list],
    ki_dataset: List[list],
    model_parts: dict,
    max_eval_samples: Optional[int] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    wt_gene_map = build_gene_to_sample_map(wt_dataset)
    ki_gene_map = build_gene_to_sample_map(ki_dataset)

    common_genes = sorted(set(wt_gene_map.keys()) & set(ki_gene_map.keys()))
    print(f"[Inference] Common genes between WT and KI: {len(common_genes)}")

    if max_eval_samples is not None and max_eval_samples < len(common_genes):
        rng = random.Random(random_seed)
        selected = common_genes[:]
        rng.shuffle(selected)
        common_genes = sorted(selected[:max_eval_samples])
        print(f"[Inference] Subsampled to {len(common_genes)} genes")
    else:
        print(f"[Inference] Using all {len(common_genes)} genes")

    wt_pred_cache = {}
    records = []

    for i, gene_name in enumerate(common_genes):
        if (i + 1) % 10 == 0 or i == 0 or i == len(common_genes) - 1:
            print(f"[Inference] Progress: {i+1}/{len(common_genes)}")

        wt_sample = wt_gene_map[gene_name]
        ki_sample = ki_gene_map[gene_name]

        try:
            if gene_name not in wt_pred_cache:
                wt_pred_cache[gene_name] = infer_expression_from_prepared_sample(wt_sample, model_parts)

            wt_pred = wt_pred_cache[gene_name]
            ki_pred = infer_expression_from_prepared_sample(ki_sample, model_parts)
            pred_delta = ki_pred - wt_pred

            ki_meta = ki_sample[5]
            true_direction = ki_meta.get("true_direction")
            log2fc = ki_meta.get("log2FC")

            pred_direction = "up" if pred_delta > 0 else ("down" if pred_delta < 0 else "flat")

            records.append({
                "gene_name": gene_name,
                "pred_delta": float(pred_delta),
                "pred_direction": pred_direction,
                "true_direction": true_direction,
                "log2FC": float(log2fc) if log2fc is not None else np.nan
            })

        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f"[Inference] Done. Successful predictions: {len(df)}")
    return df


# ============================================================
# Metrics
# ============================================================
def compute_binary_classification_metrics(result_df: pd.DataFrame) -> Dict[str, float]:
    df = result_df.copy()
    df = df[df["true_direction"].isin(["up", "down"])].copy()
    df = df[df["pred_direction"].isin(["up", "down"])].copy()

    if len(df) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "up_accuracy": np.nan,
            "down_accuracy": np.nan
        }

    y_true = df["true_direction"].tolist()
    y_pred = df["pred_direction"].tolist()

    labels = ["up", "down"]

    def f1_for_label(label: str) -> float:
        tp = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    accuracy = np.mean([yt == yp for yt, yp in zip(y_true, y_pred)])

    up_mask = [yt == "up" for yt in y_true]
    down_mask = [yt == "down" for yt in y_true]

    up_accuracy = np.mean([yt == yp for yt, yp in zip(y_true, y_pred) if yt == "up"]) if any(up_mask) else np.nan
    down_accuracy = np.mean([yt == yp for yt, yp in zip(y_true, y_pred) if yt == "down"]) if any(down_mask) else np.nan

    macro_f1 = np.mean([f1_for_label(x) for x in labels])

    return {
        "n": int(len(df)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "up_accuracy": float(up_accuracy) if not np.isnan(up_accuracy) else np.nan,
        "down_accuracy": float(down_accuracy) if not np.isnan(down_accuracy) else np.nan
    }


def compute_delta_logfc_correlation(result_df: pd.DataFrame) -> Dict[str, float]:
    df = result_df.copy()
    df = df.dropna(subset=["pred_delta", "log2FC"])

    if len(df) < 2:
        return {
            "n": int(len(df)),
            "pcc_delta_vs_log2fc": np.nan
        }

    corr = np.corrcoef(df["pred_delta"].values, df["log2FC"].values)[0, 1]
    return {
        "n": int(len(df)),
        "pcc_delta_vs_log2fc": float(corr)
    }


def summarize_results(result_df: pd.DataFrame) -> Dict[str, float]:
    cls_metrics = compute_binary_classification_metrics(result_df)
    corr_metrics = compute_delta_logfc_correlation(result_df)

    summary = {}
    summary.update(cls_metrics)
    summary.update(corr_metrics)
    return summary


def print_summary(summary: Dict[str, float]):
    print("\n================ Tutorial Summary ================")
    print(f"Evaluated samples      : {summary.get('n', 'NA')}")
    print(f"Accuracy               : {summary.get('accuracy', np.nan):.4f}")
    print(f"Macro-F1               : {summary.get('macro_f1', np.nan):.4f}")
    print(f"Up accuracy            : {summary.get('up_accuracy', np.nan):.4f}")
    print(f"Down accuracy          : {summary.get('down_accuracy', np.nan):.4f}")
    print("==================================================\n")


# ============================================================
# High-level tutorial helpers
# ============================================================
def collect_gene_names_from_rows(rows: List[dict]) -> List[str]:
    genes = []
    seen = set()
    for row in rows:
        gene = str(row.get("Gene", "")).strip()
        if gene and gene not in seen:
            seen.add(gene)
            genes.append(gene)
    return genes


def save_results_table(result_df: pd.DataFrame, output_path: str):
    ensure_dir(os.path.dirname(output_path))
    result_df.to_csv(output_path, index=False)
    print(f"[Output] Result table saved to: {output_path}")


def save_summary_json(summary: Dict[str, float], output_path: str):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Output] Summary saved to: {output_path}")

def prepare_tutorial_gene_rows(
    up_csv_path: str,
    down_csv_path: str,
    max_samples_per_direction: Optional[int] = None,
    random_seed: int = 42
) -> Tuple[List[dict], str, str, str]:
    parsed_up = parse_tf_deg_filename(up_csv_path)
    parsed_down = parse_tf_deg_filename(down_csv_path)

    if parsed_up is None:
        raise ValueError(f"Cannot parse filename: {up_csv_path}")
    if parsed_down is None:
        raise ValueError(f"Cannot parse filename: {down_csv_path}")

    cell_up, tf_up, dir_up = parsed_up
    cell_down, tf_down, dir_down = parsed_down

    if cell_up != cell_down:
        raise ValueError("Up and down CSV files must use the same cell line")
    if tf_up != tf_down:
        raise ValueError("Up and down CSV files must use the same TF")
    if dir_up.lower() != "up":
        raise ValueError("The first CSV must be the 'up' file")
    if dir_down.lower() != "down":
        raise ValueError("The second CSV must be the 'down' file")

    rows = merge_up_down_rows(
        up_csv_path=up_csv_path,
        down_csv_path=down_csv_path,
        max_samples_per_direction=max_samples_per_direction,
        random_seed=random_seed
    )

    print(f"[Tutorial] Merged up/down gene rows: {len(rows)}")
    print(f"[Tutorial] Cell line: {cell_up}")
    print(f"[Tutorial] TF: {tf_up}")

    return rows, cell_up, tf_up, f"{cell_up}_{tf_up}"


def build_tutorial_directory_tree(
    tutorial_root: str,
    run_name: str
) -> Dict[str, str]:
    """
    Create a compact directory tree for the tutorial run.
    """
    run_dir = os.path.join(tutorial_root, run_name)

    paths = {
        "run_dir": run_dir,
        "wt_dir": os.path.join(run_dir, "wild_type_inputs"),
        "ki_dir": os.path.join(run_dir, "ki_inputs"),
        "meta_dir": os.path.join(run_dir, "meta"),
        "result_dir": os.path.join(run_dir, "results")
    }

    for p in paths.values():
        ensure_dir(p)

    return paths


def save_gene_rows_as_csv(rows: List[dict], output_csv_path: str):
    """
    Optional helper for debugging/tutorial reproducibility.
    """
    if len(rows) == 0:
        df = pd.DataFrame(columns=["Gene", "log2FC", "_tutorial_direction"])
    else:
        df = pd.DataFrame(rows)

    ensure_dir(os.path.dirname(output_csv_path))
    df.to_csv(output_csv_path, index=False)
    print(f"[Output] Tutorial gene table saved to: {output_csv_path}")
   


