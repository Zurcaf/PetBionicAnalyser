#!/usr/bin/env python3
"""
petBionic CSV Cleaner
=====================
Corrige dois bugs do firmware antigo nos CSVs de teste:

  Bug 1 – "20 iguais": load_cell_est_kg congelado em blocos de 20 linhas.
    Causa: o firmware antigo chamava get_units(1) (2ª leitura bloqueante do HX711)
    para calcular kg depois de já ter consumido a amostra com readRaw(). O HX711
    ainda não tinha nova conversão, dava timeout, e o kg ficava no valor anterior.

  Bug 2 – "valores ligeiramente acima do 0": load_cell_raw ≈ 2500 intercalado
    com leituras reais. Causa: readRaw() não tinha return no caminho de timeout
    → comportamento indefinido → registo do ESP32 ≈ 2500 como valor devolvido.

Limpeza:
  1. Remove linhas com raw no cluster de lixo (≈1500–3800 contagens).
  2. Remove valores impossíveis (outliers de ADC, |raw| > 2 000 000).
  3. Recalcula load_cell_est_kg correctamente: (raw - offset) / factor.

Calibração padrão derivada dos dados de aquecimento:
  offset = -16 000  (cluster sem carga ≈ -13 918; kg congelado = 0.112 kg
                      → -13 918 - 0.112×18 570 ≈ -15 998 ≈ -16 000)
  factor = 18 570   (kHx711CalibrationFactor no firmware)

Uso:
  python csv_cleaner.py                        # processa todo TestData/
  python csv_cleaner.py caminho/ficheiro.csv   # ficheiro único
  python csv_cleaner.py --offset -15500 --factor 18570
  python csv_cleaner.py --in-place            # substitui os originais (cuidado!)
  python csv_cleaner.py --output-dir cleaned/ # pasta de saída separada
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


# ── calibração padrão ─────────────────────────────────────────────────────────

DEFAULT_OFFSET = -16_000.0   # contagens HX711 correspondentes a 0 kg
DEFAULT_FACTOR = 18_570.0    # contagens / kg  (kHx711CalibrationFactor)

# ── detecção de lixo ──────────────────────────────────────────────────────────

OUTLIER_MAX    = 2_000_000   # |raw| acima disto = glitch óbvio do ADC
GARBAGE_MARGIN = 1_500       # janela em torno do pico de lixo a remover


# ─────────────────────────────────────────────────────────────────────────────

def detect_garbage_range(raw_values: list[int]) -> tuple[int, int]:
    """
    Encontra o cluster de lixo: pico abrupto nos valores positivos de raw.
    Devolve (low, high) que deve ser removido.
    Se não houver pico claro, devolve (-1, -1).
    """
    pos = [v for v in raw_values if 0 < v < OUTLIER_MAX]
    if len(pos) < 50:
        return -1, -1

    bin_width = 200
    hist = Counter(v // bin_width for v in pos)
    if not hist:
        return -1, -1

    peak_bin = max(hist, key=hist.get)
    peak_count = hist[peak_bin]

    # Bins vizinhos (exclui o pico)
    neighbors = [hist.get(peak_bin + d, 0) for d in range(-5, 6) if d != 0]
    mean_neighbor = sum(neighbors) / len(neighbors) if neighbors else 1

    if peak_count < 8 * mean_neighbor:
        return -1, -1   # sem pico claro

    peak_center = peak_bin * bin_width + bin_width // 2
    return peak_center - GARBAGE_MARGIN, peak_center + GARBAGE_MARGIN


def is_valid_raw(raw: int, garbage_low: int, garbage_high: int) -> bool:
    if abs(raw) > OUTLIER_MAX:
        return False
    if garbage_low <= raw <= garbage_high:
        return False
    return True


def recalc_kg(raw: int, offset: float, factor: float) -> float:
    return (raw - offset) / factor


def clean_file(
    src: Path,
    dst: Path,
    offset: float,
    factor: float,
    garbage_low: int,
    garbage_high: int,
) -> tuple[int, int]:
    """
    Limpa um CSV.  Devolve (linhas_originais, linhas_removidas).
    """
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0, 0
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        return 0, 0

    if "load_cell_raw" not in fieldnames:
        print(f"  AVISO: {src.name} não tem coluna load_cell_raw — ignorado.")
        return 0, 0

    kept = []
    removed = 0
    for r in rows:
        try:
            raw = int(r["load_cell_raw"])
        except ValueError:
            removed += 1
            continue

        if not is_valid_raw(raw, garbage_low, garbage_high):
            removed += 1
            continue

        if "load_cell_est_kg" in fieldnames:
            r["load_cell_est_kg"] = f"{recalc_kg(raw, offset, factor):.3f}"

        kept.append(r)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    return len(rows), removed


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpa CSVs petBionic (remove lixo HX711, recalcula kg).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input", nargs="?",
        help="Ficheiro CSV ou directório (padrão: TestData/ junto a este script).",
    )
    parser.add_argument(
        "--offset", type=float, default=DEFAULT_OFFSET,
        help=f"Offset de calibração em contagens (padrão: {DEFAULT_OFFSET}).",
    )
    parser.add_argument(
        "--factor", type=float, default=DEFAULT_FACTOR,
        help=f"Factor de calibração em contagens/kg (padrão: {DEFAULT_FACTOR}).",
    )
    parser.add_argument(
        "--garbage-low", type=int, default=None,
        help="Limite inferior do cluster de lixo (auto-detectado se omitido).",
    )
    parser.add_argument(
        "--garbage-high", type=int, default=None,
        help="Limite superior do cluster de lixo (auto-detectado se omitido).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Pasta de saída (padrão: ficheiro_cleaned.csv junto ao original).",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Sobrescreve os ficheiros originais (não recomendado sem backup).",
    )
    args = parser.parse_args()

    # localiza CSVs
    script_dir = Path(__file__).resolve().parent
    if args.input:
        root = Path(args.input)
    else:
        root = script_dir.parent / "TestData"

    if root.is_file():
        csv_files = [root]
    elif root.is_dir():
        csv_files = sorted(root.rglob("*.csv"))
    else:
        sys.exit(f"Caminho não encontrado: {root}")

    if not csv_files:
        sys.exit(f"Nenhum ficheiro CSV encontrado em {root}")

    print(f"petBionic CSV Cleaner")
    print(f"{'─'*50}")
    print(f"Ficheiros encontrados : {len(csv_files)}")
    print(f"Calibração            : offset={args.offset:.0f}  factor={args.factor:.1f} cnt/kg")

    # auto-detecta cluster de lixo lendo todos os raw de uma vez
    if args.garbage_low is None or args.garbage_high is None:
        print("A detectar cluster de lixo...", end=" ", flush=True)
        all_raw: list[int] = []
        for p in csv_files:
            try:
                with p.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        v = row.get("load_cell_raw", "").strip()
                        if v:
                            try:
                                all_raw.append(int(v))
                            except ValueError:
                                pass
            except Exception:
                pass

        g_low, g_high = detect_garbage_range(all_raw)
        if g_low == -1:
            g_low, g_high = 1500, 3800
            print(f"sem pico claro → usando padrão [{g_low}, {g_high}]")
        else:
            print(f"detectado [{g_low}, {g_high}]")

        if args.garbage_low is not None:
            g_low = args.garbage_low
        if args.garbage_high is not None:
            g_high = args.garbage_high
    else:
        g_low, g_high = args.garbage_low, args.garbage_high
        print(f"Cluster de lixo       : [{g_low}, {g_high}] (manual)")

    print(f"Outliers removidos    : |raw| > {OUTLIER_MAX:,}")
    print(f"{'─'*50}")

    total_orig = total_removed = 0

    for src in csv_files:
        if args.in_place:
            dst = src
        elif args.output_dir:
            try:
                rel = src.relative_to(root if root.is_dir() else root.parent)
            except ValueError:
                rel = Path(src.name)
            dst = args.output_dir / rel
        else:
            dst = src.with_stem(src.stem + "_cleaned")

        orig, removed = clean_file(src, dst, args.offset, args.factor, g_low, g_high)

        total_orig    += orig
        total_removed += removed
        kept = orig - removed
        pct  = 100.0 * removed / orig if orig else 0.0

        tag = "" if dst == src else f"  → {dst.name}"
        print(f"  {src.name}")
        print(f"    {orig:>6} linhas  →  {kept:>6} mantidas  ({removed} removidas, {pct:.0f}%){tag}")

    print(f"{'─'*50}")
    kept_total = total_orig - total_removed
    pct_total  = 100.0 * total_removed / total_orig if total_orig else 0.0
    print(f"TOTAL  {total_orig:,} linhas  →  {kept_total:,} mantidas  ({total_removed:,} removidas, {pct_total:.0f}%)")

    # Verificação de sanidade: amostra de kg recalculados
    print(f"\nVerificação de sanidade (primeiras leituras válidas do 1º ficheiro):")
    if csv_files:
        cleaned = (
            csv_files[0].with_stem(csv_files[0].stem + "_cleaned")
            if not args.in_place and not args.output_dir
            else (
                args.output_dir / csv_files[0].relative_to(root if root.is_dir() else root.parent)
                if args.output_dir else csv_files[0]
            )
        )
        try:
            with cleaned.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for i, r in enumerate(rows[:8]):
                print(f"  raw={int(r['load_cell_raw']):>8d}  kg={float(r['load_cell_est_kg']):>7.3f}")
        except Exception as e:
            print(f"  (não foi possível ler: {e})")


if __name__ == "__main__":
    main()
