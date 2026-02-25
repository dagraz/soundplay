#!/usr/bin/env bash
# reich.sh — Steve Reich-style phasing from a chord SPX file.
#
# Each spectral component of the chord is looped with a slightly different
# period (achieved by adding progressively more tail silence), so the
# components gradually drift in and out of phase with each other.
#
# Usage:
#   reich.sh INPUT.spx OUTPUT.wav [OPTIONS]
#
# Options:
#   --loops N        Number of times to loop each component (default: 32)
#   --drift PCT      Extra padding per component as % of chord duration (default: 5)
#   --no-remainder   Exclude the remainder from the mix
#   --plot           Save a spectrogram PNG for the input and each component
#   --plot-notes     Add note-frequency guidelines to plots (implies --plot)
#   --keep-parts     Don't delete the temporary part files
#   --output-dir DIR Use DIR for temp files (default: system temp)
#
# Example:
#   reich.sh chord.spx reich_out.wav --loops 48 --drift 3
#   reich.sh chord.spx reich_out.wav --plot-notes --keep-parts --output-dir out/

set -euo pipefail

# ------------------------------------------------------------------ defaults
LOOPS=32
DRIFT_PCT=5
INCLUDE_REMAINDER=1
PLOT=0
PLOT_NOTES=0
KEEP_PARTS=0
WORK_DIR=""

# ------------------------------------------------------------------ args
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 INPUT.spx OUTPUT.wav [--loops N] [--drift PCT] [--no-remainder] [--plot] [--plot-notes] [--keep-parts] [--output-dir DIR]" >&2
    exit 1
fi

INPUT="$1";  shift
OUTPUT="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loops)       LOOPS="$2";        shift 2 ;;
        --drift)       DRIFT_PCT="$2";    shift 2 ;;
        --no-remainder) INCLUDE_REMAINDER=0; shift ;;
        --plot)        PLOT=1;            shift ;;
        --plot-notes)  PLOT=1; PLOT_NOTES=1; shift ;;
        --keep-parts)  KEEP_PARTS=1;      shift ;;
        --output-dir)  WORK_DIR="$2";     shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ------------------------------------------------------------------ setup
VENV_DIR="$(dirname "$0")/.venv"
ACTIVATE="$VENV_DIR/bin/activate"
if [[ -f "$ACTIVATE" ]]; then
    source "$ACTIVATE"
fi

if [[ -z "$WORK_DIR" ]]; then
    WORK_DIR="$(mktemp -d)"
    CREATED_WORK_DIR=1
else
    mkdir -p "$WORK_DIR"
    CREATED_WORK_DIR=0
fi

cleanup() {
    if [[ $KEEP_PARTS -eq 0 && $CREATED_WORK_DIR -eq 1 ]]; then
        rm -rf "$WORK_DIR"
    else
        echo "Temp files kept in: $WORK_DIR" >&2
    fi
}
trap cleanup EXIT

echo "Input  : $INPUT" >&2
echo "Output : $OUTPUT" >&2
echo "Loops  : $LOOPS" >&2
echo "Drift  : ${DRIFT_PCT}% per component" >&2
echo "" >&2

# ------------------------------------------------------------------ plot helper
plot_spx() {
    local spx="$1"
    local png="${spx%.spx}.png"
    local title="$2"
    local extra_args=()
    [[ $PLOT_NOTES -eq 1 ]] && extra_args+=(--notes --note-labels)
    sp-plot "${extra_args[@]}" --title "$title" "$spx" "$png" 2>/dev/null
    echo "  Plot: $png" >&2
}

# ------------------------------------------------------------------ decompose
PARTS_DIR="$WORK_DIR/parts"
mkdir -p "$PARTS_DIR"

echo "==> Decomposing..." >&2
sp-decompose "$INPUT" --output-dir "$PARTS_DIR" >&2

# Plot the main input
if [[ $PLOT -eq 1 ]]; then
    echo "==> Plotting..." >&2
    plot_spx "$INPUT" "$(basename "$INPUT") — full chord"
fi

# Collect component files (everything except remainder unless included)
STEM="$(basename "${INPUT%.spx}")"
mapfile -t ALL_PARTS < <(ls "$PARTS_DIR/${STEM}_"*hz.spx 2>/dev/null | sort)

if [[ $INCLUDE_REMAINDER -eq 1 && -f "$PARTS_DIR/${STEM}_remainder.spx" ]]; then
    ALL_PARTS+=("$PARTS_DIR/${STEM}_remainder.spx")
fi

N=${#ALL_PARTS[@]}
if [[ $N -eq 0 ]]; then
    echo "Error: no component files found in $PARTS_DIR" >&2
    exit 1
fi

echo "" >&2
echo "==> Found $N component(s). Padding and looping..." >&2

# Plot each component
if [[ $PLOT -eq 1 ]]; then
    for PART in "${ALL_PARTS[@]}"; do
        plot_spx "$PART" "$(basename "${PART%.spx}")"
    done
fi

# ------------------------------------------------------------------ get duration (seconds, float)
DURATION=$(sp-info --json "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['duration'])")
echo "Chord duration: ${DURATION}s" >&2

# ------------------------------------------------------------------ pad + loop each component
LOOPED=()
for i in "${!ALL_PARTS[@]}"; do
    PART="${ALL_PARTS[$i]}"
    NAME="$(basename "${PART%.spx}")"

    # Padding = i * drift_pct% of chord duration
    PAD=$(python3 -c "print(f'{$i * $DRIFT_PCT / 100.0 * $DURATION:.4f}')")

    LOOPED_FILE="$WORK_DIR/looped_${i}_${NAME}.spx"

    echo "  [$((i+1))/$N] $(basename "$PART")  pad=${PAD}s" >&2
    sp-expand --pad-end "$PAD" "$PART" - | sp-loop -n "$LOOPS" - "$LOOPED_FILE" 2>/dev/null

    LOOPED+=("$LOOPED_FILE")
done

# ------------------------------------------------------------------ join
echo "" >&2
echo "==> Joining ${#LOOPED[@]} looped components..." >&2
MIXED_SPX="$WORK_DIR/mixed.spx"
sp-join --no-strict "${LOOPED[@]}" -o "$MIXED_SPX" >&2

# ------------------------------------------------------------------ resynth
echo "" >&2
echo "==> Resynthesising..." >&2
sp-resynth "$MIXED_SPX" "$OUTPUT" >&2

echo "" >&2
echo "Done: $OUTPUT" >&2
