export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_CONFIG_FILE=/dev/null
PY=${PY:-python}
cd "$(dirname "$0")"
set -e
VT=../data/votes; SP=../data/split; RAW=../data/raw; R=../results/s9
J="$PY s9_direct.py --votes $VT --raw $RAW --split $SP"
for S in 0 1 2; do
  O=$R/brighter_eng/full/seed$S;        [ -f $O/info.json ] || $J --data brighter_eng --seed $S --out $O
  O=$R/brighter_eng/n1868_thin7/seed$S; [ -f $O/info.json ] || $J --data brighter_eng --seed $S --ntrain 1868 --thin 7 --out $O
  O=$R/chaosnli/rl_thin7/seed$S;        [ -f $O/info.json ] || $J --data chaosnli --seed $S --thin 7 --out $O
  for F in 0 1 2 3 4; do
    O=$R/dices_990/thin7/fold$F/seed$S; [ -f $O/info.json ] || $J --data dices_990 --seed $S --fold $F --thin 7 --out $O
  done
done
echo S9_DONE
