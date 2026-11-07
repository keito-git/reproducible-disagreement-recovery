# prep, model download (own dir), frozen-LLM embeddings, majority-label models, second datasets, analysis.
PY=${PY:-python}
cd "$(dirname "$0")"
set -e
VT=../data/votes; SP=../data/split; RAW=../data/raw; R=../results/s13
export PIP_CONFIG_FILE=/dev/null
[ -f $VT/brighter_deu.pkl ] || $PY s11_prep.py --raw $RAW --votes $VT --split $SP
XL=../models/xlm-roberta-large
[ -f $XL/config.json ] || $PY -c "from huggingface_hub import snapshot_download; snapshot_download('FacebookAI/xlm-roberta-large', local_dir='$XL', allow_patterns=['*.json','*.safetensors','sentencepiece.bpe.model','tokenizer*'])"
[ -f $R/llm/dices_990.pkl ] || $PY s12_llm.py --votes $VT --raw $RAW --split $SP --model Qwen/Qwen2.5-7B-Instruct --out $R/llm
J="$PY s4_judge.py train --votes $VT --raw $RAW --split $SP --lr 2e-5"
for S in 0 1 2; do
  O=$R/hard/brighter_eng/n1868_thin7/seed$S; [ -f $O/info.json ] || $J --data brighter_eng --model roberta-large --seed $S --ntrain 1868 --thin 7 --hard --out $O
  O=$R/hard/chaosnli/rl_thin7/seed$S; [ -f $O/info.json ] || $J --data chaosnli --model roberta-large --seed $S --thin 7 --hard --out $O
  for F in 0 1 2 3 4; do
    O=$R/hard/dices_990/thin7/fold$F/seed$S; [ -f $O/info.json ] || $J --data dices_990 --model roberta-large --seed $S --fold $F --thin 7 --hard --out $O
    O=$R/dices_350/full/fold$F/seed$S; [ -f $O/info.json ] || $J --data dices_350 --model roberta-large --seed $S --fold $F --out $O
    O=$R/dices_350/thin7/fold$F/seed$S; [ -f $O/info.json ] || $J --data dices_350 --model roberta-large --seed $S --fold $F --thin 7 --out $O
  done
done
# BRIGHTER German: XLM-R large, lr chosen on dev JSD (seed 0), then 3 seeds
JG="$PY s4_judge.py train --votes $VT --raw $RAW --split $SP --data brighter_deu --model $XL"
for LR in 1e-5 2e-5; do O=$R/brighter_deu/sel/lr$LR; [ -f $O/info.json ] || $JG --lr $LR --seed 0 --out $O; done
BEST=$($PY -c "import json;print(min(['1e-5','2e-5'],key=lambda l:json.load(open('$R/brighter_deu/sel/lr'+l+'/info.json'))['dev_jsd']))")
echo "deu best lr $BEST"
mkdir -p $R/brighter_deu/main; [ -d $R/brighter_deu/main/seed0 ] || cp -r $R/brighter_deu/sel/lr$BEST $R/brighter_deu/main/seed0
for S in 1 2; do O=$R/brighter_deu/main/seed$S; [ -f $O/info.json ] || $JG --lr $BEST --seed $S --out $O; done
echo S13_TRAIN_DONE
OMP_NUM_THREADS=4 nice $PY s13_map.py --votes $VT --raw $RAW --split $SP --res ../results --out ../results/s13_map.json
echo S13_DONE
