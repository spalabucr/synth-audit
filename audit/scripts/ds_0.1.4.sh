EXP_NAME=ds_0.1.4
N_REPS=10000
N_PROCS=32
DATA_NAME=adult

MODEL=DSynthPB_v014
if [[ "$MODEL" == "DPartPB" || "$MODEL" == "DSynthPB" ]]; then
    NEIGHBOUR=edit
else
    NEIGHBOUR=addremove
fi

for EPSILON in 1.0 2.0 4.0 10.0
do
    echo "[$(date +%F_%T)] $MODEL, $EPSILON" | tee -a logs/$EXP_NAME.txt
    python3 prep_synths.py --data_name $DATA_NAME --n_reps $N_REPS --model $MODEL --epsilon $EPSILON \
    --neighbour $NEIGHBOUR --worstcase --narrow --use_provisional --n_procs $N_PROCS \
    --out_dir exp_data/$EXP_NAME/$DATA_NAME --save_model_only
done
echo "[$(date +%F_%T)] Done" | tee -a logs/$EXP_NAME.txt
