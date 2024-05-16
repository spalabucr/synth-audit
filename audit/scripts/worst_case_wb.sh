EXP_NAME=worst_case_wb
N_REPS=10000
N_PROCS=32
DATA_NAME=adult

for MODEL in DPartPB DSynthPB NIST_MST MST DPWGAN DPWGANCity
do
    if [[ "$MODEL" == "DPartPB" || "$MODEL" == "DSynthPB" ]]; then
        NEIGHBOUR=edit
    else
        NEIGHBOUR=addremove
    fi

    # different worst case setting for different SDGs
    args=()
    if [[ "$MODEL" == "DSynthPB" ]]; then
        args+=(--narrow)
    elif [[ "$MODEL" == "DPartPB" ]]; then
        args+=(--narrow)
        args+=(--repeat)
    else
        args+=(--repeat)
    fi

    for EPSILON in 1.0 2.0 4.0 10.0
    do
        echo "[$(date +%F_%T)] $MODEL (${args[@]}), $EPSILON" | tee -a logs/$EXP_NAME.txt
        python3 prep_synths.py --data_name $DATA_NAME --n_reps $N_REPS --model $MODEL --epsilon $EPSILON \
        --neighbour $NEIGHBOUR --worstcase "${args[@]}" --use_provisional --n_procs $N_PROCS \
        --out_dir exp_data/$EXP_NAME/$DATA_NAME --save_model_only
    done
done
echo "[$(date +%F_%T)] Done" | tee -a logs/$EXP_NAME.txt
