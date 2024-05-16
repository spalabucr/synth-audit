EXP_NAME=test_worst_case
EPSILON=4.0
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

    for NARROW in TRUE FALSE
    do
        for REPEAT in TRUE FALSE
        do
            args=()

            if [[ "$NARROW" == "TRUE" ]]; then
                args+=(--narrow)
            fi

            if [[ "$REPEAT" == "TRUE" ]]; then
                args+=(--repeat_target)
            fi

            echo "[$(date +%F_%T)] $MODEL, ${args[@]}" | tee -a logs/$EXP_NAME.txt
            python3 prep_synths.py --data_name $DATA_NAME --n_reps $N_REPS --model $MODEL --epsilon $EPSILON \
            --neighbour $NEIGHBOUR --worstcase "${args[@]}" --use_provisional --n_procs $N_PROCS \
            --out_dir exp_data/$EXP_NAME/$DATA_NAME 
        done
    done
done
echo "[$(date +%F_%T)] Done" | tee -a logs/$EXP_NAME.txt
