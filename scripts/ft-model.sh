w2v_path=$1
split=$2
ngpu=$3

seed=1

root=example

valid_subset=val_1

data=$4
train_subset=$5
valid_subset=$6
user_dir=`pwd`/sew_asapp


export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

function train {
    case $split in
    10m)
        task=ft-10m
        config=base_10m
    ;;
    1h)
        task=ft-1h
        config=base_1h
    ;;
    10h)
        task=ft-10h
        config=base_10h
    ;;
    100h)
        task=ft-100h
        config=base_100h
    ;;
    *)
        echo "unknown task ft-$split"
        exit
    ;;
    esac

    echo $task
    tag=${tag}-s${seed}

    save=`pwd`/save-${task}/${root}/${tag}
    tb_save=`pwd`/tb-logs-${task}/${root}/${tag}


    fairseq-hydra-train \
        hydra.run.dir=$save \
        hydra.output_subdir=$save \
        common.user_dir=$user_dir \
        common.tensorboard_logdir=$tb_save \
        common.log_interval=100 \
        task.data=$data \
        model.w2v_path=$w2v_path \
        dataset.train_subset=$train_subset \
        dataset.valid_subset=$valid_subset \
        distributed_training.distributed_world_size=$ngpu \
        common.seed=$seed \
        dataset.max_tokens=$((3200000 * 2))  \
        optimization.update_freq="[$((8 / $ngpu / 2))]" \
        --config-dir config/finetuning \
        --config-name $config

}

pre_tag=${w2v_path##*/} # get basename (/my/path/to/file.pt --> file.pt)
tag=${pre_tag%.*}-ft-${split}
train
