
function train(){
    c1=$1
    c2=$2
    factor=$3
    echo "###############" `date +"%Y-%m-%d %H:%M:%S"` "train $c1 $c2 $factor start" 
    python3 /home/yangdongxu/work/model/Leopard_pytorchVersion/train.py \
    --factor $factor \
    --fa /shared/zhangyuxuan/data/annotation/hg38.fa \
    --train  $c1 \
    --valid $c2 \
    --data "/shared/zhangyuxuan/projects/Model/scripts/1.finetune/18.Leopard/data/h5data/" \
    --gpu 3 \
    -b 128 \
    -r "200" \
    --num_workers 32 > train_logs/train_${c1}_${c2}_${factor}.log 2>&1 
    echo "###############" `date +"%Y-%m-%d %H:%M:%S"` "train $c1 $c2 $factor done" 
}

mkdir -p train_logs

# train ABC ABC MAX 
# train A549 GM12878 MAX
# train GM12878 H1-hESC MAX
# train H1-hESC HCT116 MAX
# train HCT116 HeLa-S3 MAX
# train HeLa-S3 HepG2 MAX
# train HepG2 K562 MAX
# train K562 A549 MAX

train GM12878 H1-hESC TAF1
train H1-hESC HeLa-S3 TAF1
train HeLa-S3 K562 TAF1
train K562 GM12878 TAF1