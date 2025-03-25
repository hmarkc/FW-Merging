# set -e pipefail

outdir=${outdir:="outs/llama_merged"}
mkdir -p ${outdir}

models_to_merge=(
)

function run_avg_merge(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--base-model "meta-llama/Llama-2-7b-hf" \
--yaml-file config/average_merge.yml \
--outdir $outdir \
--lora 'qwen_lora.json'

}

function run_dare_task_arith(){

pos

for i in 0.7 ; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--yaml-file config/dare_merge.yml \
--mask-rate $i \
--outdir outs/llama_merged/dare_task \
--lora 'llama_lora.json'

done

}

function run_dare_tie(){

pos

for i in 0.7 ; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--yaml-file config/dare_merge2.yml \
--mask-rate $i \
--outdir outs/llama_merged/dare_tie \
--lora 'llama_lora.json'

done

}

function run_task_arith(){

for j in 0.3; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--yaml-file config/task_arithmetic.yml \
--scaling $j \
--outdir outs/llama_merged/task_arith \
--lora 'llama_lora.json'

done

}

function run_tie(){

pos


for i in 0.7; do
for j in 0.3; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_to_merge[@]} \
--src-merge ${models_to_merge[@]} \
--yaml-file config/ties_merge.yml \
--mask-rate $i \
--scaling $j \
--outdir outs/llama_merged/tie \
--lora 'llama_lora.json'

done
done

}


function run_frank_wolfe(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--yaml-file config/frank_wolfe_merge2.yml  \
--exclude-param ".*classifier.*" ".*bias.*"  \
--step-size 0.1 \
--max-iters 10 \
--outdir outs/llama_merged/frank_wolfe/task \
--lora 'llama_lora.json'


}

