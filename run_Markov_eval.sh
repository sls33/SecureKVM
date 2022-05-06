set -eux

party_index=$1
# dataset="csdnn"
dataset="rockyou"
model_n="3"

./evalPW_mpc --pw 915376842 -C mpc_omen/models/${dataset}_${model_n}_gram/${dataset}_createConfig -w --player $party_index