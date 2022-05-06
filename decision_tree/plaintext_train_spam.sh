set -eux
depth=$1
features=$2
python spam_detection_train.py $depth $features
dot -Tpdf dt.dot -o dt_spam_${features}_${depth}.pdf