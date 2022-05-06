set -eux
appendix=('breast' 'parkinsons' 'car', 'spambase', 'wine')
index=$1
python plaintext_train.py $index
dot -Tpdf dt.dot -o dt_${appendix[index]}.pdf