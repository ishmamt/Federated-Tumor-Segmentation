# python main.py --strategy fedOAP 2>&1 | tee logs/fedOAP.txt
# python main.py --strategy fedDP 2>&1 | tee logs/fedDP.txt
# python main.py --strategy fedREP 2>&1 | tee logs/fedREP.txt
# python main.py --strategy fedPER 2>&1 | tee logs/fedPER.txt
# python main.py --strategy fedAVG 2>&1 | tee logs/fedAVG.txt
# python main.py --strategy fedADAGRAD 2>&1 | tee logs/fedADAGRAD.txt

for model in fedAVG fedAVGM fedADAGRAD fedPER fedREP fedDP fedOAP
# for model in 
do
  for i in 1 2 3 4 5
  # for i in 1
  do
    echo "Run $model for $i iteration"
    python main.py --strategy $model --run $i 2>&1 | tee "logs/${model}${i}.txt"
  done
done