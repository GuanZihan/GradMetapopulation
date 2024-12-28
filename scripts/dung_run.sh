#!/bin/bash
rm prediction*.csv
rm target.csv
python src/main.py -st MA -j -d 0 --seed 1234 -m meta -di bogota -date 0_moving -i --Delta 1 -x 5 -counterfactual
python src/main.py -st MA -j -d 0 --seed 1234 -m meta -di bogota -date 0_moving -i --Delta 1 -x 10 -counterfactual
python src/main.py -st MA -j -d 0 --seed 1234 -m meta -di bogota -date 0_moving -i --Delta 2 -x 5 -counterfactual
python src/main.py -st MA -j -d 0 --seed 1234 -m meta -di bogota -date 0_moving -i --Delta 2 -x 10 -counterfactual
python src/main.py -st MA -j -d 0 --seed 1234 -m meta -di bogota -date 0_moving -i --Delta 0 -counterfactual
python src/plot.py