for i in 0 1 5 9
do
    target_week=$((42 + $i))
    python src/main.py -d 0 --seed 2345 -m meta -di bogota -date "${i}_moving" -note "${i}_moving_forecasting[lap]" -week $target_week -config_file Configs/bogota_lap.json
done