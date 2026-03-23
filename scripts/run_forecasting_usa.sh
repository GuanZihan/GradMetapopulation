for i in 0
do
    target_week=$((32 + $i))
    python src/main.py -d 0 --seed 2345 -m meta -di COVID -date "${i}_moving" -note "${i}_moving_forecasting_usa[lap2]" -week $target_week -config_file Configs/us_forecasting.json
done