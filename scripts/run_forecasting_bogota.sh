for i in 0 1 5 9
do
target_week=$((42 + $i))
# python src/prepare_dataset_forecasting_bogota.py --moving_window $i --week $target_week
# python src/prepare_dataset_forecasting_bogota.py --moving_window $i --week $target_week --test
# python src/privatize_bogota_lap.py --moving_window $i --eps 1
# python src/main.py -d 0 --seed 2345 -m meta -di bogota -date "${i}_moving" -note "${i}_moving_forecasting[public-single]" -week $target_week -config_file Configs/bogota_public_single.json -i
python src/main.py -d 0 --seed 2345 -m meta -di bogota -date "${i}_moving" -note "${i}_moving_forecasting[rr-single]" -week $target_week -config_file Configs/bogota_rr_single.json -i
# python src/main.py -d 0 --seed 2345 -m meta -di bogota -date "${i}_moving" -note "${i}_moving_forecasting[public]" -week $target_week -config_file Configs/bogota_public.json -i
# python src/main.py -d 0 --seed 2345 -m meta -di bogota -date "${i}_moving" -note "${i}_moving_forecasting[rr]" -week $target_week -config_file Configs/bogota_rr.json -i
done