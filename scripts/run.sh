for i in 0
do
target_week=$((49 + $i))
# python src/prepare_dataset.py --moving_window $i --week $target_week
# python src/prepare_dataset.py --moving_window $i --week $target_week --test
# python src/prcess_data_dung.py --moving_window $i --eps 1
python src/main.py -st MA -j -d 0 --seed 2345 -m meta -di bogota -date "${i}_moving" -note "${i}_moving" -week $target_week -config_file Configs/bogota.json
done