for i in 0
do
target_week=$((49 + $i))
python src/prepare_dataset.py --moving_window $i --week $target_week --case_only
python src/prcess_data_dung.py --moving_window $i --eps 1
python src/forecasting_pets.py --date "${i}_moving" --data "online/train_${i}_moving_lstm.csv"
done