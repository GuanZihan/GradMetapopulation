for i in 0
do
train_week=$((49 + $i))
test_week=$((49 + $i+4))
python prepare_dataset.py --moving_window $i --week $train_week
python prepare_dataset.py --moving_window $i --week $test_week --test
python prcess_data_dung.py --moving_window $i --eps 1
python main.py -st MA -j -d 0 --seed 2345 -m meta -di bogota -date "${i}_moving" -note "seed_2345"
done



