for i in 0
do
target_week=$((43 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"
# target_week=$((19 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((18 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((17 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((16 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((15 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((14 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((13 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((12 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((11 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((10 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((9 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((8 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-29"

# target_week=$((19 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-22"


# target_week=$((18 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-15"


# target_week=$((17 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-08"

# target_week=$((16 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-08-01"


# target_week=$((15 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-07-25"

# target_week=$((14 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-07-18"

# target_week=$((13 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-07-11"

# target_week=$((12 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-07-04"

# target_week=$((11 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-06-27"

# target_week=$((10 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-06-20"

# target_week=$((9 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-06-13"

# target_week=$((8 + $i))
# python src/prepare_dataset_nowcasting.py --moving_window $i --week $target_week --revision "2020-06-06"

# python src/prcess_data_dung.py --moving_window $i --eps 1

echo $target_week
# python src/main.py -d 0 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
#                    -note "2021_01_16_${i}_moving_nowcasting_6_revision[public]" -week $target_week -config_file Configs/bogota_nowcasting_public.json \
#                    -revision_files 2020-12-05_0_moving_2020-12-05.csv 2020-12-12_0_moving_2020-12-12.csv 2020-12-19_0_moving_2020-12-19.csv 2020-12-26_0_moving_2020-12-26.csv 2021-01-02_0_moving_2021-01-02.csv 2021-01-09_0_moving_2021-01-09.csv \
#                     -i
python src/main.py -d 0 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
                   -note "2021_01_16_${i}_moving_nowcasting_6_revision[lap]" -week $target_week -config_file Configs/bogota_nowcasting_lap.json \
                   -revision_files 2020-12-05_0_moving_2020-12-05.csv 2020-12-12_0_moving_2020-12-12.csv 2020-12-19_0_moving_2020-12-19.csv 2020-12-26_0_moving_2020-12-26.csv 2021-01-02_0_moving_2021-01-02.csv 2021-01-09_0_moving_2021-01-09.csv \
                    -i
# python src/main.py -d 0 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
#                    -note "2021_01_16_${i}_moving_nowcasting_6_revision[rr]" -week $target_week -config_file Configs/bogota_nowcasting_rr.json \
#                    -revision_files 2020-12-05_0_moving_2020-12-05.csv 2020-12-12_0_moving_2020-12-12.csv 2020-12-19_0_moving_2020-12-19.csv 2020-12-26_0_moving_2020-12-26.csv 2021-01-02_0_moving_2021-01-02.csv 2021-01-09_0_moving_2021-01-09.csv \
#                    -i



# python src/main.py -d 0 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
#                    -note "2021_01_16_${i}_moving_nowcasting_5_revision[lap]" -week $target_week -config_file Configs/bogota_nowcasting_lap.json \
#                    -revision_files 2020-12-12_0_moving_2020-12-12.csv 2020-12-19_0_moving_2020-12-19.csv 2020-12-26_0_moving_2020-12-26.csv 2021-01-02_0_moving_2021-01-02.csv 2021-01-09_0_moving_2021-01-09.csv \
#                    &
# python src/main.py -d 1 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
#                    -note "2021_01_16_${i}_moving_nowcasting_4_revision[lap]" -week $target_week -config_file Configs/bogota_nowcasting_lap.json \
#                    -revision_files 2020-12-19_0_moving_2020-12-19.csv 2020-12-26_0_moving_2020-12-26.csv 2021-01-02_0_moving_2021-01-02.csv 2021-01-09_0_moving_2021-01-09.csv \
#                    &
# python src/main.py -d 1 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
#                    -note "2021_01_16_${i}_moving_nowcasting_3_revision[lap]" -week $target_week -config_file Configs/bogota_nowcasting_lap.json \
#                    -revision_files 2020-12-26_0_moving_2020-12-26.csv 2021-01-02_0_moving_2021-01-02.csv 2021-01-09_0_moving_2021-01-09.csv \
#                    &
# python src/main.py -d 1 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
#                    -note "2021_01_16_${i}_moving_nowcasting_2_revision[lap]" -week $target_week -config_file Configs/bogota_nowcasting_lap.json \
#                    -revision_files 2021-01-02_0_moving_2021-01-02.csv 2021-01-09_0_moving_2021-01-09.csv \
#                    &
# python src/main.py -d 1 --task nowcasting --seed 2345 -m meta -di bogota -date "${i}_moving" \
#                    -note "2021_01_16_${i}_moving_nowcasting_1_revision[lap]" -week $target_week -config_file Configs/bogota_nowcasting_lap.json \
#                    -revision_files 2021-01-09_0_moving_2021-01-09.csv \
#                    &
# wait
done