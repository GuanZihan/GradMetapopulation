for i in 0
do
    target_week=$((40 + $i))
    python src/main.py --task nowcasting -d 0 --seed 2345 -m meta -di COVID -date "0_moving" -note "2021_01_16_${i}_moving_nowcasting_8_revision[lap2]" -week $target_week -config_file Configs/us_nowcasting.json \
        -revision_files 2020-11-21_0_moving_202047_dp2.csv 2020-11-28_0_moving_202048_dp2.csv 2020-12-05_0_moving_202049_dp2.csv 2020-12-12_0_moving_202050_dp2.csv 2020-12-19_0_moving_202051_dp2.csv 2020-12-26_0_moving_202052_dp2.csv 2021-01-02_0_moving_202053_dp2.csv 2021-01-09_0_moving_202101_dp2.csv
done