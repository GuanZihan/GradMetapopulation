from epiweeks import Week

import argparse
import json
from utils import save_params
        


if __name__ == "__main__":
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='GradMeta for COVID-19.')
    parser.add_argument('-m','--model_name', help='Model name.', default = 'GradABM')
    parser.add_argument('-di','--disease', help='Disease: COVID or Flu.', default = 'COVID')
    parser.add_argument('-s', '--seed', type=int, help='Seed for python random, numpy and torch', default = 6666)
    parser.add_argument('-n', '--num_runs', type=int, help='Number of runs', default = 1)
    # parser.add_argument('-st','--state', help='State to predict', default = 'MA')
    # parser.add_argument('-c','--county_id', help='County to predict, only when not using joint training', default = '25001')
    parser.add_argument('-d','--dev', nargs='+',type=str, default='0',help='Device number to use. Put list for multiple.')    
    parser.add_argument('-ew','--pred_ew',type=str, default='202116',help='Prediction week in CDC format')
    # parser.add_argument('-j','--joint', action='store_true',help='Train all counties jointly')
    parser.add_argument('-i','--inference_only', action='store_true',help='Will not train if True, inference only')
    # parser.add_argument('-no','--noise', type=int, help='Noise level for robustness experiments', default = 0)
    # parser.add_argument('-f', '--results_file_postfix', help='Postfix to be appended to output dir for ease of interpretation', default = '')
    # parser.add_argument('-da', '-dataset', default="./Data/Processed/county_data.csv")
    parser.add_argument('-date', default="03-03")
    parser.add_argument('-note', type=str, default="")
    # parser.add_argument('--privacy', action="store_true")
    # parser.add_argument('--Delta', type=int, default=1)
    parser.add_argument('-x', type=int, default=5)
    # parser.add_argument('-counterfactual', action='store_true')
    parser.add_argument('-week', default=0, type=int)
    parser.add_argument('-config_file', required=True, help="Path to the config file.")
    parser.add_argument('-revision_files', metavar='S', type=str, nargs='+', help='a list of strings')
    parser.add_argument('--task', default='forecasting', choices=['forecasting', 'nowcasting'], type=str)
    parser.add_argument('--mask_column', default='', type=str)
    parser.set_defaults(inference_only=False)  # make true when removing no joint
    args = parser.parse_args()

    if args.disease == 'bogota' and args.task == 'forecasting':
        from forecasting_gradmeta_bogota_single import train_predict
    elif args.disease == 'COVID' and args.task == 'forecasting':
        from forecasting_gradmeta_usa import train_predict
    elif args.disease == 'bogota' and args.task == 'nowcasting':
        from nowcasting_gradmeta_bogota import train_predict
    elif args.disease == 'COVID' and args.task == 'nowcasting':
        from nowcasting_gradmeta_us import train_predict

    disease = args.disease
    model_name = args.model_name
    pred_ew = Week.fromstring(args.pred_ew)
    args.pred_week = pred_ew.cdcformat()

    with open(args.config_file, "r") as f:
        configs = json.load(f)

    counties_predicted, predictions, learned_params = train_predict(args, configs)