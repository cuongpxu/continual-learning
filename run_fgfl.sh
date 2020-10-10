#!/bin/bash
python main.py --scenario task --loss fgfl --fgfl_gamma 0.25 --fgfl_delta 0.5 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 0.25 --fgfl_delta 1.0 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 0.25 --fgfl_delta 1.5 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 0.25 --fgfl_delta 2.0 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 0.25 --fgfl_delta 3.0 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 0.5 --fgfl_delta 0.25 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 1.0 --fgfl_delta 0.25 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 1.5 --fgfl_delta 0.25 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 2.0 --fgfl_delta 0.25 --metrics --pdf --results-dir ./results_fgfl
python main.py --scenario task --loss fgfl --fgfl_gamma 3.0 --fgfl_delta 0.25 --metrics --pdf --results-dir ./results_fgfl
