#!/bin/bash
python test_all_dataset/analyze_results.py --checkpoint_path results_fft
python test_all_dataset/analyze_results.py --checkpoint_path results_hira
python test_all_dataset/analyze_results.py --checkpoint_path results_lora
python test_all_dataset/analyze_results.py --checkpoint_path results_ia3