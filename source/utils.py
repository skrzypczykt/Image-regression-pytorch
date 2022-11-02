import torch
# metadata
MIN_YEAR = 1820.0
MAX_YEAR = 1880.0
AGE_NORMALIZATION_VALUE = MAX_YEAR - MIN_YEAR

def normalize_year(year_raw):
    return (year_raw - MIN_YEAR) / AGE_NORMALIZATION_VALUE

def denormalize_year(year_norm:torch.tensor):
    return (year_norm * AGE_NORMALIZATION_VALUE + MIN_YEAR).round()