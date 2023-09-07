import re

# Constant Lists
shots = ['1-shot', '5-shot', '10-shot']
tasks = ["colon", "endo", "chest"]
exps = ["exp1", "exp2", "exp3", "exp4", "exp5"]

# Patterns
EXP_PATTERN = re.compile(r'exp(\d+)')

# Mappings
TASK_2_CLASS_COUNT = {
    'colon': 2,  # Binary classification
    'chest': 19,  # 19-class multi-label classification
    'endo': 4  # 4-class multi-label classification
}
TASK_2_CLASS_NAMES = {
    'colon': ['tumor'],
    'chest': ['pleural_effusion', 'nodule', 'pneumonia', 'cardiomegaly', 'hilar_enlargement', 'fracture_old',
              'fibrosis',
              'aortic_calcification', 'tortuous_aorta', 'thickened_pleura', 'TB', 'pneumothorax', 'emphysema',
              'atelectasis', 'calcification', 'pulmonary_edema', 'increased_lung_markings', 'elevated_diaphragm',
              'consolidation'],
    'endo': ['ulcer', 'erosion', 'polyp', 'tumor']
}
METRIC_TAG_2_FULLNAME = {
    "AUC": "AUC/AUC_multiclass",
    "AUC_L": "AUC/AUC_multilabe",
    "mAP": "multi-label/mAP",
    "AGG": "Aggregate",
    "ACC": "accuracy/top1"
}