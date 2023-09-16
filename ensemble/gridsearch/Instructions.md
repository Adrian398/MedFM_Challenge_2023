# Instructions

### **1. Create The Validation Directory** 
Use `task_submission.py` to load the data from all available valid model runs. A model run is valid if it contains 
   - a `submission.csv` file,
   - a `validation.csv` file, and 
   - a `performance.json` file.
   
   Furthermore, a root validation directory with the current **timestamp** is created. 
Within this directory a subdirectory for each task (chest, endo, and colon) is created. Diving deeper in the structure, 
a strategy subdirectory for each available strategy is created. Strategies that make use of the top-k models receive 
another subdirectory for each top-k value. The amount of top-k subdirectories created within a strategy depends on the 
**least** amount of trained model runs that are available for the dedicated task within each shot and experiment setting 
(e.g., `colon/1-shot/exp1`). Strategies that do not make use of the top-k models, such as the `expert` strategy, do 
not receive top-k subdirectories. Instead, the strategy directory is utilized in the same manner as a top-k folder.

In the end we are performing an ensemble gridsearch with every available and implemented ensemble strategy, every 
available top-k (starting from 2, since 1 corresponds to the expert strategy), within every task given a specific 
number or state of usable trained model runs.

The leaf directories (mostly top-k) then contain:
- A `config.json` file, providing information about the timestamp, the top-k setting, the available model-count for the task during the directory generation, and the strategy used.  
  Example File:
  ```json
  {
    "timestamp": "09-09_21-11-07",
    "top-k": 2,
    "model-count": 202,
    "strategy": "rank-based-weighted"
  }
  ```

- A `report.txt` file, providing a breakdown about what *exact* trained model runs have been utilized for which of the task's class, in order to plug together the full prediction CSV for the specific ensemble strategy. This information is given for every shot and every experiment within the task (3 x 5 = 15 times). Additionally, each breakdown is followed by a summary which provides insight about how often any specific trained model run has been used. Some models may have been used for multiple classes.  
  ```txt
  ...
  Setting: colon/10-shot/exp5
  Class 1: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-074118 (Weight: 0.7077)
  Class 1: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-061615 (Weight: 0.5736)
  Class 1: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-063616 (Weight: 0.5379)
  Class 1: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-072123 (Weight: 0.4027)
  Class 1: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-071558 (Weight: 0.0000)
  Class 2: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-074118 (Weight: 0.7077)
  Class 2: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-061615 (Weight: 0.5736)
  Class 2: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-063616 (Weight: 0.5379)
  Class 2: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-072123 (Weight: 0.4027)
  Class 2: colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-071558 (Weight: 0.0000)
  
  Model Summary:
  colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-074118 used 2 times
  colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-061615 used 2 times
  colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-063616 used 2 times
  colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-072123 used 2 times
  colon/10-shot/resnet101_bs32_lr1e-06e-06_exp5__seed_test_20230831-071558 used 2 times
  ```
- A `result` directory, assembled by leveraging the specific ensemble strategy, following the **strict** structure of the template directory structure that must be submitted for validation/evaluation.



### **2. Test The Performance**
Use `test_task_submission.py` to fetch the latest timestamps from the `ensemble/gridsearch/validation` directory. 
The content within every `result` directory in the leaf directories that were previously created is then assessed and 
compared against the labels of the officially given validation dataset.  

#### 2.1 Ensemble: Create The Isolated Task Results
During this process, every leaf folder receives a `results.json` file that contains the isolated evaluated metrics for the given task **only**.

Example File:
``` json
{
  "task": {
    "exp1": {
      "colon_1-shot": {
        "ACC_metric": "88.24069820854386",
        "AUC_metric": "93.8569415228759"
      },
      "colon_5-shot": {
        "ACC_metric": "93.10978410656867",
        "AUC_metric": "98.89610350094628"
      },
      "colon_10-shot": {
        "ACC_metric": "96.2333486449242",
        "AUC_metric": "99.51699738709786"
      }
    },
    ...
    "exp5": {
      "colon_1-shot": {
        "ACC_metric": "90.6293063849334",
        "AUC_metric": "96.3955563759613"
      },
      "colon_5-shot": {
        "ACC_metric": "92.7652733118971",
        "AUC_metric": "98.15307283647849"
      },
      "colon_10-shot": {
        "ACC_metric": "94.39595774000918",
        "AUC_metric": "98.88778675040214"
      }
    }
  },
  "aggregates": "94.06116003444221"
}
```

Continuing, every task directory receives a `log.txt` file, which is a 
listing of all strategy results for the specific isolated task.

Example File:
```txt
Model-Count     Strategy             Top-K      PredictionDir                            Aggregate 
360             expert               None       expert                                   93.2380   
360             pd-weighted          2          pd-weighted/top-2                        93.2380   
360             pd-log-weighted      2          pd-log-weighted/top-2                    93.2380  
...             ...                  ...        ...                                      ...
360             pd-log-weighted      6          pd-log-weighted/top-6                    94.0336   
360             pd-weighted          6          pd-weighted/top-6                        94.0379   
360             rank-based-weighted  5          rank-based-weighted/top-5                94.0612 
```

#### 2.2 Ensemble: Create The Compound Task Results
Now that we have (hopefully) determined the best ensemble strategy per task, this information is automatically being extracted 
to the `best_ensemble_per_task.json` file, which is located within the timestamp directory (root directory of the run). The model count is a
relevant information in the way as that we can tell why two exact same runs might have different aggregate results. It increases the traceability.
If more or less models are incorporated, the selected models, and thus the results may change.

Example File:
```json
{
    "colon": {
        "Model-Count": "360",
        "Strategy": "rank-based-weighted",
        "Top-K": "5",
        "Aggregate": "94.0612"
    },
    "endo": {
        "Model-Count": "255",
        "Strategy": "expert",
        "Top-K": "None",
        "Aggregate": "49.3993"
    },
    "chest": {
        "Model-Count": "202",
        "Strategy": "rank-based-weighted",
        "Top-K": "7",
        "Aggregate": "42.7893"
    }
}
```

With the knowledge of the best performing ensemble strategies for every task, we can then create the compound `results.json` file
within the root directory by fusing the metrics from each individual task `results.json` together. The **aggregates** 
value can then be calculated in the same manner as it is officially done when submitting an evaluation submission. Thus, our resulting
aggregates value can be compared (with care) to the results on the leaderboard (with some bias).

Example File:
``` json
{
    "task": {
        "exp1": {
            "colon_1-shot": {
                "ACC_metric": "88.24069820854386",
                "AUC_metric": "93.8569415228759"
            },
        ...
        "exp5": {
            "colon_1-shot": {
                "ACC_metric": "90.6293063849334",
                "AUC_metric": "96.3955563759613"
            },
            "colon_5-shot": {
                "ACC_metric": "92.7652733118971",
                "AUC_metric": "98.15307283647849"
            },
            "colon_10-shot": {
                "ACC_metric": "94.39595774000918",
                "AUC_metric": "98.88778675040214"
            },
            "endo_1-shot": {
                "AUC_metric": "59.397904097980444",
                "mAP_metric": "19.23602543487621"
            },
            "endo_5-shot": {
                "AUC_metric": "70.47320126016038",
                "mAP_metric": "31.471859307586612"
            },
            "endo_10-shot": {
                "AUC_metric": "77.85878654678444",
                "mAP_metric": "45.42870289188368"
            },
            "chest_1-shot": {
                "AUC_metric": "62.19089057048929",
                "mAP_metric": "14.172025038685426"
            },
            "chest_5-shot": {
                "AUC_metric": "70.14264106717697",
                "mAP_metric": "19.440925517763564"
            },
            "chest_10-shot": {
                "AUC_metric": "72.14206364755375",
                "mAP_metric": "22.02639507919861"
            }
        }
    },
    "aggregates": 62.08324583896831
}
```