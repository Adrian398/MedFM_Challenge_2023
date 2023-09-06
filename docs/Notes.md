## Experiment Notes

### Was können wir noch machen

- Besten Datensatz finden
  - bestes Supportset
  - Qualität der Bilder
  - Labelverteilung
    - Klassen maximieren (möglichst viele Pathologien)
    - Klassen minimieren (nur eine Pathologie pro Bild, wenn möglich)
  - Validation Set prüfen
    - die vorgegebenen Validation Splits haben nicht alle Bilder
    - Prüfen, ob es da Begrenzungen gibt
  - Leave one out testen
- Ensemble infer --> prep-infer beste k Modelle finden
  - Restriktion: 10GB RAM --> mem Tensorboard
  - ensemble als Gewichtsaveraging
    - Nach jeder Epoche Modell speichern
  - Training: Jede Epoche speichern --> k beste Epochen nehmen
- Maximieren auf AUC bzw. Aggregate testen
- AUC-Klassenperformance überprüfen
  - besonders bei Chest
  - micro/macro?
  - macro
- Grid-Search
  - Ray-tune?
  - Welche Parameter?
- Andere Architekturen?
  - Was bietet mmpretrain?
    - Convnext (v2)
    - Efficientnet
    - DenseNet
  - Einfluss von einzelnen Bildern auf das Training/Modell





Wichtiges Metric ist Ergebnis auf dem Hold out Validation set und Test SET!!!
Beste Modelle:
  - Swinv2
  - Swinvpt
  - Clip

Model:
  - Model soup oder ensembel bauen
  - VRAM swinv2 nur inference (Adrian) 
  - swinv2 tiny vram (Marcel)
  - grid search (Marcel) 
  - Endo swinv2 mit 10 shot (Marcel)
  - Endo swinv2 vs swinvpt 1shot (Marcel)
  - Chest Swinv2 1-shot, 5-shot, 10-shot (Marcel)
  - Wird Dataugmentation richtig durchgeführt? (Amar)

Datensatz:
  - Endo einfach die Patienten mit den meisten Bildern! 
  - (leave one out vllt nocht implementieren(Adrian))
  - alle Patienten durchgehen um die besten zu finden! Vor allem wichtig für Chest und Colon (Micha)

Hyperparameter search:
 - hauptsächlich swinv2
 - Learning rate optimieren (1e-6)

Metric pro Klasse optimieren: 
 - compute AUC pro klassen (micha) Endo, chest
 - mAP mirco oder macro? (micha)
 - jede klasse einzeln anschauen 
 - Bei Patienten mit wenig bei colon testen







Final results:

Allgemein
- Performance auf testset machen
- Ensemble prüfen (validation machen)
- convnext testen (test set performance ansehen)
- dataaugmentation resnet (Amar nochmal ansehen)

Endo:
- mit den preprocessing testen

Chest:
- Efficient net
- 