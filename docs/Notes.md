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
- Grid-Search
  - Ray-tune?
  - Welche Parameter?
- Andere Architekturen?
  - Was bietet mmpretrain?
    - Convnext (v2)
    - Efficientnet
    - DenseNet
  - Einfluss von einzelnen Bildern auf das Training/Modell
