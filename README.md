# Predicting Hits with KNN: 2026 Balls in Play

A k nearest neighbors (KNN) model built in R that predicts whether a ball in play is a hit or an out, using every 2026 MLB regular season ball in play with full bat tracking data (Statcast via Baseball Savant).

This is a follow up to my [SVM project](https://github.com/Cesarioj1998/LA_EV-2026-SVM-Model), which used only launch angle and exit velocity. Here the question was whether adding more predictors would improve the model. The nine predictors are launch speed, launch angle, hit distance, ballpark, bat speed, swing length, attack angle, attack direction, and swing path tilt.

## Results

With k = 191, the model correctly classified 79.9% of 29,163 balls in play it never saw during training, compared to a 67.7% baseline from predicting an out every time. Cross validated accuracy on the training set was 79.87%, so the model holds up on new data and isn't overfitting.

The interesting part: the SVM reached 80.6% with just two predictors. The extra seven didn't add much, likely because KNN gives every predictor an equal say in measuring distance, so weaker predictors add noise.

| | Actual out | Actual hit |
| --- | --- | --- |
| Predicted out | 18,517 | 4,582 |
| Predicted hit | 1,278 | 4,786 |

The model caught 93.5% of outs but only 51.1% of hits. When it predicted a hit, it was right 78.9% of the time.

![Actual outcomes, test set](KNN%20Outcomes/Actual%20Outcomes%20%282026%29%2C%20KNN%20Test%20Set.png)
![Predicted outcomes, test set](KNN%20Outcomes/KNN%20Predicted%20Outcomes%20%282026%29%2C%20Test%20Set%20%28k%3D191%29.png)

## Where it gets it wrong

The errors aren't random. Missed hits (blue) are mostly ground balls that found a hole, soft contact, and bloops. False hits (orange) are mostly lineouts and deep fly balls caught near the wall. These outcomes depend on things none of the nine predictors capture, like spray direction, fielder positioning, and sprint speed.

![Prediction errors, test set](KNN%20Outcomes/KNN%20Prediction%20Errors%20%282026%29%2C%20Test%20Set%20%28k%3D191%29.png)

## Choosing k

The best k was found with leave one out cross validation on a 75% training set (87,488 balls in play). Most of the gain came between k = 1 and k = 11, and accuracy was essentially flat from about k = 150 on, so k = 191 sits in the middle of a plateau rather than at a sharp peak.

![Accuracy vs k](KNN%20Outcomes/KNN%20Accuracy%20vs%20k%20%282026%29%2C%20Training%20Set.png)

## Next steps

- Rank the nine predictors with a random forest, then test whether dropping the weakest ones improves KNN
- Add spray angle, calculated from the hit coordinates and adjusted for batter handedness
- Lower the voting cutoff to trade some false hits for fewer missed hits

## Files

- `KNN LA + EV Project.R`: full R script (kknn package)
- `knn_k_search_results.csv`: cross validated accuracy for each k tested, so the slow search doesn't need to be rerun
- `KNN Outcomes/`: all four plots
