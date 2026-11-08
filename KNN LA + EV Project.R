# Predicting Hits with KNN: 2026 Balls in Play
# Uses launch speed, launch angle, hit distance, park, bat speed, swing length,
# attack angle, attack direction & swing path tilt
# to classify each ball in play as a hit or an out

# Load the kknn package (KNN classification)
library(kknn)

# ---- Load data ----
data <- read.csv("Mar_Sept_2026 Hits v Outs.csv", header=TRUE)

# ---- Create response variable: 1 = hit, 0 = out ----
data$hit <- data$events %in% c("single", "double", "triple", "home_run")
data$hit <- ifelse(data$hit, 1, 0)

# Baseline: share of outs in the full data
print(table(data$hit))

# ---- Clean data: keep only balls in play with all nine predictors ----
# home_team stands in for ballpark (every game is in the home team's park)
vars <- c(
  "launch_speed",
  "launch_angle",
  "hit_distance_sc",
  "home_team",
  "bat_speed",
  "swing_length",
  "attack_angle",
  "attack_direction",
  "swing_path_tilt",
  "hit"
)



bip <- data[complete.cases(data[, vars]), vars]

# Treat ballpark as a category, not text or a number
bip$home_team <- as.factor(bip$home_team)

# Baseline: share of hits and outs in the cleaned data
print(table(bip$hit))
print(nrow(bip))

# --- Set k ---

#Make k a set of values to test
k <- seq(151, 251, by = 2)

# ---- Set up response for classification ----
# Make hit a factor so kknn treats this as hit vs out (classification), 
# not a number
bip$hit <- as.factor(bip$hit)

# ---- Random sample of 75% of balls in play for training ----
set.seed(42)
idx <- sample(nrow(bip), round(0.75 * nrow(bip)))
bip_train <- bip[idx,]

# ---- Test set: every ball in play not in the training set ----
bip_test <- bip[-idx,]

# ---- Leave one out cross validation on the training set at this k ----
# hit ~ . uses every other column in bip as a predictor (the nine we kept)
# rectangular kernel = every neighbor gets an equal vote (plain KNN)
# scale = TRUE puts all predictors on the same scale so no one variable dominates
model_cv <- train.kknn(hit ~ ., data = bip_train, ks = k,
                       kernel = "rectangular", scale = TRUE)

# Cross validated accuracy for this k (1 minus the misclassification rate)
print(1 - model_cv$MISCLASS)

# Best k from the leave one out search (151 to 251 on the training set)
k_best <- 191

# ---- Test the final model on the held out 25% ----
# Find each test ball's 191 nearest neighbors in the training set and
# classify it as a hit or out by majority vote
model <- kknn(hit ~ ., train = bip_train, test = bip_test, k = k_best,
              kernel = "rectangular", scale = TRUE)

# Save the predicted class (0 or 1) for each test ball
# (kknn already made these when the model was built with test = bip_test)
pred_test <- model$fitted.values

# Test accuracy: share of test balls classified correctly
print(sum(pred_test == bip_test$hit) / nrow(bip_test))

# Confusion matrix (rows = predicted, columns = actual)
print(table(pred_test, bip_test$hit))

# ---- Plots ----
#Adds a hit/out legend to the bottom left of each plot
add_legend <- function() {
  legend("bottomleft",
         legend = c("Hit", "Out"),
         col = c("red", "gray"),
         pch = 16,
         bty = "n")
}

# Actual outcomes, test set
plot(bip_test$launch_speed, bip_test$launch_angle,
     col = ifelse(bip_test$hit == 1, "red", "gray"),
     pch = 16, cex = 0.2,
     xlab = "Exit Velocity (mph)",
     ylab = "Launch Angle (degrees)",
     main = "Actual Outcomes (2026), KNN Test Set")
add_legend()

# Predicted outcomes, test set
plot(bip_test$launch_speed, bip_test$launch_angle,
     col = ifelse(pred_test == 1, "red", "gray"),
     pch = 16, cex = 0.2,
     xlab = "Exit Velocity (mph)",
     ylab = "Launch Angle (degrees)",
     main = paste("KNN Predicted Outcomes (2026), Test Set (k=", k_best, ")", sep = ""))
add_legend()

# ---- Plot: where the model gets it wrong ----
# Label each test ball by how its prediction compares to what actually happened
#   Correct    = prediction matched the actual outcome
#   Missed hit = actually a hit, but the model called it an out
#   False hit  = actually an out, but the model called it a hit
result <- ifelse(pred_test == bip_test$hit, "Correct",
                 ifelse(bip_test$hit == 1, "Missed hit", "False hit"))

# Count of each label (should line up with the confusion matrix)
print(table(result))

# Start with an empty plot using the same axes as the other charts,
# so the three groups can be layered on top one at a time
plot(bip_test$launch_speed, bip_test$launch_angle, type = "n",
     xlab = "Exit Velocity (mph)",
     ylab = "Launch Angle (degrees)",
     main = paste("KNN Prediction Errors (2026), Test Set (k=", k_best, ")", sep = ""))

# Correct predictions first, in light gray, so the errors sit on top
is_correct <- result == "Correct"
points(bip_test$launch_speed[is_correct], bip_test$launch_angle[is_correct],
       col = "gray85", pch = 16, cex = 0.2)

# Missed hits in blue (slightly bigger dots so they stand out)
is_missed <- result == "Missed hit"
points(bip_test$launch_speed[is_missed], bip_test$launch_angle[is_missed],
       col = "blue", pch = 16, cex = 0.3)

# False hits in orange
is_false <- result == "False hit"
points(bip_test$launch_speed[is_false], bip_test$launch_angle[is_false],
       col = "orange", pch = 16, cex = 0.3)

# Legend for the three groups
legend("bottomleft",
       legend = c("Correct", "Missed hit", "False hit"),
       col = c("gray85", "blue", "orange"),
       pch = 16,
       bty = "n")

# ---- Accuracy vs k plot ----

# k values for the curve: every 10 from 1 to 301 (all odd, and includes 191)
k_curve <- seq(1, 301, by = 10)

# Leave one out cross validation on the training set for each k in the curve
# (slow: same runtime as the original 1 to 301 search)
model_curve <- train.kknn(hit ~ ., data = bip_train, ks = k_curve,
                          kernel = "rectangular", scale = TRUE)

# Put each k and its accuracy into a table and save it,
# so the search doesn't need to be rerun later
k_results <- data.frame(k = k_curve,
                        accuracy = 1 - as.vector(model_curve$MISCLASS))
write.csv(k_results, "knn_k_search_results.csv", row.names = FALSE)

# On later runs, comment out the search and save lines above and use this instead
# k_results <- read.csv("knn_k_search_results.csv")

# Baseline: accuracy from calling every ball in the training set an out
baseline <- mean(bip_train$hit == 0)

# Plot accuracy for each k, with points connected by a line
plot(k_results$k, k_results$accuracy, type = "b", pch = 16, cex = 0.6,
     ylim = c(0.65, 0.81),
     xlab = "k (number of neighbors)",
     ylab = "Cross Validated Accuracy",
     main = "KNN Accuracy vs k (2026), Training Set")

# Dashed gray line at the baseline, dashed red line at the chosen k
abline(h = baseline, lty = 2, col = "gray50")
abline(v = k_best, lty = 2, col = "red")

# Legend for the two reference lines
legend("right",
       legend = c("Baseline (all outs)", paste("Chosen k =", k_best)),
       col = c("gray50", "red"),
       lty = 2,
       bty = "n")
