# ======================================================
# STAT801 Video Assignment – Support Vector Machines
# Dataset: creditworthiness.csv
# ======================================================

# ------------------------------------------------------
# Preamble
# ------------------------------------------------------
set.seed(801)
options(digits = 4)

# Load packages
library(e1071)   # SVM, Naive Bayes
library(MASS)    # LDA, QDA
library(class)   # KNN
library(caret)   # confusionMatrix, accuracy
library(tidyverse)
library(ggplot2)
library(gridExtra)

# ------------------------------------------------------
# Load and inspect data
# ------------------------------------------------------
W <- read.csv("creditworthiness.csv")
str(W)
summary(W)
colSums(is.na(W))

# ------------------------------------------------------
# Prepare response variable
# ------------------------------------------------------
names(W)[which(names(W) == "credit.rating")] <- "Creditworthy"
W$Creditworthy <- as.factor(W$Creditworthy)

# Remove unlabeled cases (0 = unclassified)
W <- W[W$Creditworthy != 0, ]
W$Creditworthy <- droplevels(W$Creditworthy)
cat("\n--- Labelled Data Summary ---\n")
print(table(W$Creditworthy))
# 1 2 3 -> 1962 obs

# ------------------------------------------------------
# Split into train/test (70/30)
# ------------------------------------------------------
set.seed(801)
train_id <- sample(1:nrow(W), 0.7 * nrow(W))
train <- W[train_id, ]
test  <- W[-train_id, ]

# ------------------------------------------------------
# 1. LDA
# ------------------------------------------------------
lda.fit <- lda(Creditworthy ~ ., data = train)
lda.pred <- predict(lda.fit, test)
cat("\n--- LDA Confusion Matrix ---\n")
print(confusionMatrix(lda.pred$class, test$Creditworthy))

# ------------------------------------------------------
# 2. QDA (with PCA preprocessing — ISLR recommendation)
# ------------------------------------------------------
train.X <- scale(train[, -which(names(train) == "Creditworthy")])
test.X  <- scale(test[, -which(names(test) == "Creditworthy")])

pca.fit <- prcomp(train.X, center = TRUE, scale. = TRUE)
summary(pca.fit)

# Keep first 10 PCs
train.pca <- data.frame(pca.fit$x[, 1:10],
                        Creditworthy = train$Creditworthy)
test.pca <- data.frame(predict(pca.fit, newdata = test.X)[, 1:10],
                       Creditworthy = test$Creditworthy)

qda.fit <- qda(Creditworthy ~ ., data = train.pca)
qda.pred <- predict(qda.fit, test.pca)
cat("\n--- QDA (with PCA) Confusion Matrix ---\n")
print(confusionMatrix(qda.pred$class, test.pca$Creditworthy))

# ------------------------------------------------------
# 3. Naive Bayes
# ------------------------------------------------------
nb.fit <- naiveBayes(Creditworthy ~ ., data = train)
nb.pred <- predict(nb.fit, test)
cat("\n--- Naive Bayes Confusion Matrix ---\n")
print(confusionMatrix(nb.pred, test$Creditworthy))

# ------------------------------------------------------
# 4. KNN (standardised)
# ------------------------------------------------------
train.X <- scale(train[, -which(names(train) == "Creditworthy")])
test.X  <- scale(test[, -which(names(test) == "Creditworthy")])
train.Y <- train$Creditworthy
test.Y  <- test$Creditworthy

knn.pred <- knn(train.X, test.X, train.Y, k = 5)
cat("\n--- KNN Confusion Matrix (k=5) ---\n")
print(confusionMatrix(knn.pred, test.Y))

# ------------------------------------------------------
# 5. SVM (Linear and RBF kernels)
# ------------------------------------------------------

# Linear kernel
svm.linear <- svm(Creditworthy ~ ., data = train,
                  kernel = "linear", cost = 1, scale = TRUE)
svm.linear.pred <- predict(svm.linear, test)
cat("\n--- SVM (Linear) Confusion Matrix ---\n")
print(confusionMatrix(svm.linear.pred, test$Creditworthy))

# RBF kernel (default parameters)
svm.rbf <- svm(Creditworthy ~ ., data = train,
               kernel = "radial", cost = 1, gamma = 0.1)
svm.rbf.pred <- predict(svm.rbf, test)
cat("\n--- SVM (RBF, cost=1, gamma=0.1) Confusion Matrix ---\n")
print(confusionMatrix(svm.rbf.pred, test$Creditworthy))

# ------------------------------------------------------
# 6. SVM parameter tuning (RBF kernel)
# ------------------------------------------------------
tune.out <- tune(svm, Creditworthy ~ ., data = train,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100),
                               gamma = c(0.01, 0.1, 1)))
summary(tune.out)

best.model <- tune.out$best.model
best.pred <- predict(best.model, test)
cat("\n--- Best SVM (RBF tuned) Confusion Matrix ---\n")
print(confusionMatrix(best.pred, test$Creditworthy))

# ------------------------------------------------------
# 7. Accuracy summary
# ------------------------------------------------------
results <- data.frame(
  Model = c("LDA", "QDA (PCA)", "Naive Bayes", "KNN (k=5)", "SVM (Best RBF)"),
  Accuracy = c(
    mean(lda.pred$class == test$Creditworthy),
    mean(qda.pred$class == test.pca$Creditworthy),
    mean(nb.pred == test$Creditworthy),
    mean(knn.pred == test.Y),
    mean(best.pred == test$Creditworthy)
  )
)
cat("\n--- Model Accuracy Summary ---\n")
print(results)

# ------------------------------------------------------
# 8. Visualisation of SVM tuning results
# ------------------------------------------------------
perf <- tune.out$performances
perf$accuracy <- 1 - perf$error

ggplot(perf, aes(x = factor(gamma), y = factor(cost), fill = accuracy)) +
  geom_tile(color = "white") +
  scale_fill_viridis_c(option = "C") +
  labs(
    title = "SVM Tuning Results (RBF Kernel)",
    subtitle = "Cross-validated Accuracy by Cost and Gamma",
    x = "Gamma",
    y = "Cost"
  ) +
  theme_minimal(base_size = 13)

# ------------------------------------------------------
# 9. Optional: Decision Boundary (2D visual demo)
# ------------------------------------------------------
if (ncol(W) > 3) {
  df_plot <- train[, c(1, 2, ncol(W))]
  colnames(df_plot) <- c("X1", "X2", "Creditworthy")
  svm.demo <- svm(Creditworthy ~ ., data = df_plot,
                  kernel = "radial", cost = 1, gamma = 0.1)
  grid <- expand.grid(
    X1 = seq(min(df_plot$X1), max(df_plot$X1), length = 100),
    X2 = seq(min(df_plot$X2), max(df_plot$X2), length = 100)
  )
  grid$pred <- predict(svm.demo, grid)
  ggplot(df_plot, aes(X1, X2, color = Creditworthy)) +
    geom_point(size = 1.5) +
    geom_contour(data = grid, aes(z = as.numeric(pred)),
                 color = "black", bins = 3) +
    labs(title = "SVM Decision Boundary (2D Demo)")
}

# ------------------------------------------------------
# 10. Predict the remaining 538 unclassified cases
# ------------------------------------------------------

# Reload full dataset to access unlabeled observations
W_full <- read.csv("creditworthiness.csv")
names(W_full)[which(names(W_full) == "credit.rating")] <- "Creditworthy"
W_full$Creditworthy <- as.factor(W_full$Creditworthy)

# Extract unclassified cases (Creditworthy == 0)
unclassified <- W_full[W_full$Creditworthy == 0, ]

# Predict their credit ratings using the best-performing SVM model
pred_unclassified <- predict(best.model, newdata = unclassified)

# Display summary of predicted credit ratings
cat("\n--- Predicted Credit Ratings for 538 Unclassified Cases ---\n")
print(table(pred_unclassified))

# Optionally, attach predictions to the original data
unclassified$PredictedRating <- pred_unclassified

# Save results for reference
write.csv(unclassified, "Predicted_CreditRatings_538.csv", row.names = FALSE)
cat("\nPredictions saved to: Predicted_CreditRatings_538.csv\n")

