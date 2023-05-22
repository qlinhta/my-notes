library(ggplot2)
library(gridExtra)
library(grid)
library(GGally)

df <- read.csv("dataset/breast-cancer-wisconsin.data", header = T, stringsAsFactors = F)

summary(df)

df$X <- NULL

df <- df[, -1]
df$diagnosis <- factor(ifelse(df$diagnosis == "B", "Benign", "Malignant"))

str(df)

# Correlation
library(PerformanceAnalytics)
chart.Correlation(df[, c(2:11)], histogram = TRUE, col = "grey10", pch = 1, main = "Mean")
chart.Correlation(df[, c(12:21)], histogram = TRUE, col = "grey10", pch = 1, main = "SE")
chart.Correlation(df[, c(22:31)], histogram = TRUE, col = "grey10", pch = 1, main = "Worst")

ggpairs(df[, c(2:11, 1)], aes(color = diagnosis, alpha = 0.75), lower = list(continuous = "smooth")) +
  theme_bw() +
  labs(title = "Mean") +
  theme(plot.title = element_text(face = 'bold', color = 'black', hjust = 0.5, size = 10))
ggpairs(df[, c(12:21, 1)], aes(color = diagnosis, alpha = 0.75), lower = list(continuous = "smooth")) +
  theme_bw() +
  labs(title = "SE") +
  theme(plot.title = element_text(face = 'bold', color = 'black', hjust = 0.5, size = 10))
ggpairs(df[, c(22:31, 1)], aes(color = diagnosis, alpha = 0.75), lower = list(continuous = "smooth")) +
  theme_bw() +
  labs(title = "Worst") +
  theme(plot.title = element_text(face = 'bold', color = 'black', hjust = 0.5, size = 10))

# Embedding PCA
library(factoextra)
df_pca <- transform(df)

pca <- prcomp(df_pca[, -1], cor = TRUE, scale = TRUE)
summary(pca)

fviz_eig(pca, addlabels = TRUE, ylim = c(0, 60), geom = c("bar"), barfill = "grey", barcolor = "grey", linecolor = "red", ncp = 10) +
  labs(title = "PCA variances", x = "Principal Components", y = "% of variances")

# Contribution PC1, 2, 3
library(gridExtra)
p1 <- fviz_contrib(pca, choice = "var", axes = 1, fill = "grey", color = "grey", top = 10)
p2 <- fviz_contrib(pca, choice = "var", axes = 2, fill = "grey", color = "grey", top = 10)
p3 <- fviz_contrib(pca, choice = "var", axes = 3, fill = "grey", color = "grey", top = 10)
grid.arrange(p1, p2, p3, ncol = 3)

# Biplot
library("factoextra")
fviz_pca_biplot(pca, col.ind = df$diagnosis, col = "black",
                palette = c("blue", "red"), geom = "point", repel = TRUE,
                legend.title = "Diagnosis", addEllipses = TRUE)

boxplot2g = function(x, y = NULL, groups = NULL, smooth = loess, smooth.args = list(span = 0.1), colv = NULL, alpha = 1, n = 360, ...) {
  prbs <- c(0.25, 0.5, 0.75)
  if (is.null(y)) {
    stopifnot(ncol(x) == 2)
    data <- as.data.frame(x)
  }else {
    data <- as.data.frame(cbind(x, y))
  }
  if (is.null(groups)) {
    data$groups <- as.factor(0)
  }else {
    data$groups <- as.factor(groups)
  }
  labs <- names(data)
  names(data) <- c("x", "y", "groups")
  DM <- data.matrix(data)


  if (is.logical(smooth)) {
    do.smooth <- smooth
  }else {
    do.smooth <- TRUE
  }
  if (do.smooth) {
    poss.args <- names(formals(smooth))
    spec.args <- names(smooth.args)

    ind <- match(spec.args, poss.args)
    for (i in seq_along(ind)) {
      formals(smooth)[ind[i]] <- smooth.args[[i]]
    }
    if ("span" %in% poss.args) {
      formals(smooth)$span <- formals(smooth)$span / 3
    }
  }else {
    smooth <- NULL
  }
  phi = seq(360 / n, 360, 360 / n) / 180 * pi
  e1 <- new.env()
  e1$vectors <- cbind(sin(phi), cos(phi))
  ntv <- nlevels(data$groups)
  if (is.null(colv)) {
    #print(ntv)
    if (ntv == 1) {
      colv = 1
    }else {
      colv <- rainbow(ntv)
    }
  }
  e1$colv <- colv
  e1$lvls <- levels(data$groups)

  e1$gp <- ggplot(data = data, aes(x = x, y = y, colour = groups)) + geom_point(alpha = alpha)
  if (ntv == 1) {
    groupbox2d(x = data, env = e1, prbs = prbs, smooth = smooth, do.smooth)
  }else {
    by(data, groups, groupbox2d, env = e1, prbs = prbs, smooth = smooth)
  }
  return(e1$gp)
}

groupbox2d = function(x, env, prbs, past, smooth) {
  grp <- x[1, 3]
  colid <- match(grp, env$lvls)
  if (any(colid)) {
    colv <- env$colv[]
  }else {
    colv <- env$col[1]
  }
  xs <- x[, 1:2]
  mm <- apply(xs, 2, mean)
  xs <- data.matrix(xs) - rep(mm, each = nrow(xs))
  S <- cov(xs)
  if (requireNamespace("MASS", quietly = TRUE)) {
    Sinv <- MASS::ginv(S)
    SSinv <- svd(Sinv)
    SSinv <- SSinv$u %*% diag(sqrt(SSinv$d))
    SS <- MASS::ginv(SSinv)
  }else {
    Sinv <- solve(S)
    SSinv <- svd(Sinv)
    SSinv <- SSinv$u %*% diag(sqrt(SSinv$d))
    SS <- solve(SSinv)
  }
  xs <- xs %*% SSinv
  prj <- xs %*% t(env$vectors)
  qut <- t(apply(prj, 2, function(z) {
    quarts <- quantile(z, probs = prbs)
    iqr <- quarts[3] - quarts[1]
    w1 <- min(z[which(z >= quarts[1] - 1.5 * iqr)])
    return(c(w1, quarts))
  }))
  if (!is.null(smooth)) {
    n <- nrow(qut)
    qut <- apply(qut, 2, function(z) {
      x <- 1:(3 * n)
      z <- rep(z, 3)
      ys <- predict(smooth(z ~ x))
      return(ys[(n + 1):(2 * n)])
    })
  }
  ccBox <- env$vectors * qut[, 2]
  md <- data.frame((env$vectors * qut[, 3]) %*% SS)
  md <- sapply(md, mean) + mm
  md[3] <- grp
  ccWsk <- env$vectors * qut[, 1]
  ccc <- data.frame(rbind(ccBox, ccWsk) %*% SS + rep(mm, each = 2 * nrow(ccBox)))
  ccc$grp <- as.factor(rep(c("box", "wsk"), each = nrow(ccBox)))
  ccc$groups <- factor(grp)
  md <- data.frame(md[1], md[2], grp)
  names(md) <- names(ccc)[-3]
  X1 <- NULL
  X2 <- NULL
  groups <- NULL
  env$gp <- env$gp +
    geom_point(data = md, aes(x = X1, y = X2, colour = groups), size = 5) +
    scale_colour_manual(values = colv)
  env$gp <- env$gp + geom_path(data = ccc, aes(x = X1, y = X2, group = grp, colour = groups), alpha = 1 / 8)
  env$gp <- env$gp + geom_polygon(data = ccc, aes(x = X1, y = X2, group = grp, colour = groups, fill = groups), alpha = 1 / 8)
  env$gp <- env$gp + geom_point(data = md, aes(x = X1, y = X2), size = 3, alpha = 1, colour = "white")
  env$gp <- env$gp + geom_point(data = md, aes(x = X1, y = X2), size = 1, alpha = 1)
  return(invisible(TRUE))
}

b1 <- boxplot2g(df$radius_worst, df$perimeter_mean, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Perimeter mean vs. Radius worst", x = "Radius worst", y = "Perimeter mean") +
  theme_bw()
b2 <- boxplot2g(df$area_worst, df$radius_worst, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Area worst vs. Radius worst", x = "Radius worst", y = "Area worst") +
  theme_bw()
b3 <- boxplot2g(df$texture_mean, df$texture_worst, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Texture mean vs. Texture worst", x = "Texture worst", y = "Texture mean") +
  theme_bw()
b4 <- boxplot2g(df$area_worst, df$perimeter_mean, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Perimeter mean vs. Area worst", x = "Area worst", y = "Perimeter mean") +
  theme_bw()
grid.arrange(b1, b2, b3, b4, ncol = 2)

b9 <- boxplot2g(df$fractal_dimension_worst, df$area_se, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Low correlated features", subtitle = "Area SE vs. Fractal dimmension worst", x = "Fractal dimmension worst", y = "Area SE") +
  theme_bw()
b10 <- boxplot2g(df$fractal_dimension_worst, df$radius_se, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Low correlated features", subtitle = "Radius SE vs. Fractal dimmension worst", x = "Fractal dimmension worst", y = "Radius SE") +
  theme_bw()
b11 <- boxplot2g(df$texture_mean, df$smoothness_mean, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Low correlated features", subtitle = "Smoothness mean vs. Texture mean", x = "Texture mean", y = "Smoothness mean") +
  theme_bw()
b12 <- boxplot2g(df$perimeter_worst, df$fractal_dimension_se, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Low correlated features", subtitle = "Fractal dimmension SE vs. Perimeter worst", x = "Perimeter worst", y = "Fractal dimension SE") +
  theme_bw()
grid.arrange(b9, b10, b11, b12, ncol = 2)

b13 <- boxplot2g(df$radius_worst, df$area_worst, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Area worst vs. Radius worst", x = "Radius worst", y = "Area worst") +
  theme_bw()
b14 <- boxplot2g(df$radius_worst, df$perimeter_worst, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Perimeter worst vs. Radius worst", x = "Radius worst", y = "Perimeter worst") +
  theme_bw()
b15 <- boxplot2g(df$radius_worst, df$perimeter_mean, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Perimeter mean vs. Radius worst", x = "Radius worst", y = "Perimeter mean") +
  theme_bw()
b16 <- boxplot2g(df$radius_worst, df$area_mean, df$diagnosis, smooth = loess, NULL, NULL) +
  labs(title = "Highly correlated features", subtitle = "Area mean vs. Radius worst", x = "Radius worst", y = "Area mean") +
  theme_bw()
grid.arrange(b13, b14, b15, b16, ncol = 2)