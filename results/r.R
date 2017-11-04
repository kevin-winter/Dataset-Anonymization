library(readr)
x <- read_csv("~/Documents/workspace/_DATA SCIENCE/Dataset Anonymization/results/out_all.csv")
x <- x[!(x$algorithm %in% c("bnb", "gnb")),]
x$dt = x$dt_accuracy_sampled / x$dt_accuracy_original
scores <- c("accuracy", "cramers_v", "pearson", "iva", "dt")
xs <- x[scores]
scatterplotMatrix(xs, diagonal = "boxplot", var.labels=c("acc","mcv","pcd","iva","cad"), smoother=F, groups=x$algorithm, legend.plot = F)