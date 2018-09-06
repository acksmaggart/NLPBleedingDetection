library(exact2x2)
# The contingency table of produced by McNemarContingencyCreation.py is
# 72 20
# 93 475

x <- matrix(c(72, 93, 20, 475), 2, 2)

mcnemar.exact(x)
