path = "/Users/bptripp/code/DOC-tf/generated-files/doc/border-relu1_1_0"
result_file = "probabilities.csv"
files = dir(path = path, pattern = ".csv", all.files = FALSE,
        full.names = FALSE, recursive = FALSE,
        ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)

result = data.frame()

for (file in files) {
  if (file != result_file) {
    ss = unlist(strsplit(file, "[-\\.]"))
    index = strtoi(ss[length(ss)-1], base=10)
    
    data = read.csv(file=paste(path, file, sep = "/"), header=TRUE, sep=",")
    anova = aov(count ~ object + foreground, data = data)
    p_object_null = summary(anova)[[1]][["Pr(>F)"]][[1]]
    p_foreground_null = summary(anova)[[1]][["Pr(>F)"]][[2]]
    
    
    result = rbind(result, list(index, p_object_null, p_foreground_null))
  }
}

# doesn't seem possible to do this when initializing empty data frame
colnames(result) = c("index", "p-object", "p-foreground")

write.csv(result, file=paste(path, result_file, sep = "/"))
