suppressMessages(library(argparser))
suppressMessages(library(tidyverse))
suppressMessages(library(readxl))
suppressMessages(library(ComBatFamQC))
suppressMessages(library(parallel))

## Read in arguments
p <- arg_parser("Post-harmonization processing step", hide.opts = FALSE)
p <- add_argument(p, "data", help = "path to the CSV or EXCEL file that contains data to be harmonized, covariates and batch information")
p <- add_argument(p, "--type", short = '-t', help = "post-harmonization processing type, eg: residual or age_trend", default = "age_trend")
p <- add_argument(p, "--features", short = '-f', help = "position of features/rois data(column numbers), eg: 1-5,9")
p <- add_argument(p, "--AGE", help = "column position of the age variable")
p <- add_argument(p, "--SEX", help = "column position of the sex variable")
p <- add_argument(p, "--ICV", help = "column position of the ICV variable")
p <- add_argument(p, "--Female", help = "female indicator, the value represent female in sex column.")
p <- add_argument(p, "--lowerquantile", short = '-l', help = "Specify a lower bound quantile. eg: 0.05, 0.25.", default = 0.25)
p <- add_argument(p, "--upperquantile", short = '-u', help = "Specify a upper bound quantile. eg: 0.75, 0.95.", default = 0.75)
p <- add_argument(p, "--Mu", help = "An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the mu formula. smooth: y ~ pb(age), linear: y ~ age, default: y ~ 1.", default = "smooth")
p <- add_argument(p, "--Sigma", help = "An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the sigma formula. smooth: ~ pb(age), linear: ~ age, default: ~ 1.", default = "smooth")
p <- add_argument(p, "--Nu", help = "An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the nu formula. smooth: ~ pb(age), linear: ~ age, default: ~ 1.", default = "default")
p <- add_argument(p, "--Tau", help = "An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the tau formula. smooth: ~ pb(age), linear: ~ age, default: ~ 1.", default = "default")
p <- add_argument(p, "--covariates", short = '-c', help = "position of covariates (column numbers)", default = "NULL")
p <- add_argument(p, "--model", short = '-m', help = "select the model function for harmonization, eg: lm, gam", default = "lm")
p <- add_argument(p, "--smooth", short = '-s', help = "provide the variables that require a smooth function", default = "NULL")
p <- add_argument(p, "--interaction", help = "specify the interaction terms in the format of col_index1*col_index2, eg 2*3,11*12", default = "NULL")
p <- add_argument(p, "--int_type", help = "specify an interaction type for gam models, eg: linear, factor-smooth, tensor", default = "linear")
p <- add_argument(p, "--random", short = '-r', help = "specify the random intercept-effects", default = "NULL")
p <- add_argument(p, "--rm", help = "specify the covariates (column numbers) to remove effects from, eg: 1-5,9", default = "NULL")
p <- add_argument(p, "--exist.model", help = "A boolean variable indicating whether an existing model is to be used", default = FALSE)
p <- add_argument(p, "--model.path", help = "path to the existing model", default = "NULL")
p <- add_argument(p, "--outdir", short = '-o', help = "full path (including the file name) where residual data should be written")
p <- add_argument(p, "--mout", help = "full path where regression models to be saved")
p <- add_argument(p, "--cores", help = "number of cores used for paralleling computing, please provide a numeric value", default = "all")
argv <- parse_args(p)

# Preprocess inputs
message('Checking inputs...')
if(is.na(argv$data)) stop("Missing input data") else {
  if(!grepl("csv$|xls$", argv$data)) stop("Input file must be a csv or an excel file") else {
    if(grepl("csv$", argv$data)) df = read.csv(argv$data) else df = read_excel(argv$data)
  }
}
df = data.frame(df)
if(is.na(argv$features)) stop("Please identify the position of features/rois.") else {
  col = gsub("-",":",argv$features)
  col_vec = eval(parse(text = paste0("c(", col, ")")))
}

features = colnames(df[col_vec])

if (argv$cores == "all"){
  cores = detectCores()
}else{
  cores = as.numeric(argv$cores)
}

if(argv$type == "age_trend"){
  message("Start....")
  age = colnames(df)[as.numeric(argv$AGE)]
  sex = colnames(df)[as.numeric(argv$SEX)]
  icv = colnames(df)[as.numeric(argv$ICV)]
  print(paste0(c(age, sex, icv)))
  df[[sex]] = as.factor(df[[sex]])
  df[[sex]] = sapply(df[[sex]], function(x){
    if(x == argv$Female){return("F")}else{return("M")}
  }, USE.NAMES = FALSE)
  message("Create sub_df(s)....")
  # Create sub_df for different features
  df_var = paste0("sub_df_", 1:length(features))
  for (i in 1:length(features)){
    sub_df = df[,c(features[i], age, sex, icv)] %>% na.omit() 
    colnames(sub_df) = c("y", "age", "sex", "icv")
    assign(df_var[i], sub_df)
  }
  
  # Create age_list
  
  age_list = mclapply(1:length(features), function(w){
    age_sub = age_list_gen (sub_df = eval(parse(text = paste0("sub_df_",w))),  lq = as.numeric(argv$lowerquantile), hq = as.numeric(argv$upperquantile), mu = argv$Mu, sigma = argv$Sigma, nu = argv$Nu, tau = argv$Tau)
    return(age_sub)
  }, mc.cores = cores) 
  names(age_list) = features
  
  quantile_type = c(paste0("quantile_", 100*as.numeric(argv$lowerquantile)), "median", paste0("quantile_", 100*as.numeric(argv$upperquantile)))
  
  ComBatFamQC::age_shiny(age_list, features, quantile_type)
}else if(argv$type == "residual"){
  
  if(argv$covariates == "NULL") {
    cov_col = NULL
    covariates = NULL
  } else {
    cov_col = gsub("-",":",argv$covariates)
    cov_col = eval(parse(text = paste0("c(", cov_col, ")")))
    covariates = colnames(df)[cov_col]
  }
  
  if(argv$model == "gam"){
    if(argv$covariates == "NULL") stop("Please provide covariates for gam model")
    if(argv$smooth == "NULL") stop("Please provide variables that require a smoothing function") else {
      smooth_col = gsub("-",":",argv$smooth)
      smooth_col = eval(parse(text = paste0("c(", smooth_col, ")")))
      smooth_var = colnames(df)[smooth_col]
      smooth = smooth_var
    }
  }else{
    smooth = eval(parse(text = argv$smooth))
  }
  
  if(argv$model == "lmer"){
    if(argv$random == "NULL") stop("Please specify random intercept-effects") else {
      random_col = gsub("-",":",argv$random)
      random_col = eval(parse(text = paste0("c(", random_col, ")")))
      random_var = colnames(df)[random_col]
      random = random_var
    }
  }else{
    random_col = eval(parse(text = argv$random))
    random = eval(parse(text = argv$random))
  }
  
  ## Interaction Wranggling
  if(argv$interaction == "NULL"){
    interaction = eval(parse(text = argv$interaction))
    smooth_int_type = NULL
  }else{
    interaction_l = lapply(str_split(argv$interaction, ",")[[1]], function(x) str_split(x,  "\\*")[[1]])
    interaction = sapply(interaction_l, function(x){
      x1 = colnames(df)[as.numeric(x[1])]
      x2 = colnames(df)[as.numeric(x[2])]
      element = paste0(x1, ",", x2)
    }, USE.NAMES = FALSE)
    smooth_int_type = str_split(argv$int_type, ",")[[1]]
  }
  
  if(argv$rm == "NULL"){
    rm = NULL
  }else{
    rm_col = gsub("-",":",argv$rm)
    rm_col = eval(parse(text = paste0("c(", rm_col, ")")))
    rm = colnames(df)[rm_col]
  }
  
  # Generate residuals
  result = residual_gen(type = argv$model, features = features, covariates = covariates, interaction = interaction, smooth = smooth, smooth_int_type = smooth_int_type, random = random, df = df, rm = rm, model = argv$exist.model, model_path = argv$model.path, cores = cores)
  
  if(!is.na(argv$outdir)){
    message("Saving residual data......")
    write_csv(result$residual, argv$outdir)
    message(sprintf("Results saved at %s", argv$outdir))  
  }
  
  if(!is.na(argv$mout)){
    message("Saving model......")
    saveRDS(result$model, argv$mout)
    message(sprintf("Model saved at %s", argv$mout))  
  }
}

