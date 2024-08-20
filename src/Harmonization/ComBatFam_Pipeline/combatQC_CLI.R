suppressMessages(library(argparser))
suppressMessages(library(tidyverse))
suppressMessages(library(dplyr))
suppressMessages(library(readxl))
suppressMessages(library(ComBatFamily))
suppressMessages(library(ComBatFamQC))
suppressMessages(library(lme4))
suppressMessages(library(mgcv))
suppressMessages(library(parallel))

## Read in arguments
p <- arg_parser("Streamline interactive batch effect diagnostics and apply ComBat harmonization", hide.opts = FALSE)
p <- add_argument(p, "data", help = "path to the CSV or EXCEL file that contains data to be harmonized, covariates and batch information")
p <- add_argument(p, "--visualization", short = '-v', help = "a boolean variable indicating whether to run harmonization or batch effect visualization. TRUE indicates batch effect visualization.", default = TRUE)
p <- add_argument(p, "--after", short = '-a', help = "a boolean variable indicating whether the dataset is before or after harmonization. TRUE indicates the dataset is after harmonization.", default = FALSE)
p <- add_argument(p, "--family", short = '-f', help = "combat family to use, comfam or covfam", default = "comfam")
p <- add_argument(p, "--features", help = "position of data to be harmonized(column numbers), eg: 1-5,9")
p <- add_argument(p, "--covariates", short = '-c', help = "position of covariates (column numbers)", default = "NULL")
p <- add_argument(p, "--batch", short = '-b', help = "position of batch column (column number)")
p <- add_argument(p, "--model", short = '-m', help = "select the model function for harmonization, eg: lm, gam", default = "lm")
p <- add_argument(p, "--smooth", short = '-s', help = "provide the variables that require a smooth function", default = "NULL")
p <- add_argument(p, "--interaction", help = "specify the interaction terms in the format of col_index1*col_index2, eg 2*3,11*12", default = "NULL")
p <- add_argument(p, "--int_type", help = "specify an interaction type for gam models, eg: linear, categorical-continuous, factor-smooth, tensor, smooth-smooth", default = "linear")
p <- add_argument(p, "--random", short = '-r', help = "specify the random intercept-effects", default = "NULL")
p <- add_argument(p, "--eb", help = "whether to use ComBat model with empirical Bayes for mean and variance", default = TRUE)
p <- add_argument(p, "--score_eb", help = "whether to use ComBat model with empirical Bayes for score mean and variance", default = FALSE)
p <- add_argument(p, "--robust.LS", help = "whether to use robust location and scale estimators for error variance and site effect parameters", default = FALSE)
p <- add_argument(p, "--ref.batch", help = "reference batch", default = "NULL")
p <- add_argument(p, "--score.model", help = "model for scores, defaults to NULL for fitting basic location and scale model without covariates on the scores", default = "NULL")
p <- add_argument(p, "--score.args", help = "list of arguments for score model, input should have the following format arg1,arg2,...", default = "NULL")
p <- add_argument(p, "--percent.var", help = "the number of harmonized principal component scores is selected to explain this proportion of the variance", default = 0.95)
p <- add_argument(p, "--n.pc", help = "if specified, this number of principal component scores is harmonized", default = "NULL")
p <- add_argument(p, "--std.var", help = "if TRUE, scales variances to be equal to 1 before PCA", default = TRUE)
p <- add_argument(p, "--predict", help = "a boolean variable indicating whether to run ComBat from scratch or apply existing model to new dataset (currently only work for original ComBat and ComBat-GAM)", default = FALSE)
p <- add_argument(p, "--object", help = "path to the ComBat model to be used for prediction, should have an extension of .rds", default = "NULL")
p <- add_argument(p, "--reference", help = "path to the CSV or EXCEL file that contains dataset to be considered as the reference group", default = "NULL")
p <- add_argument(p, "--outdir", short = '-o', help = "full path (including the file name) where harmonized data should be written")
p <- add_argument(p, "--mout", help = "full path where ComBat model to be saved")
p <- add_argument(p, "--cores", help = "number of cores used for paralleling computing, please provide a numeric value", default = "all")
p <- add_argument(p, "--mdmr", help = "a boolean variable indicating whether to run the MDMR test.", default = TRUE)
argv <- parse_args(p)

# Preprocess inputs
message('Checking inputs...')

## Checking the Data input
if(is.na(argv$data)) stop("Missing input data") else {
  if(!grepl("csv$|xls$", argv$data)) stop("Input file must be a csv or an excel file") else {
    message('Loading data...')
    if(grepl("csv$", argv$data)) df = read_csv(argv$data) else df = read_excel(argv$data)
  }
}

df = data.frame(df)
## Checking data
if(!argv$predict){
## Checking the Batch input
  message('Beginning data verification...')
  if(is.na(argv$batch)) stop("Please identify the position of batch column") else {
    bat_col = as.numeric(argv$batch)
    batch = colnames(df)[bat_col]
    message(paste0('The provided batch column is: ', batch))
  }
  
  ## Checking the Features input
  if(is.na(argv$features)) stop("Please identify the position of data to be harmonized") else {
    col = gsub("-",":",argv$features)
    col_vec = eval(parse(text = paste0("c(", col, ")")))
    features = colnames(df)[col_vec]
    features_wrong = colnames(df[features])[sapply(df[features], is.character)]
    if(length(features) == 1){
      if(length(features_wrong) == 0){
        message('Only one feature needs to be harmonized, and the data type is correct!')
      }else{
        message('Only one feature needs to be harmonized, and the data type is incorrect: it is character data!')
        stop("Incorrect feature type!")
      }
    }else{
      if(length(features_wrong) == 0){
        message(paste0('There are ', length(features), ' features to be harmonized, and the data types of all the feature columns are correct!'))
      }else{
        message(paste0('There are ', length(features), ' features to be harmonized, and the data types of ', length(features_wrong), ' feature columns are incorrect: they are character data!'))
        message('The wrong features are: ', paste0(features_wrong, collapse = ","))
        stop(paste0("Incorrect feature type! Considering removing ", paste0(which(colnames(df) %in% features_wrong), collapse = ","), " column."))
      }
    }
  }
  
  ## Checking the Covariates input
  if(argv$covariates == "NULL") {
    cov_col = NULL
    covariates = NULL
  } else {
    cov_col = gsub("-",":",argv$covariates)
    cov_col = eval(parse(text = paste0("c(", cov_col, ")")))
    covariates = colnames(df)[cov_col]
    if(length(covariates) == 1){
      message(paste0('The provided covariate column is: ', covariates))
    }else{
      message(paste0('The provided covariate columns are: ', paste0(covariates, collapse = ",")))
    }
  }
  
  ## Checking the Smooth Terms input
  if(argv$model == "gam"){
    if(argv$covariates == "NULL") stop("Please provide covariates for gam model")
    if(argv$smooth == "NULL") stop("Please provide variables that require a smoothing function") else {
      smooth_col = gsub("-",":",argv$smooth)
      smooth_col = eval(parse(text = paste0("c(", smooth_col, ")")))
      smooth_var = colnames(df)[smooth_col]
      smooth = smooth_var
      if(length(smooth) == 1){
        message(paste0('The provided smooth term is: ', smooth))
      }else{
        message(paste0('The provided smooth terms are: ', paste0(smooth, collapse = ",")))
      }
    }
  }else{
    smooth = eval(parse(text = argv$smooth))
  }
  
  ## Checking the Random Effects input
  if(argv$model == "lmer"){
    if(argv$random == "NULL") stop("Please specify random intercept-effects") else {
      random_col = gsub("-",":",argv$random)
      random_col = eval(parse(text = paste0("c(", random_col, ")")))
      random_var = colnames(df)[random_col]
      random = random_var
      if(length(random) == 1){
        message(paste0('The provided random effect variable is: ', random))
      }else{
        message(paste0('The provided random effect variables are: ', paste0(random, collapse = ",")))
      }
    }
  }else{
    random_col = eval(parse(text = argv$random))
    random = eval(parse(text = argv$random))
  }
  
  ## Read in the Reference Data (if it is provided)
  if(argv$reference != "NULL"){
    if(!grepl("csv$|xls$", argv$reference)) stop("Input file must be a csv or an excel file") else {
      if(grepl("csv$", argv$reference)) reference_df = read_csv(argv$reference) else reference_df = read_excel(argv$reference)
    }
    if(sum(!c(col_vec, bat_col, cov_col, random_col) %in% colnames(reference_df)) > 0) stop("Columns do not match between your reference dataset and the dataset to be harmonized!")
    reference_df = reference_df[complete.cases(reference_df[c(col_vec, bat_col, cov_col, random_col)]),]
  }else{
    reference_df = NULL
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
  message('The interaction term can be found below: ', paste0(gsub(",", "*", interaction), collapse = ","))
  message('The interaction term type can be found below: ', paste0(smooth_int_type, collapse = ","))
  object = NULL
}else{
  message("Starting data check for out-of-sample harmonization......")
  if(argv$object == "NULL") stop("Please provide a ComBat model for out-of-sample harmonization") else {
    object = readRDS(argv$object)
    batch = object$batch.name
    model_type = class(object$ComBat.model$fits[[1]])[1]
    features = colnames(object$ComBat.model$estimates$stand.mean)
    form_str = as.character(formula(object$ComBat.model$fits[[1]]))[3]
    if(model_type == "lm"){
      covariates = str_split(form_str, "\\+")[[1]][which(!grepl("batch|:", str_split(form_str, "\\+")[[1]]))]
      covariates = sapply(covariates, function(x) gsub(" ", "", x), USE.NAMES = FALSE)
      random = NULL
      type = "lm"
    }else if(model_type == "lmerMod"){
      covariates = str_split(form_str, "\\+")[[1]][which(!grepl("batch|:|\\(1", str_split(form_str, "\\+")[[1]]))]
      covariates = sapply(covariates, function(x) gsub(" ", "", x), USE.NAMES = FALSE)
      random = str_split(form_str, "\\+")[[1]][which(grepl("\\(1", str_split(form_str, "\\+")[[1]]))]
      random = sapply(random, function(x) gsub(" ", "", gsub("\\)", "", str_split(x, "\\|")[[1]][2])), USE.NAMES = FALSE)
      type = "lmer"
    }else if(model_type == "gam"){
      covariates = str_split(form_str, "\\+")[[1]][which(!grepl("batch|:|s\\(", str_split(form_str, "\\+")[[1]]))]
      covariates = sapply(covariates, function(x) gsub(" ", "", x), USE.NAMES = FALSE)
      smooth_term = str_split(form_str, "\\+")[[1]][which(grepl("s\\(", str_split(form_str, "\\+")[[1]]))]
      smooth_term = sapply(smooth_term, function(x) gsub("\\) ", "", gsub(" s\\(", "", x)), USE.NAMES = FALSE)
      covariates = c(covariates, smooth_term)
      random = NULL
      type = "gam"
    }
    if(sum(!c(covariates, random, features) %in% colnames(df)) > 0) stop("Columns do not match!")
  }
}



# Batch Effect Visualization
if(argv$visualization){
  message("preparing datasets for visualization......")
  result = visual_prep(type = argv$model, features = features, batch = batch, covariates = covariates, interaction = interaction, random = random, smooth = smooth, smooth_int_type = smooth_int_type, df = df, 
                       cores = if(argv$cores == "all") detectCores() else as.numeric(argv$cores),
                       mdmr = argv$mdmr)
  
  message("datasets are ready! Starting shiny app......")
  comfam_shiny(result, argv$after)
}else{
  message("Starting harmonization......")
  if(argv$predict){harm_result = combat_harm(df = df, predict = argv$predict, object = object)}else{
                  harm_result = combat_harm(features = features, batch = batch, covariates = covariates, df = df, type = argv$model, random = random, smooth = smooth, interaction = interaction, smooth_int_type = smooth_int_type, family = argv$family, 
                            ref.batch = if(argv$ref.batch == "NULL") eval(parse(text = argv$ref.batch)) else argv$ref.batch,
                            predict = argv$predict, 
                            object = object,
                            reference = reference_df)}
  
  comf_df = harm_result$harmonized_df
  if(is.na(argv$outdir)){message("Warning: The saving path is not provided, the harmonized data won't be saved!")}
  write_csv(comf_df, paste0(argv$outdir))
  message(sprintf("Results saved at %s", paste0(argv$outdir)))
  
  if(!is.na(argv$mout)){
    message("Saving ComBat model......")
    if(harm_result$com_family == "comfam"){
      harm_result$combat.object$ComBat.model$dat.combat = NULL
    }else{
      harm_result$combat.object$ComBat.model$dat.covbat = NULL
    }
    saveRDS(harm_result$combat.object, argv$mout)
    message(sprintf("ComBat model saved at %s", argv$mout))  
  }
}
  
