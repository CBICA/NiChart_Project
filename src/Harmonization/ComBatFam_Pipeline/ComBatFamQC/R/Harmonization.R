require(ComBatFamily)
require(tidyverse)

#' ComBatFamily Harmonization
#'
#' Conduct harmonization using the ComBatFamily package, which currently includes 4 types of harmonization techniques: 1) Original ComBat, 2) Longitudinal ComBat, 3) ComBat-GAM, 4) CovBat.
#'
#' @param result A list derived from `visual_prep()` that contains datasets for shiny visualization. Can be skipped if `features`, `batch`, `covariates` and `df` are provided.
#' @param features Features to be harmonized. \emph{n x p} data frame or matrix of observations where \emph{p} is the number of features and \emph{n} is the number of subjects. Can be skipped if `result` is provided.
#' @param batch Factor indicating batch (often equivalent to site or scanner). Can be skipped if `result` is provided.
#' @param covariates Name of covariates supplied to `model`. Can be skipped if `result` is provided.
#' @param df Dataset to be harmonized. Can be skipped if `result` is provided.
#' @param type A model function name that is used or to be used in the ComBatFamily Package (eg: "lmer", "lm").
#' @param random Variable name of a random effect in linear mixed effect model.
#' @param smooth Variable name that requires a smooth function.
#' @param interaction Expression of interaction terms supplied to `model` (eg: "age,diagnosis").
#' @param smooth_int_type A vector that indicates the types of interaction in `gam` models. By default, smooth_int_type is set to be NULL, "linear" represents linear interaction terms. 
#' "categorical-continuous", "factor-smooth" both represent categorical-continuous interactions ("factor-smooth" includes categorical variable as part of the smooth), 
#' "tensor" represents interactions with different scales, and "smooth-smooth" represents interaction between smoothed variables.
#' @param family combat family to use, comfam or covfam.
#' @param ref.batch reference batch.
#' @param predict A boolean variable indicating whether to run ComBat from scratch or apply existing model to new dataset (currently only work for original ComBat and ComBat-GAM).
#' @param object Existing ComBat model.
#' @param reference Dataset to be considered as the reference group.
#' @param out_ref_include A boolean variable indicating whether the reference data should be included in the harmonized data output.
#' @param ... Additional arguments to `comfam` or `covfam` models.
#' 
#' @return `combat_harm` returns a list containing the following components:
#' \item{eb_df}{A dataframe contains empirical Bayes estimates}
#' \item{com_family}{ComBat family to be considered: comfam, covfam}
#' \item{harmonized_df}{Harmonized dataset}
#' \item{combat.object}{saved ComBat model}
#'
#' @import tidyverse
#' @import ComBatFamily
#' @importFrom stats formula
#' 
#' @export

combat_harm <- function(result = NULL, features = NULL, batch = NULL, covariates = NULL, df = NULL, type = "lm", random = NULL, smooth = NULL, interaction = NULL, smooth_int_type = NULL, family = "comfam", ref.batch = NULL, predict = FALSE, object = NULL, reference = NULL, out_ref_include = TRUE, ...){
  
  if(!is.null(result)){
    message("Taking the ComBatQC result as the input...")
    df = result$info$df
    char_var = result$info$char_var
    batch = result$info$batch
    features = result$info$features
    cov_shiny = result$info$cov_shiny
    obs_n = nrow(df)
    df = df[complete.cases(df[c(features, batch, cov_shiny, random)]),]
    obs_new = nrow(df)
    if((obs_n - obs_new) != 0){
      print(paste0(obs_n - obs_new, " observations are dropped due to missing values."))
    }else{
      print("No observation is dropped due to missing values.")
    }
    df[[batch]] = as.factor(df[[batch]])
    df[char_var] =  lapply(df[char_var], as.factor)
  }else{
    if(!predict){
      message("The ComBatQC result is not provided, the required parameters should be specified...")}else{
      if(is.null(object)) stop("Please provide the saved ComBat model!")
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
    }
    obs_n = nrow(df)
    df = df[complete.cases(df[c(features, batch, covariates, random)]),]
    obs_new = nrow(df)
    if((obs_n - obs_new) != 0){
      print(paste0(obs_n - obs_new, " observations are dropped due to missing values."))
    }else{
      print("No observation is dropped due to missing values.")
    }
    df[[batch]] = as.factor(df[[batch]])
    char_var = covariates[sapply(df[covariates], function(col) is.character(col) || is.factor(col))]
    enco_var = covariates[sapply(df[covariates], function(col) length(unique(col)) == 2 && all(unique(col) %in% c(0,1)))]
    df[char_var] =  lapply(df[char_var], as.factor)
    df[enco_var] =  lapply(df[enco_var], as.factor)
    cov_shiny = covariates
    char_var = c(char_var, enco_var)
    
    # Summary
    summary_df = df %>% group_by(eval(parse(text = batch))) %>% summarize(count = n(), percentage = 100*count/nrow(df))
    colnames(summary_df) = c(batch, "count", "percentage (%)")
    summary_df = summary_df %>% mutate(remove = case_when(count < 3 ~ "removed",
                                                          .default = "keeped"))
    batch_rm = summary_df %>% filter(remove == "removed") %>% pull(eval(parse(text = batch))) %>% droplevels()
    if(length(batch_rm) > 0){
      print(paste0("Batch levels that contain less than 3 observations are dropped: ", length(batch_rm), " levels are dropped, corresponding to ", df %>% filter(eval(parse(text = batch)) %in% batch_rm) %>% nrow(), " observations."))
    }else{print("Batch levels that contain less than 3 observations are dropped: no batch level is dropped.")}
    df = df %>% filter(!eval(parse(text = batch)) %in% batch_rm) 
    df[[batch]] = df[[batch]] %>% droplevels()
  }
  
  
  if(!is.null(random)){
    for (r in random){
      df[[r]] = as.factor(df[[r]])
    }
  }
  
  ## drop univariate features
  features_orig = df[features]
  n_orig = length(colnames(features_orig))
  features_new = features_orig[, apply(features_orig, 2, function(col) { length(unique(col)) > 1 })]
  n_new = length(colnames(features_new))
  dropped_col = NULL
  if (n_orig > n_new){
    dropped_col = setdiff(colnames(features_orig), colnames(features_new))
    print(paste0(n_orig - n_new, " univariate feature column(s) are dropped: ", dropped_col))
  }
  features = colnames(features_new)

  int_result = interaction_gen(type = type, covariates = cov_shiny, interaction = interaction, smooth = smooth, smooth_int_type = smooth_int_type)
  covariates = int_result$covariates
  interaction = int_result$interaction
  smooth = int_result$smooth

  
  # Empirical Estimates
  if (is.null(covariates)){
    if(type == "lmer"){
      form_c = NULL
      combat_c = df[random]
    }else{
      form_c = NULL
      combat_c = NULL
    }
  }else{
    if(type == "lmer"){
      form_c = df[covariates]
      combat_c = cbind(df[cov_shiny], df[random])
    }else{
      form_c = df[covariates]
      combat_c = df[cov_shiny]
    }
  }
  
  if (is.null(reference)){
    if (!predict){
      message("Starting first-time harmonization...")
      form = form_gen(x = type, c = form_c, i = interaction, random = random, smooth = smooth)
      if(family == "comfam"){
        ComBat_run = ComBatFamily::comfam(data = df[features],
                                          bat = df[[batch]], 
                                          covar = combat_c,
                                          model = eval(parse(text = type)), 
                                          formula = as.formula(form), 
                                          ref.batch = ref.batch,
                                          ...)
        gamma_hat = ComBat_run$estimates$gamma.hat
        delta_hat = ComBat_run$estimates$delta.hat
        gamma_prior = ComBat_run$estimates$gamma.prior
        delta_prior = ComBat_run$estimates$delta.prior
        batch_name = rownames(gamma_hat)
        eb_list = list(gamma_hat, delta_hat, gamma_prior, delta_prior)
        eb_name = c("gamma_hat", "delta_hat", "gamma_prior", "delta_prior")
        eb_df = lapply(1:4, function(i){
          eb_df_long = data.frame(eb_list[[i]]) %>% mutate(batch = as.factor(batch_name), .before = 1) %>% tidyr::pivot_longer(2:(dim(eb_list[[i]])[2]+1), names_to = "features", values_to = "eb_values") %>% mutate(type = eb_name[i]) 
          return(eb_df_long)
        }) %>% bind_rows()
      }else{
        ComBat_run = ComBatFamily::covfam(data = df[features],
                                          bat = df[[batch]] , 
                                          covar = combat_c,
                                          model = eval(parse(text = type)), 
                                          formula = as.formula(form),
                                          ref.batch = ref.batch,
                                          ...)
        gamma_hat = ComBat_run$combat.out$estimates$gamma.hat
        delta_hat = ComBat_run$combat.out$estimates$delta.hat
        gamma_prior = ComBat_run$combat.out$estimates$gamma.prior
        delta_prior = ComBat_run$combat.out$estimates$delta.prior
        score_gamma_hat = ComBat_run$scores.combat$estimates$gamma.hat
        score_delta_hat = ComBat_run$scores.combat$estimates$delta.hat
        score_gamma_prior = ComBat_run$scores.combat$estimates$gamma.prior
        score_delta_prior = ComBat_run$scores.combat$estimates$delta.prior
        batch_name = rownames(gamma_hat)
        eb_list = list(gamma_hat, delta_hat, gamma_prior, delta_prior, score_gamma_hat, score_delta_hat, score_gamma_prior, score_delta_prior)
        eb_name = c("gamma_hat", "delta_hat", "gamma_prior", "delta_prior", "score_gamma_hat", "score_delta_hat", "score_gamma_prior", "score_delta_prior")
        eb_df = lapply(1:8, function(i){
          eb_df_long = data.frame(eb_list[[i]]) %>% mutate(batch = as.factor(batch_name), .before = 1) %>% tidyr::pivot_longer(2:(dim(eb_list[[i]])[2]+1), names_to = "features", values_to = "eb_values") %>% mutate(type = eb_name[i]) 
          return(eb_df_long)
        }) %>% bind_rows()
      }
    }else{
      message("Starting out-of-sample harmonization using the saved ComBat Model...")
      #if(is.null(object)) stop("Please provide the saved ComBat model!")
      ComBat_run = predict(object = object$ComBat.model, newdata = df[features], newbat = df[[batch]], newcovar = combat_c, ...)
      gamma_hat = ComBat_run$estimates$gamma.hat
      delta_hat = ComBat_run$estimates$delta.hat
      gamma_prior = ComBat_run$estimates$gamma.prior
      delta_prior = ComBat_run$estimates$delta.prior
      batch_name = rownames(gamma_hat)
      eb_list = list(gamma_hat, delta_hat, gamma_prior, delta_prior)
      eb_name = c("gamma_hat", "delta_hat", "gamma_prior", "delta_prior")
      eb_df = lapply(1:4, function(i){
        eb_df_long = data.frame(eb_list[[i]]) %>% mutate(batch = as.factor(batch_name), .before = 1) %>% tidyr::pivot_longer(2:(dim(eb_list[[i]])[2]+1), names_to = "features", values_to = "eb_values") %>% mutate(type = eb_name[i]) 
        return(eb_df_long)
      }) %>% bind_rows()
    }
  }else{
    message("Starting out-of-sample harmonization using the reference dataset...")
    reference[[batch]] = as.factor(reference[[batch]])
    reference[char_var] =  lapply(reference[char_var], as.factor)
    if(!is.null(random)){
      for (r in random){
        reference[[r]] = as.factor(reference[[r]])
      }
    }
    ## check if reference data is included in the new data
    other_info = setdiff(colnames(reference), features)
    n_ref = df %>% semi_join(reference[other_info]) %>% nrow() 
    if(n_ref == nrow(reference)){
      message("The reference data is included in the new unharmonized dataset")
      untouched = reference
      new_data = df %>% anti_join(reference[other_info])
    }else{
      message("The reference data is separated from the new unharmonized dataset")
      untouched = NULL
      new_data = df
    }
    
    reference[[batch]] = "reference"
    df_c = rbind(reference, new_data)
    df_c[[batch]] = as.factor(df_c[[batch]])
    if (is.null(covariates)){
      form_c = NULL
      combat_c = NULL
    }else{
      if(type == "lmer"){
        form_c = df_c[covariates]
        combat_c = cbind(df_c[cov_shiny], df_c[random])
      }else{
        form_c = df_c[covariates]
        combat_c = df_c[cov_shiny]
      }
    }
    form = form_gen(x = type, c = form_c, i = interaction, random = random, smooth = smooth)
    if(family == "comfam"){
      ComBat_run = ComBatFamily::comfam(data = df_c[features],
                                        bat = df_c[[batch]], 
                                        covar = combat_c,
                                        model = eval(parse(text = type)), 
                                        formula = as.formula(form), 
                                        ref.batch = "reference",
                                        ...)
      gamma_hat = ComBat_run$estimates$gamma.hat
      delta_hat = ComBat_run$estimates$delta.hat
      gamma_prior = ComBat_run$estimates$gamma.prior
      delta_prior = ComBat_run$estimates$delta.prior
      batch_name = rownames(gamma_hat)
      eb_list = list(gamma_hat, delta_hat, gamma_prior, delta_prior)
      eb_name = c("gamma_hat", "delta_hat", "gamma_prior", "delta_prior")
      eb_df = lapply(1:4, function(i){
        eb_df_long = data.frame(eb_list[[i]]) %>% mutate(batch = as.factor(batch_name), .before = 1) %>% tidyr::pivot_longer(2:(dim(eb_list[[i]])[2]+1), names_to = "features", values_to = "eb_values") %>% mutate(type = eb_name[i]) 
        return(eb_df_long)
      }) %>% bind_rows()
    }else{
      ComBat_run = ComBatFamily::covfam(data = df_c[features],
                                        bat = df_c[[batch]] , 
                                        covar = combat_c,
                                        model = eval(parse(text = type)), 
                                        formula = as.formula(form),
                                        ref.batch = "reference",
                                        ...)
      gamma_hat = ComBat_run$combat.out$estimates$gamma.hat
      delta_hat = ComBat_run$combat.out$estimates$delta.hat
      gamma_prior = ComBat_run$combat.out$estimates$gamma.prior
      delta_prior = ComBat_run$combat.out$estimates$delta.prior
      score_gamma_hat = ComBat_run$scores.combat$estimates$gamma.hat
      score_delta_hat = ComBat_run$scores.combat$estimates$delta.hat
      score_gamma_prior = ComBat_run$scores.combat$estimates$gamma.prior
      score_delta_prior = ComBat_run$scores.combat$estimates$delta.prior
      batch_name = rownames(gamma_hat)
      eb_list = list(gamma_hat, delta_hat, gamma_prior, delta_prior, score_gamma_hat, score_delta_hat, score_gamma_prior, score_delta_prior)
      eb_name = c("gamma_hat", "delta_hat", "gamma_prior", "delta_prior", "score_gamma_hat", "score_delta_hat", "score_gamma_prior", "score_delta_prior")
      eb_df = lapply(1:8, function(i){
        eb_df_long = data.frame(eb_list[[i]]) %>% mutate(batch = as.factor(batch_name), .before = 1) %>% tidyr::pivot_longer(2:(dim(eb_list[[i]])[2]+1), names_to = "features", values_to = "eb_values") %>% mutate(type = eb_name[i]) 
        return(eb_df_long)
      }) %>% bind_rows()
    }
  }
  
  # Result
  used_col = c(features, cov_shiny, batch)
  other_col = setdiff(colnames(df), used_col)
  other_info = df[other_col]
  
  if (is.null(reference)){
    if(family == "covfam"){
      com_family = "covfam"
      comf_df = ComBat_run$dat.covbat
      comf_df = cbind(other_info, df[batch], df[cov_shiny], comf_df)
    }else{
      com_family = "comfam"
      comf_df = ComBat_run$dat.combat
      comf_df = cbind(other_info, df[batch], df[cov_shiny], comf_df)
    }
  }else{
    if(family == "covfam"){
      com_family = "covfam"
      comf_df = ComBat_run$dat.covbat[(nrow(reference)+1):nrow(df_c),]
      comf_df = cbind(new_data[other_col], new_data[batch], new_data[cov_shiny], comf_df)
      comf_df = comf_df[colnames(df)]
      if(out_ref_include){comf_df = rbind(untouched, comf_df)}
    }else{
      com_family = "comfam"
      comf_df = ComBat_run$dat.combat[(nrow(reference)+1):nrow(df_c),]
      comf_df = cbind(new_data[other_col], new_data[batch], new_data[cov_shiny], comf_df)
      comf_df = comf_df[colnames(df)]
      if(out_ref_include){comf_df = rbind(untouched, comf_df)}
    }
  }
  comf_df = comf_df[colnames(df)]
  combat_result =  list("eb_df" = eb_df, "com_family" = com_family, "harmonized_df" = comf_df, "combat.object" = list("ComBat.model" = ComBat_run, "batch.name" = batch))
  return(combat_result)
}