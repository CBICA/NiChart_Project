require(stats)
require(lme4)
require(parallel)
require(mgcv)
require(dplyr)
require(tidyverse)
require(stringr)

#' Post Harmonization Residual Generation
#'
#' Extract residuals after harmonization.
#'
#' @param type A model function name that is to be used (eg: "lmer", "lm", "gam").
#' @param features Features/rois to extract residuals from. \emph{n x p} data frame or matrix of observations where \emph{p} is the number of features and \emph{n} is the number of subjects.
#' @param covariates Name of covariates supplied to `model`.
#' @param interaction Expression of interaction terms supplied to `model` (eg: "age,diagnosis").
#' @param random Variable name of a random effect in linear mixed effect model.
#' @param smooth Variable name that requires a smooth function.
#' @param smooth_int_type Indicates the type of interaction in `gam` models. By default, smooth_int_type is set to be "linear", representing linear interaction terms. 
#' "categorical-continuous", "factor-smooth" both represent categorical-continuous interactions ("factor-smooth" includes categorical variable as part of the smooth), 
#' "tensor" represents interactions with different scales, and "smooth-smooth" represents interaction between smoothed variables.
#' @param df Harmonized dataset to extract residuals from.
#' @param rm variables to remove effects from.
#' @param model A boolean variable indicating whether an existing model is to be used.
#' @param model_path path to the existing model.
#' @param cores number of cores used for parallel computing.
#'
#' @return `residual_gen` returns a list containing the following components:
#' \item{model}{a list of regression models for all rois}
#' \item{residual}{Residual dataframe}
#' 
#' 
#' @import parallel
#' @import tidyverse
#' @import dplyr
#' @import stringr
#' @importFrom broom tidy
#' @importFrom mgcv gam 
#' @importFrom lme4 lmer
#' @importFrom stats lm median model.matrix prcomp predict qnorm update var anova as.formula coef resid na.omit complete.cases
#' 
#' @export
#' 
#' 

residual_gen <- function(type, features, covariates, interaction = NULL, random = NULL, smooth = NULL, smooth_int_type = NULL, df, rm = NULL, model = FALSE, model_path = NULL, cores = detectCores()){
  ## Characterize/factorize categorical variables
  obs_n = nrow(df)
  df = df[complete.cases(df[c(features, covariates, random)]),]
  obs_new = nrow(df)
  if((obs_n - obs_new) != 0){
    print(paste0(obs_n - obs_new, " observations are dropped due to missing values."))
  }else{
    print("No observation is dropped due to missing values.")
  }
  char_var = covariates[sapply(df[covariates], function(col) is.character(col) || is.factor(col))]
  enco_var = covariates[sapply(df[covariates], function(col) length(unique(col)) == 2 && all(unique(col) %in% c(0,1)))]
  df[char_var] =  lapply(df[char_var], as.factor)
  df[enco_var] =  lapply(df[enco_var], as.factor)
  cov_shiny = covariates
  char_var = c(char_var, enco_var)

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
  used_col = c(features)
  other_col = setdiff(colnames(df), used_col)
  other_info = df[other_col]
  
  ## generate interactions 
  int_result = interaction_gen(type = type, covariates = covariates, interaction = interaction, smooth = smooth, smooth_int_type = smooth_int_type)
  interaction_orig = interaction
  smooth_orig = smooth
  covariates = int_result$covariates
  interaction = int_result$interaction
  smooth = int_result$smooth
  
  if(!model){
    models = mclapply(features, function(y){
      model = model_gen(y = y, type = type, batch = NULL, covariates = covariates, interaction = interaction, random = random, smooth = smooth, df = df)
      return(model)
    }, mc.cores = cores)
  }else{
    models = readRDS(model_path)
  }
  
  if(!is.null(rm)){
    if(type!="lmer"){
      residuals = mclapply(1:length(features), function(i){
        model_coef = coef(models[[i]])
        rm_names = c()
        for (x in rm){
          sub_name = names(model_coef)[which(grepl(x, names(model_coef)))]
          rm_names = c(rm_names, sub_name)
        }
        rm_coef = model_coef[names(model_coef) %in% rm_names]
        predict_y = model.matrix(models[[i]])[, which(grepl(paste0(names(rm_coef), collapse = "|"), colnames(model.matrix(models[[i]]))))] %*% t(t(unname(rm_coef)))
        residual_y = df[[features[i]]] - predict_y
        residual_y = data.frame(residual_y)
      }, mc.cores = cores) %>% bind_cols()
    }else{
      df[[random]] = as.factor(df[[random]])
      residuals = mclapply(1:length(features), function(i){
        model_coef = coef(models[[i]])[[1]]
        rm_names = c()
        for (x in rm){
          sub_name = names(model_coef)[which(grepl(x, names(model_coef)))]
          rm_names = c(rm_names, sub_name)
        }
        rm_coef = model_coef[names(model_coef) %in% rm_names] %>% distinct()
        predict_y = model.matrix(models[[i]])[, which(grepl(paste0(names(rm_coef), collapse = "|"), colnames(model.matrix(models[[i]]))))] %*% t(rm_coef)
        residual_y = df[[features[i]]] - predict_y
        residual_y = data.frame(residual_y)
      }, mc.cores = cores) %>% bind_cols()
    }
    colnames(residuals) = features
    residuals = cbind(other_info, residuals)
    residuals = residuals[colnames(df)]
  }else{residuals = df}
  result = list("model" = models, "residual"= residuals)
  return(result)
}

utils::globalVariables(c("features", "covariates", "intercept", "random"))
