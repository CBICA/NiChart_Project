require(parallel)
require(gamlss)
require(gamlss.dist)
require(dplyr)
require(tidyr)
require(shiny)
require(DT)
require(bslib)
require(bsicons)
require(shinydashboard)


#' Lifespan Age Trends
#'
#' Provide estimated lifespan age trends of neuroimaging-derived brain structures through shiny app.
#'
#' @param age_list A list contains all rois' true volumes and age trend estimates.
#' @param features A vector of roi names.
#' @param quantile_type A vector of quantile types.
#'
#' @import shiny
#' @import bsicons
#' @import shinydashboard
#' @import bslib 
#' @import parallel
#' @import tidyverse
#' @import tidyr
#' @import dplyr
#' @import ggplot2
#' @importFrom gamlss gamlss gamlss.control predictAll getQuantile ps pb
#' @importFrom gamlss.dist BCT NO
#' @importFrom DT datatable formatStyle styleEqual DTOutput renderDT
#' @importFrom utils head
#' 
#' @export

age_shiny = function(age_list, features, quantile_type){
  quantile_type = quantile_type
  ui = function(request) {
    fluidPage(
      theme = bslib::bs_theme(version = 4, bootswatch = "minty"),
      titlePanel("LIFESPAN Age Trends"),
      sidebarLayout(
        sidebarPanel(
          selectInput("features", "Select ROI", choices = features, selected = features[1]),
          radioButtons("sex", "Sex control", choices = c("Female", "Male", "Female vs. Male (Only for visualization)"), selected = "Female"),
          radioButtons("quantile", "Select the quantile level", choices = quantile_type, selected = quantile_type[1])
          ),
        mainPanel(
          tabsetPanel(
            tabPanel("Age Trend",  
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Age Trend Plots",
                         shiny::plotOutput("ageplot"))),
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Age Trend Table",
                         DT::DTOutput("agetable")))
                     )
          )   
        )
      )
    )
  }
  
  server = function(input, output, session) {
    output$ageplot = shiny::renderPlot({
      result = age_list[[input$features]]
      if(input$sex == "Female"){
        ggplot(result$true_df, aes(x = .data[["age"]], y = .data[["y"]])) +
          geom_point(color = "steelblue") +
          geom_line(data = result$predicted_df_sex %>% filter(sex == "F"), mapping = aes(x = age, y = prediction, linetype = type), color = "red") +
          labs(x = "Age", y = "ROI Volume")
      }else if(input$sex == "Male"){
        ggplot(result$true_df, aes(x = .data[["age"]], y = .data[["y"]])) +
          geom_point(color = "steelblue") +
          geom_line(data = result$predicted_df_sex %>% filter(sex == "M"), mapping = aes(x = age, y = prediction, linetype = type), color = "purple") +
          labs(x = "Age", y = "ROI Volume")
      }else if(input$sex == "Female vs. Male (Only for visualization)"){
        ggplot(result$true_df, aes(x = .data[["age"]], y = .data[["y"]])) +
          geom_point(color = "steelblue") +
          geom_line(data = result$predicted_df_sex %>% filter(type == "median"), mapping = aes(x = age, y = prediction, color = sex)) +
          labs(x = "Age", y = "ROI Volume")
      }
    })
    output$agetable = DT::renderDT({
      result = age_list[[input$features]]
      if(input$sex == "None"){
        age_table = result$predicted_df %>% filter(type == input$quantile) %>% dplyr::select(age, prediction) %>% rename("AverageVolume" = "prediction", "Age" = "age") 
        min_age = floor(min(age_table$Age)/10)*10
        max_age = floor(max(age_table$Age)/10)*10
        age_table = lapply(seq(min_age, max_age, 10), function(x){
          sub_age = age_table %>% filter(Age >= x) %>% head(1)
          return(sub_age)
        }) %>% bind_rows()
      }else if(input$sex == "Female"){
        age_table = result$predicted_df_sex %>% filter(type == input$quantile, sex == "F") %>% dplyr::select(age, prediction) %>% rename("AverageVolume" = "prediction", "Age" = "age") 
        min_age = floor(min(age_table$Age)/10)*10
        max_age = floor(max(age_table$Age)/10)*10
        age_table = lapply(seq(min_age, max_age, 10), function(x){
          sub_age = age_table %>% filter(Age >= x) %>% head(1)
          return(sub_age)
        }) %>% bind_rows()
        age_table[["PercentageChange (%)"]] = c(NA, 100*diff(age_table$AverageVolume)/na.omit(lag(age_table$AverageVolume)))
        age_table = age_table %>% mutate(Age = sprintf("%.0f", Age),
                                         AverageVolume = sprintf("%.3f", AverageVolume),
                                         `PercentageChange (%)` = sprintf("%.3f", `PercentageChange (%)`))
      }else if(input$sex == "Male"){
        age_table = result$predicted_df_sex %>% filter(type == input$quantile, sex == "M") %>% dplyr::select(age, prediction) %>% rename("AverageVolume" = "prediction", "Age" = "age") 
        min_age = floor(min(age_table$Age)/10)*10
        max_age = floor(max(age_table$Age)/10)*10
        age_table = lapply(seq(min_age, max_age, 10), function(x){
          sub_age = age_table %>% filter(Age >= x) %>% head(1)
          return(sub_age)
        }) %>% bind_rows()
        age_table[["PercentageChange (%)"]] = c(NA, 100*diff(age_table$AverageVolume)/na.omit(lag(age_table$AverageVolume)))
        age_table = age_table %>% mutate(Age = sprintf("%.0f", Age),
                                         AverageVolume = sprintf("%.3f", AverageVolume),
                                         `PercentageChange (%)` = sprintf("%.3f", `PercentageChange (%)`))
      }else if(input$sex == "Female vs. Male (Only for visualization)"){
        age_table_F = result$predicted_df_sex %>% filter(type == input$quantile, sex == "F") %>% dplyr::select(age, prediction) %>% rename("AverageVolume_F" = "prediction", "Age" = "age")
        age_table_M = result$predicted_df_sex %>% filter(type == input$quantile, sex == "M") %>% dplyr::select(age, prediction) %>% rename("AverageVolume_M" = "prediction", "Age" = "age")
        min_age = floor(min(age_table_F$Age)/10)*10
        max_age = floor(max(age_table_F$Age)/10)*10
        age_table_F = lapply(seq(min_age, max_age, 10), function(x){
          sub_age = age_table_F %>% filter(Age >= x) %>% head(1)
          return(sub_age)
        }) %>% bind_rows()
        age_table_M = lapply(seq(min_age, max_age, 10), function(x){
          sub_age = age_table_M %>% filter(Age >= x) %>% head(1)
          return(sub_age)
        }) %>% bind_rows()
        age_table = cbind(age_table_F, age_table_M[c("AverageVolume_M")])
        age_table[["PercentageChange_F (%)"]] = c(NA, 100*diff(age_table$AverageVolume_F)/na.omit(lag(age_table$AverageVolume_F)))
        age_table[["PercentageChange_M (%)"]] = c(NA, 100*diff(age_table$AverageVolume_M)/na.omit(lag(age_table$AverageVolume_M)))
        age_table = age_table %>% mutate(Age = sprintf("%.0f", Age),
                                         AverageVolume_F = sprintf("%.3f", AverageVolume_F),
                                         AverageVolume_M = sprintf("%.3f", AverageVolume_M),
                                         `PercentageChange_F (%)` = sprintf("%.3f", `PercentageChange_F (%)`),
                                         `PercentageChange_M (%)` = sprintf("%.3f", `PercentageChange_M (%)`)
                                         )
      }
      age_table %>% DT::datatable(options = list(columnDefs = list(list(className = 'dt-center', 
                                                            targets = "_all")))) 
    })
  }
  shinyApp(ui = ui, server = server)
}

#' Age Trend Estimates Generation
#'
#' A GAMLSS model using a Box-Cox t distribution was fitted separately to rois of interest, 
#' to establish normative reference ranges as a function of age for the volume of a specific roi.
#'
#' @param sub_df A two-column dataset that contains age and roi volume related information. column y: roi volumes, column age: age.
#' @param lq The lower bound quantile. eg: 0.25, 0.05
#' @param hq The upper bound quantile. eg: 0.75, 0.95
#' @param mu An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the mu formula.
#' "smooth": y ~ pb(age), "linear": y ~ age, "default": y ~ 1.
#' @param sigma An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the sigma formula.
#' "smooth": ~ pb(age), "linear": ~ age, "default": ~ 1.
#' @param nu An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the nu formula.
#' "smooth": ~ pb(age), "linear": ~ age, "default": ~ 1.
#' @param tau An indicator of whether to smooth age variable, include it as a linear term or only include the intercept in the tau formula.
#' "smooth": ~ pb(age), "linear": ~ age, "default": ~ 1.
#'
#' @return `age_list_gen` returns a list containing the following components:
#' \item{true_df}{a dataframe contains the true age and roi volume infomation}
#' \item{predicted_df}{a dataframe contains the estimated age trend}
#' \item{predicted_df_sex}{a dataframe contains the estimated age trend adjusting sex}
#' 
#' @export

age_list_gen = function(sub_df, lq = 0.25, hq = 0.75, mu = "smooth", sigma = "smooth", nu= "default", tau = "default"){
  if(mu == "smooth") {
    #mu_form = as.formula("y ~ pb(age)")
    mu_form_sex = as.formula("y ~ pb(age) + sex + icv")
    }else if(mu == "linear"){
    #mu_form = as.formula("y ~ age")
    mu_form_sex = as.formula("y ~ age + sex + icv")
    }else if(mu == "default"){
      #mu_form = as.formula("y ~ 1")
      mu_form_sex = as.formula("y ~ sex + icv")
    }
  
  if(sigma == "smooth") {
    sig_form = as.formula("~ pb(age)")
  }else if(sigma == "linear"){
    sig_form = as.formula("~ age")
  }else if(sigma == "default"){
    sig_form = as.formula("~ 1")
  }
  
  if(nu == "smooth") {
    nu_form = as.formula("~ pb(age)")
  }else if(nu == "linear"){
    nu_form = as.formula("~ age")
  }else if(nu == "default"){
    nu_form = as.formula("~ 1")
  }
  
  if(tau == "smooth") {
    tau_form = as.formula("~ pb(age)")
  }else if(tau == "linear"){
    tau_form = as.formula("~ age")
  }else if(tau == "default"){
    tau_form = as.formula("~ 1")
  }
                   
  #mdl = gamlss(mu_form, 
  #             sigma.formula=sig_form,
  #             nu.formula=nu_form,
  #             tau.formula=tau_form,
  #             family=BCT(),
  #             data = sub_df,
  #             control = gamlss.control(n.cyc = 100))
  
  mdl_sex = gamlss(mu_form_sex, 
               sigma.formula=sig_form,
               nu.formula=nu_form,
               tau.formula=tau_form,
               family=NO(),
               data = sub_df,
               control = gamlss.control(n.cyc = 100))
  # predict hypothetical data
  min_age = min(sub_df[["age"]])
  max_age = max(sub_df[["age"]])
  age_test = seq(from = min_age, to = max_age,length.out = 1000)
  mean_icv = mean(sub_df$icv)
  #y_test = matrix(data=mean(sub_df[["y"]]),nrow=1000,ncol=1)
  #data_test = data.frame(cbind(y_test, age_test)) 
  #colnames(data_test) = c("y", "age")
  
  #params = predictAll(object = mdl,data = sub_df, newdata=data_test, 
  #                    output='matrix',type="response",
  #                    y.value="median",what=c("mu", "sigma", "nu", "tau"))
  quantiles = c(lq, 0.5, hq)
  #predictions_quantiles = matrix(data=0,ncol=3,nrow=1000)
  #for (i in 1:length(quantiles)){
  #  Qua <- getQuantile(obj = mdl, quantile = quantiles[i], term="age", fixed.at=list())
  #  predictions_quantiles[,i] = Qua(age_test)
  #}
  #colnames(predictions_quantiles) = c(paste0("quantile_", 100*lq), "median", paste0("quantile_", 100*hq))
  #age_df = data.frame(cbind(age = age_test, predictions_quantiles)) %>% 
  #  pivot_longer(colnames(predictions_quantiles), names_to = "type", values_to = "prediction")
  #
  age_df_sex = lapply(c("F", "M"), function(x){
    predictions_quantiles_female = matrix(data=0,ncol=3,nrow=1000)
    for (i in 1:length(quantiles)){
      Qua <- getQuantile(obj = mdl_sex, quantile = quantiles[i], term="age", fixed.at=list(sex = x, icv = mean_icv))
      predictions_quantiles_female[,i] = Qua(age_test)
    }
    colnames(predictions_quantiles_female) = c(paste0("quantile_", 100*lq), "median", paste0("quantile_", 100*hq))
    age_df_sex = data.frame(cbind(age = age_test, predictions_quantiles_female)) %>% 
      pivot_longer(colnames(predictions_quantiles_female), names_to = "type", values_to = "prediction") %>% mutate(sex = x)
  }) %>% bind_rows()
  
  return_list = list("true_df" = sub_df, "predicted_df_sex" = age_df_sex)
  return(return_list)
}

utils::globalVariables(c("Age", "AverageVolume", "AverageVolume_F", "AverageVolume_M", "PercentageChange (%)", "PercentageChange_F (%)", "PercentageChange_M (%)", "age", "sex", "icv", "prediction", "type", "true_df", "y", "mdl", "mdl_sex", "age_df_sex"))

