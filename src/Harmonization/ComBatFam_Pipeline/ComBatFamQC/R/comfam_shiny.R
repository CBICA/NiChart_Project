require(shiny)
require(DT)
require(tidyverse)
require(dplyr)
require(tidyr)
require(bslib)
require(bsicons)
require(ggplot2)
require(shinydashboard)

#' Batch Effect Interactive Visualization
#'
#' Provide interactive batch/site effect visualization through shiny app.
#'
#' @param result A list derived from `visual_prep()` that contains datasets for shiny visualization.
#' @param after A boolean variable indicating whether the dataset is before or after harmonization. The default value is FALSE.
#' @param ... Additional arguments to `comfam` or `covfam` models.
#'
#' @import tidyverse
#' @import tidyr
#' @import dplyr
#' @import ggplot2
#' @import shiny
#' @import bsicons
#' @import shinydashboard
#' @import bslib
#' @importFrom DT datatable formatStyle styleEqual DTOutput renderDT
#' @importFrom stats reorder
#' @importFrom utils write.csv
#'
#' @export

comfam_shiny = function(result, after = FALSE, ...){
  info = result$info
  type = info$type
  df = info$df
  batch = info$batch
  features = info$features
  covariates = info$cov_shiny
  char_var = info$char_var
  num_var = setdiff(covariates, char_var)

  ## UI Design
  ui = function(request) {
    fluidPage(
      theme = bslib::bs_theme(version = 4, bootswatch = "minty"),
      titlePanel("Batch Effect Diagnostics"),
      sidebarLayout(
        sidebarPanel(
          conditionalPanel(condition="input.tabselected==2",
                           fluidRow(
                             shinydashboard::box(
                               width = NULL,
                               title = "Data Summary",
                               radioButtons("type", "Select output type", choices = c("Plot", "Table"), selected = "Plot"),
                               uiOutput("cov_status")
                             )
                           )
          ),
          conditionalPanel(condition="input.tabselected==3",
                           selectInput("feature", "Select Feature", choices = features, selected = features[1]),
                           radioButtons("resid_all", "Select whether to include all batch levels", choices = c("Yes", "No"), selected = "Yes"),
                           uiOutput("resid_all_control"),
                           radioButtons("resid_color", "Select whether to color by batch variable", choices = c("Yes", "No"), selected = "No"),
                           radioButtons("resid_label", "Select whether to include labels for the x axis.", choices = c("Yes", "No"), selected = "No"),
                           uiOutput("resid_label_control")
          ),
          conditionalPanel(condition="input.tabselected==4",
                           fluidRow(
                             shinydashboard::box(
                               width = 12,
                               title = "Selection of Principal Components ",
                               selectInput("PC1", "Select the first PC", choices = colnames(result$pr.feature$x), selected = colnames(result$pr.feature$x)[1]),
                               selectInput("PC2", "Select the second PC", choices = colnames(result$pr.feature$x), selected = colnames(result$pr.feature$x)[2]),
                               radioButtons("pca_label", "Select whether to show the legend of plots.", choices = c("Yes", "No"), selected = "No"),
                               radioButtons("pca_all", "Select whether to include all batch levels", choices = c("Yes", "No"), selected = "Yes"),
                               uiOutput("pca_all_control"))),
                           fluidRow(
                             shinydashboard::box(
                               width = 12,
                               title = "Variance Explained",
                               style = "background-color: white;",
                               DT::DTOutput("pc_variance"))),
                           fluidRow(
                             shinydashboard::box(
                               width = 12,
                               title = "MDMR",
                               style = "background-color: white;",
                               uiOutput("mdmr_control")))

          ),
          conditionalPanel(condition="input.tabselected==6",
                           fluidRow(
                             shinydashboard::box(
                               width = NULL,
                               title = "Harmonization Setup",
                               radioButtons("com_type", "Select ComBatFam type", choices = c("comfam", "covfam"), selected = "comfam"),
                               uiOutput("com_type_note"),
                               radioButtons("com_model", "Select ComBat Family model to be used", choices = c("lm", "lmer", "gam"), selected = type),
                               uiOutput("com_model_note"),
                               radioButtons("eb_control", "Select whether the Empirical Bayes (EB) method should be used", choices = c("Yes", "No"), selected = "Yes"),
                               uiOutput("eb_control_select"),
                               uiOutput("smooth_select"),
                               uiOutput("random_select"),
                               selectInput("ref_bat_select", "Select the reference batch", choices = c("None", levels(df[[batch]])), selected = "None"),
                               textInput("interaction", "Enter the potential interaction terms:", value = paste0(gsub("\\,", "\\*", info$interaction_orig), collapse = ",")),
                               uiOutput("interaction_note"),
                               uiOutput("smooth_int_type_control"),
                               uiOutput("smooth_int_type_control_note")
                             )
                           ),
                           fluidRow(
                             shinydashboard::box(
                               width = NULL,
                               title = "Harmonization",
                               textInput("save_path", "Save harmonized dataframe to:"),
                               actionButton("ComBat", "Harmonize and Save Data"),
                               verbatimTextOutput("output_msg"),
                               textInput("model_save_path", "Save ComBat Model to:"),
                               actionButton("ComBat_model", "Save ComBat Model"),
                               verbatimTextOutput("output_msg_model")
                             )
                           ),
                           selectInput("batch_selection", "Select the batches to be shown on the graph", choices = c("All", levels(info$df[[batch]])), selected = "All"),
                           uiOutput("cov_eb")

          ),
          conditionalPanel(condition="input.tabselected==5",
                           fluidRow(
                             shinydashboard::box(
                               width = NULL,
                               title = "Additive Batch Effect",
                               if(type == "lmer"){
                                 radioButtons("test_batch", "Type of Statistical Tests for Additive Batch Effect", choices = c("ANOVA", "Kruskal-Wallis", "Kenward-Roger (liner mixed model)"), selected = "ANOVA")
                               }else{
                                 radioButtons("test_batch", "Type of Statistical Tests for Additive Batch Effect", choices = c("ANOVA", "Kruskal-Wallis"), selected = "ANOVA")
                               },
                               uiOutput("test_batch_explain"))),
                           fluidRow(
                             shinydashboard::box(
                               width = NULL,
                               title = "Multiplicative Batch Effect",
                               radioButtons("test_variance", "Type of Statistical Tests for Multiplicative Batch Effect", choices = c("Fligner-Killeen", "Levene's Test", "Bartlett's Test"), selected = "Fligner-Killeen"),
                               uiOutput("test_variance_explain"))),
                           fluidRow(
                             shinydashboard::box(
                               width = NULL,
                               uiOutput("mul_adjustment")))
          ),
          conditionalPanel(condition="input.tabselected==1",
                           radioButtons("data_view", "Overview Type", choices = c("Complete Data", "Exploratory Analysis"), selected = "Complete Data"),
                           uiOutput("explore_bar")
          )),
        mainPanel(
          tabsetPanel(
            tabPanel("Data Overview", value = 1,
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Usage",
                         style = "background-color: #f0f0f0;",
                         shiny::htmlOutput("data_usage_explain"))),
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Data",
                         DT::DTOutput("data_frame"))),
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         uiOutput("data_ui")))),
            tabPanel("Summary", value = 2,
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Batch Sample Size Summary",
                         shiny::uiOutput("output"))),
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Covariate Distribution",
                         shiny::uiOutput("cov_output")))),
            tabPanel("Residual Plot", value = 3,
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Additive Batch Effect",
                         shiny::uiOutput("res_add_explain"),
                         shiny::plotOutput("res_add"))),
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Multiplicative Batch Effect",
                         shiny::uiOutput("res_ml_explain"),
                         shiny::plotOutput("res_ml")))),
            tabPanel("Diagnosis of Global Batch Effect", value = 4,
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "PCA",
                         shiny::plotOutput("pca"))),
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "T-SNE",
                         shiny::plotOutput("tsne")))),
            tabPanel("Diagnosis of Individual Batch Effect", value = 5,
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Additive Batch Effect Test",
                         shiny::uiOutput("test_batch_ui"),
                         shiny::uiOutput("sig_pct_batch"))),
                     fluidRow(
                       shinydashboard::box(
                         width = 12,
                         title = "Multiplicative Batch Effect Test",
                         shiny::uiOutput("test_variance_ui"),
                         shiny::uiOutput("sig_pct_variance")))),
            if(!after){
              tabPanel("Harmonization", value = 6,
                       fluidRow(
                         shinydashboard::box(
                           width = 12,
                           shiny::uiOutput("eb_explain"))),
                       fluidRow(
                         shinydashboard::box(
                           width = 12,
                           title = "Location Parameters",
                           shiny::plotOutput("eb_location"))),
                       fluidRow(
                         shinydashboard::box(
                           width = 12,
                           title = "Scale Paramaters",
                           shiny::plotOutput("eb_scale"))))
            },
            id = "tabselected"
          )
        )
      )
    )
  }
  ## Server Design
  server = function(input, output, session) {
    combat_result_s = reactiveVal(NULL)

    ############### Data Overview #############################
    output$explore_bar = shiny::renderUI({
      if(input$data_view == "Exploratory Analysis"){
        fluidRow(
          shinydashboard::box(
            width = 12,
            title = "Controlling Visualizations",
            selectInput("single_feature", "Select one feature to investigate", choices = features, selected = features[1]),
            selectInput("single_cov", "Select one covariate to investigate", choices = covariates, selected = covariates[1]),
            radioButtons("num_var_control_batch", "Select whether to separate by batch", choices = c("Yes", "No"), selected = "No"),
            shiny::uiOutput("batch_sep_control"),
            shiny::uiOutput("cov_visual_control")))
      }
    })

    output$cov_visual_control = shiny::renderUI({
      if (!is.null(covariates)){
        if(input$single_cov %in% num_var){
          fluidRow(
            shinydashboard::box(
              width = 12,
              title = "Controlling Visualizations",
              radioButtons("num_var_control", "Select the type of smooth method", choices = c("lm", "loess", "glm", "gam"), selected = "lm"),
              sliderInput("se", "Select transparency", min = 0, max = 1, value = 0.2, step = 0.1)))
        }else if(input$single_cov %in% char_var){
          fluidRow(
            shinydashboard::box(
              width = 12,
              title = "Controlling Visualizations",
              radioButtons("char_var_control", "Select the type of plot to display", choices = c("boxplot", "boxplot with points", "density plot"), selected = "boxplot")))
        }
      }
    })

    output$batch_sep_control = shiny::renderUI({
      if(input$num_var_control_batch == "Yes"){
        checkboxGroupInput("overview_batch_select", "Select batch levels to include:", choices = levels(df[[batch]]), selected = levels(df[[batch]]))
      }
    })

    output$data_ui = shiny::renderUI({
      if(input$data_view == "Exploratory Analysis"){
        fluidRow(
          column(width = 6,
                 shinydashboard::box(
                   width = NULL,
                   title = "Batch vs Feature",
                   shiny::plotOutput("batch_vi"))),
          column(width = 6,
                 shinydashboard::box(
                   width = NULL,
                   title = "Covariate vs Feature",
                   shiny::plotOutput("cov_vi"))))
      }else{shiny::htmlOutput("data_rm_explain")}
    })

    output$data_rm_explain = shiny::renderUI({
      data_rm = result$summary_df %>% filter(remove == "removed")
      if(nrow(data_rm) == 0){
        HTML(print("Batch levels that contain less than 3 observations are dropped: <strong>no batch level is dropped</strong>."))
      }else{
        HTML(paste0("Batch levels that contain less than 3 observations are dropped: <strong>", nrow(data_rm), " levels are dropped, corresponding to ", sum(data_rm$count), " observations</strong>."))
      }
    })

    output$batch_vi = shiny::renderPlot({
      if(input$num_var_control_batch == "No"){
        ggplot(df, aes(x = eval(parse(text = input$single_feature)))) +
          geom_density(fill = "blue", alpha = 0.3) +
          labs(x = input$single_feature) +
          theme(
            axis.title.x = element_text(size = 12, face = "bold"),
            axis.title.y = element_text(size = 12, face = "bold"),
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_text(size = 12, face = "bold"),
          )
      }else{
        overview_sub_df = df %>% filter(eval(parse(text = batch)) %in% input$overview_batch_select)
        ggplot(overview_sub_df, aes(x = eval(parse(text = input$single_feature)), fill = eval(parse(text = batch)))) +
          geom_density(alpha = 0.3) +
          labs(x = input$single_feature, fill = batch) +
          theme(
            axis.title.x = element_text(size = 12, face = "bold"),
            axis.title.y = element_text(size = 12, face = "bold"),
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_text(size = 12, face = "bold"),
          )
      }
    })

    output$cov_vi = shiny::renderPlot({
      if (!is.null(covariates)){
        if(input$single_cov %in% num_var){
          if(input$num_var_control_batch == "No"){
            ggplot(df, aes(x = eval(parse(text = input$single_cov)), y = eval(parse(text = input$single_feature)))) +
              geom_point() +
              geom_smooth(method = input$num_var_control, alpha = as.numeric(input$se)) +
              labs(x = input$single_cov, y = input$single_feature) +
              theme(
                axis.title.x = element_text(size = 12, face = "bold"),
                axis.title.y = element_text(size = 12, face = "bold"),
                axis.text.x = element_text(size = 12, face = "bold"),
                axis.text.y = element_text(size = 12, face = "bold"),
              )
          }else{
            overview_sub_df = df %>% filter(eval(parse(text = batch)) %in% input$overview_batch_select)
            ggplot(overview_sub_df, aes(x = eval(parse(text = input$single_cov)), y = eval(parse(text = input$single_feature)), color = eval(parse(text = batch)))) +
              geom_point() +
              geom_smooth(method = input$num_var_control, aes(fill = eval(parse(text = batch))), alpha = as.numeric(input$se)) +
              labs(x = input$single_cov, y = input$single_feature, color = batch, fill = batch) +
              theme(
                axis.title.x = element_text(size = 12, face = "bold"),
                axis.title.y = element_text(size = 12, face = "bold"),
                axis.text.x = element_text(size = 12, face = "bold"),
                axis.text.y = element_text(size = 12, face = "bold"),
              )
          }
        }else if(input$single_cov %in% char_var){
          if(input$char_var_control == "boxplot"){
            if(input$num_var_control_batch == "No"){
              ggplot(df, aes(x = eval(parse(text = input$single_cov)), y = eval(parse(text = input$single_feature)), fill = eval(parse(text = input$single_cov)))) +
                geom_boxplot() +
                scale_fill_brewer(palette="Pastel1") +
                labs(x = input$single_cov, y = input$single_feature, fill = input$single_cov) +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )

            }else{
              overview_sub_df = df %>% filter(eval(parse(text = batch)) %in% input$overview_batch_select)
              ggplot(overview_sub_df, aes(x = eval(parse(text = input$single_cov)), y = eval(parse(text = input$single_feature)), fill = eval(parse(text = batch)))) +
                geom_boxplot() +
                scale_fill_brewer(palette="Pastel1") +
                labs(x = input$single_cov, y = input$single_feature, fill = batch) +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }
          }else if(input$char_var_control == "boxplot with points"){
            if(input$num_var_control_batch == "No"){
              ggplot(df, aes(x = eval(parse(text = input$single_cov)), y = eval(parse(text = input$single_feature)), fill = eval(parse(text = input$single_cov)))) +
                geom_boxplot() +
                geom_jitter(aes(shape = eval(parse(text = input$single_cov)))) +
                scale_fill_brewer(palette="Pastel1") +
                labs(x = input$single_cov, y = input$single_feature, fill = input$single_cov, shape = input$single_cov) +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }else{
              overview_sub_df = df %>% filter(eval(parse(text = batch)) %in% input$overview_batch_select)
              ggplot(overview_sub_df, aes(x = eval(parse(text = input$single_cov)), y = eval(parse(text = input$single_feature)), fill = eval(parse(text = batch)))) +
                geom_boxplot() +
                geom_jitter(aes(shape = eval(parse(text = input$single_cov)))) +
                scale_fill_brewer(palette="Pastel1") +
                labs(x = input$single_cov, y = input$single_feature, fill = batch, shape = input$single_cov) +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }
          }else if(input$char_var_control == "density plot"){
            ggplot(df, aes(x = eval(parse(text = input$single_feature)), fill = eval(parse(text = input$single_cov)))) +
              geom_density(alpha = 0.3) +
              labs(x = input$single_feature, fill = input$single_cov) +
              theme(
                axis.title.x = element_text(size = 12, face = "bold"),
                axis.title.y = element_text(size = 12, face = "bold"),
                axis.text.x = element_text(size = 12, face = "bold"),
                axis.text.y = element_text(size = 12, face = "bold"),
              )
          }
        }
      }
    })


    output$data_usage_explain = shiny::renderUI({
      HTML(paste0("Below is a preview of the data used for batch effect evaluation (and harmonization). Please review the dataset carefully to ensure correct identification of features, covariates and batch variable. <br> ",
                  "<strong>Note</strong>: The corresponding color themes for each type of variable are as follows: <br> ",
                  "<strong>Covariates</strong> - <strong><span style='color: pink;'>pink</span></strong>; ",
                  "<strong>Features</strong> - <strong><span style='color: lightyellow;'>lightyellow</span></strong>; ",
                  "<strong>Batch</strong> - <strong><span style='color: lightblue;'>lightblue</span></strong>."))
    })

    output$data_frame = DT::renderDT({
      other = setdiff(colnames(df), c(batch, covariates, features))
      df_show = df[c(batch, covariates, features, other)]
      if(input$data_view == "Exploratory Analysis"){
        df_show %>% dplyr::select(batch, covariates, input$single_feature) %>% DT::datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                                                                                   targets = "_all")))) %>% formatStyle(
                                                                                                                                     covariates,
                                                                                                                                     backgroundColor = "pink"
                                                                                                                                   ) %>% formatStyle(
                                                                                                                                     input$single_feature,
                                                                                                                                     backgroundColor = "lightyellow"
                                                                                                                                   ) %>% formatStyle(
                                                                                                                                     batch,
                                                                                                                                     backgroundColor = "lightblue"
                                                                                                                                   )
      }else{
        df_show %>% DT::datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                        targets = "_all")))) %>% formatStyle(
                                                                          covariates,
                                                                          backgroundColor = "pink"
                                                                        ) %>% formatStyle(
                                                                          features,
                                                                          backgroundColor = "lightyellow"
                                                                        ) %>% formatStyle(
                                                                          batch,
                                                                          backgroundColor = "lightblue"
                                                                        )
      }
    })

    ############### Summary #############################
    output$cov_status = shiny::renderUI({
      if (!is.null(covariates)){
        selectInput("cov", "Select covariate", choices = covariates, selected = covariates[1])
      }
    })

    output$output = shiny::renderUI({
      if (input$type == "Plot") {
        plotOutput("plot")
      } else if (input$type == "Table") {
        DT::DTOutput("table")
      }
    })
    output$plot = shiny::renderPlot({
      ggplot(result$summary_df %>% filter(remove == "keeped"), aes(x = count, y = eval(parse(text = batch)))) +
        geom_bar(stat = "identity", fill = "aquamarine") +
        #geom_text(aes(label = count), hjust = 1.5, position = position_dodge(0.9), size = 3, colour = "black") +
        labs(x = "Count", y = "Batch") +
        theme(#plot.title = element_text(hjust = 0.5),
          axis.title.x = element_text(size = 12, face = "bold"),
          axis.title.y = element_text(size = 12, face = "bold"),
          axis.text.x = element_text(size = 12, face = "bold"),
          axis.text.y = element_text(size = 12, face = "bold"),
          axis.ticks.y = element_blank())
    })
    output$table = DT::renderDT({
      result$summary_df %>% mutate(`percentage (%)` = sprintf("%.3f", `percentage (%)`)) %>% arrange(desc(remove)) %>%
        DT::datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                            targets = "_all")))) %>% formatStyle(
                                                              'remove',
                                                              target = 'row',
                                                              backgroundColor = styleEqual(c("removed"), "lightyellow")
                                                            )
    })
    output$cov_output = shiny::renderUI({
      if (is.null(covariates)){
        textOutput("cov_text")
      }else if (input$type == "Plot") {
        plotOutput("cov_plot")
      } else if (input$type == "Table") {
        DT::DTOutput("cov_table")
      }
    })

    output$cov_text = shiny::renderText({
      print("No covariate is preserved")
    })

    output$cov_plot = shiny::renderPlot({
      if (!is.null(covariates)){
        if(input$cov %in% num_var){
          ggplot(df, aes(x = eval(parse(text = input$cov)), y = reorder(as.factor(eval(parse(text = batch))), eval(parse(text = input$cov)), Fun = median), fill = eval(parse(text = batch))))+
            geom_boxplot(alpha = 0.3) +
            #geom_point() +
            labs(x = input$cov, y = "Batch", fill = "Covariate") +
            theme(#plot.title = element_text(hjust = 0.5),
              legend.position = "none",
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
              #axis.text.y = element_blank(),
              axis.ticks.y = element_blank())
        }else if(input$cov %in% char_var){
          df_c = df %>% group_by(eval(parse(text = batch)), eval(parse(text = input$cov))) %>% dplyr::tally() %>% mutate(percentage = n/sum(n))
          colnames(df_c) = c(batch, input$cov, "n", "percentage")
          ggplot(df_c, aes(y = as.factor(eval(parse(text = batch))), x = n, fill = eval(parse(text = input$cov)))) +
            geom_bar(stat="identity", position ="fill") +
            #geom_text(aes(label = paste0(sprintf("%1.1f", percentage*100),"%")), position = position_fill(vjust=0.5), colour="black", size = 3) +
            scale_fill_brewer(palette = "Pastel1") +
            labs(x = "Percentage", y = "Batch", fill = input$cov) +
            theme(#plot.title = element_text(hjust = 0.5),
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
              #axis.text.y = element_blank(),
              axis.ticks.y = element_blank())
        }
      }
    })
    output$cov_table =  DT::renderDT({
      if (!is.null(covariates)){
        if(input$cov %in% num_var){
          cov_summary_table = df %>% group_by(eval(parse(text = batch))) %>% summarize(min = min(eval(parse(text = input$cov))), mean = mean(eval(parse(text = input$cov))), max = max(eval(parse(text = input$cov))))
          colnames(cov_summary_table) = c(batch, "min", "mean", "max")
          cov_summary_table = cov_summary_table %>% mutate(mean = round(mean, 3))
          cov_summary_table %>% DT::datatable()
        }else if(input$cov %in% char_var){
          cov_summary_table = df %>% group_by(eval(parse(text = batch)), eval(parse(text = input$cov))) %>% dplyr::tally() %>% mutate(percentage = 100*n/sum(n))
          colnames(cov_summary_table) = c(batch, input$cov, "n", "percentage (%)")
          cov_summary_table %>% mutate(`percentage (%)` = sprintf("%.3f", `percentage (%)`)) %>% DT::datatable()
        }
      }
    })

    ############### Residual Plot #############################
    output$resid_all_control = shiny::renderUI({
      if(input$resid_all == "No"){
        checkboxGroupInput("resid_batch_select", "Select batch levels to include:", choices = levels(df[[batch]]), selected = levels(df[[batch]]))
      }
    })
    output$resid_label_control = shiny::renderUI({
      if(input$resid_label == "Yes"){
        sliderInput("label_angle", "Customize the angle of the label", min = 0, max = 90, value = 0)
      }
    })

    output$res_add_explain = shiny::renderUI({
      HTML(print("A <strong>noticeable deviation of the mean from zero</strong> in the additive-residual box plot indicates the presence of an additive batch effect"))
    })

    output$res_add = shiny::renderPlot({
      add_mean = result$residual_add_df %>% group_by(result$residual_add_df[[batch]]) %>% summarize(across(features, median, .names = "mean_{.col}")) %>% ungroup()
      colnames(add_mean) = c(batch, colnames(add_mean)[-1])
      result$residual_add_df = result$residual_add_df %>% left_join(add_mean, by = c(batch))
      if(input$resid_color == "No"){

        if(input$resid_all == "Yes"){
          add_plot = ggplot(result$residual_add_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]])) +
            geom_boxplot() +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual") +
            theme(
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
            )
        }else{
          sub_plot_df = result$residual_add_df %>% filter(eval(parse(text = batch)) %in% input$resid_batch_select)
          add_plot = ggplot(sub_plot_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]])) +
            geom_boxplot() +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual") +
            theme(
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
            )
        }

        if(input$resid_label == "No"){
          add_plot +
            theme(axis.text.x = element_blank(),
                  axis.ticks.x = element_blank())
        }else{
          add_plot +
            theme(axis.text.x = element_text(angle = input$label_angle, hjust = 0.5, size = 12, face = "bold"),
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold")
            )
        }

      }else{

        if(input$resid_all == "Yes"){
          add_plot = ggplot(result$residual_add_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]], fill = eval(parse(text = batch)))) +
            geom_boxplot(alpha = 0.3) +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual")
        }else{
          sub_plot_df = result$residual_add_df %>% filter(eval(parse(text = batch)) %in% input$resid_batch_select)
          add_plot = ggplot(sub_plot_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]], fill = eval(parse(text = batch)))) +
            geom_boxplot(alpha = 0.3) +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual") +
            theme(
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
            )
        }

        if(input$resid_label == "No"){
          add_plot +
            theme(axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  legend.position = "none")
        }else{
          add_plot +
            theme(axis.text.x = element_text(angle = input$label_angle, hjust = 0.5, size = 12, face = "bold"),
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                  legend.position = "none"
            )
        }

      }
    })

    output$res_ml_explain = shiny::renderUI({
      HTML(print("A <strong>substantial variation</strong> in the multiplicative-residual box plot demonstrates a potential multiplicative batch effect."))
    })

    output$res_ml = shiny::renderPlot({
      add_mean = result$residual_add_df %>% group_by(result$residual_add_df[[batch]]) %>% summarize(across(features, median, .names = "mean_{.col}")) %>% ungroup()
      colnames(add_mean) = c(batch, colnames(add_mean)[-1])
      result$residual_ml_df = result$residual_ml_df %>% left_join(add_mean, by = c(batch))
      if(input$resid_color == "No"){

        if(input$resid_all == "Yes"){
          mul_plot = ggplot(result$residual_ml_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]])) +
            geom_boxplot() +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual") +
            theme(
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
            )
        }else{
          sub_plot_df = result$residual_ml_df %>% filter(eval(parse(text = batch)) %in% input$resid_batch_select)
          mul_plot = ggplot(sub_plot_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]])) +
            geom_boxplot() +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual") +
            theme(
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
            )
        }
        if(input$resid_label == "No"){
          mul_plot +
            theme(axis.text.x = element_blank(),
                  axis.ticks.x = element_blank())
        }else{
          mul_plot +
            theme(axis.text.x = element_text(angle = input$label_angle, hjust = 0.5, size = 12, face = "bold"),
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold")
            )
        }
      }else{
        if(input$resid_all == "Yes"){
          mul_plot = ggplot(result$residual_ml_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]], fill = eval(parse(text = batch)))) +
            geom_boxplot(alpha = 0.3) +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual") +
            theme(
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
            )
        }else{
          sub_plot_df = result$residual_ml_df %>% filter(eval(parse(text = batch)) %in% input$resid_batch_select)
          mul_plot = ggplot(sub_plot_df, aes(x = reorder(as.factor(eval(parse(text = batch))), .data[[paste0("mean_",input$feature)]]), y = .data[[input$feature]], fill = eval(parse(text = batch)))) +
            geom_boxplot(alpha = 0.3) +
            geom_hline(yintercept = 0, linetype = "dashed", col = "red") +
            labs(x = "Batch", y = "Residual") +
            theme(
              axis.title.x = element_text(size = 12, face = "bold"),
              axis.title.y = element_text(size = 12, face = "bold"),
              axis.text.x = element_text(size = 12, face = "bold"),
              axis.text.y = element_text(size = 12, face = "bold"),
            )
        }

        if(input$resid_label == "No"){
          mul_plot +
            theme(axis.text.x = element_blank(),
                  axis.ticks.x = element_blank(),
                  legend.position = "none")
        }else{
          mul_plot +
            theme(axis.text.x = element_text(angle = input$label_angle, hjust = 0.5, size = 12, face = "bold"),
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                  legend.position = "none"
            )
        }

      }
    })

    ############### Dimensionality Reduction #############################
    output$pc_variance = DT::renderDT({
      result$pca_summary %>% filter(Principal_Component %in% c(input$PC1, input$PC2)) %>% dplyr::select(Principal_Component, Variance_Explained) %>%
        add_row(Principal_Component = "Total", Variance_Explained = sum(.$Variance_Explained)) %>% mutate(Variance_Explained = sprintf("%.3f", Variance_Explained)) %>%
        DT::datatable(options = list(columnDefs = list(list(className = 'dt-center', targets = "_all"))))
    })

    output$mdmr_control = renderUI({
      if(!is.null(result$mdmr.summary)){
        tagList(list(
          uiOutput("mdmr_note"),
          DT::DTOutput("test_batch_mdmr"),
          uiOutput("mdmr_sig_text")
        ))
      }else{
        uiOutput("mdmr_skip")
      }
    })

    output$mdmr_skip = renderUI({
      HTML("The MDMR test has been skipped.")
    })
    
    output$pca_all_control = shiny::renderUI({
      if(input$pca_all == "No"){
        checkboxGroupInput("pca_batch_select", "Select batch levels to include:", choices = levels(df[[batch]]), selected = levels(df[[batch]]))
      }
    })

    output$pca = shiny::renderPlot({
      if(input$pca_all == "Yes"){
        pca_plot_base = ggplot(result$pca_df, aes(x = .data[[input$PC1]], y = .data[[input$PC2]], color = .data[[batch]])) +
          geom_point() +
          labs(x = input$PC1, y = input$PC2, color = "Batch") +
          theme(
            axis.title.x = element_text(size = 12, face = "bold"),
            axis.title.y = element_text(size = 12, face = "bold"),
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_text(size = 12, face = "bold"),
          )
        if(input$pca_label == "No"){pca_plot_base + guides(color = "none")}else{pca_plot_base}
      }else{
        sub_pca_df = result$pca_df %>% filter(eval(parse(text = batch)) %in% input$pca_batch_select)
        pca_plot_base = ggplot(sub_pca_df, aes(x = .data[[input$PC1]], y = .data[[input$PC2]], color = .data[[batch]])) +
          geom_point() +
          labs(x = input$PC1, y = input$PC2, color = "Batch") +
          theme(
            axis.title.x = element_text(size = 12, face = "bold"),
            axis.title.y = element_text(size = 12, face = "bold"),
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_text(size = 12, face = "bold"),
          )
        if(input$pca_label == "No"){pca_plot_base + guides(color = "none")}else{pca_plot_base}
      }
    })
    output$tsne = shiny::renderPlot({
      if(input$pca_all == "Yes"){
        tsne_plot_base = ggplot(result$tsne_df, aes(x = cor_1, y = cor_2, color = .data[[batch]])) +
          geom_point() +
          labs(x = "Dim 1", y = "Dim 2", color = "Batch") +
          theme(
            axis.title.x = element_text(size = 12, face = "bold"),
            axis.title.y = element_text(size = 12, face = "bold"),
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_text(size = 12, face = "bold"),
          )
        if(input$pca_label == "No"){tsne_plot_base + guides(color = "none")}else{tsne_plot_base}
      }else{
        sub_tsne_df = result$tsne_df %>% filter(eval(parse(text = batch)) %in% input$pca_batch_select)
        tsne_plot_base = ggplot(sub_tsne_df, aes(x = cor_1, y = cor_2, color = .data[[batch]])) +
          geom_point() +
          labs(x = "Dim 1", y = "Dim 2", color = "Batch") +
          theme(
            axis.title.x = element_text(size = 12, face = "bold"),
            axis.title.y = element_text(size = 12, face = "bold"),
            axis.text.x = element_text(size = 12, face = "bold"),
            axis.text.y = element_text(size = 12, face = "bold"),
          )
        if(input$pca_label == "No"){tsne_plot_base + guides(color = "none")}else{tsne_plot_base}
      }
    })

    ############### Harmonization if needed #############################
    output$com_type_note = renderUI({
      if(input$com_type == "comfam"){
        HTML(print("<strong>Note</strong>: Correcting Batch Effects <strong>(ComBat Family)</strong> <br><br>"))
      }else{
        HTML(print("<strong>Note</strong>: Correcting Covariance Batch Effects <strong>(CovBat Family)</strong> <br><br>"))
      }
    })

    output$com_model_note = renderUI({
      if(input$com_model == "lm"){
        HTML(print("<strong>Note</strong>: a method designed for batch effect correction in cross-sectional data with linear covariate effects. <strong>(Original ComBat)</strong> <br><br>"))
      }else if(input$com_model == "lmer"){
        HTML(print("<strong>Note</strong>: a method accounts for intra-subject correlation in longitudinal data by incorporating random effects into the model. <strong>(Longitudinal ComBat)</strong> <br><br>"))
      }else if(input$com_model == "gam"){
        HTML(print("<strong>Note</strong>: a method allows for preservation of non-linear covariate effects through use of the generalized additive model. <strong>(ComBat-GAM)</strong> <br><br>"))
      }
    })

    output$eb_control_select = renderUI({
      if(input$com_type == "covfam"){
        radioButtons("score_eb_control", "Select whether the EB method should be used for the ComBat in PCA Scores", choices = c("Yes", "No"), selected = "No")
      }
    })

    output$smooth_select = renderUI({
      if(input$com_model == "gam"){
        checkboxGroupInput("smooth", "Select smooth terms:", choices = result$info$cov_shiny, selected = info$smooth_orig)
      }
    })

    output$random_select = renderUI({
      if(input$com_model == "lmer"){
        selectInput("random", "Select the random effect when considering longitudinal combat", choices = colnames(info$df), selected = colnames(info$df)[1])
      }
    })

    output$interaction_note = renderUI({
      HTML(print("eg: covariate1*covariate2,covariate3*covariate4 <br><br>"))
    })

    output$smooth_int_type_control = renderUI({
      if(input$com_model == "gam"){
        textInput("smooth_int_type", "Enter the types of potential interaction terms:", value = paste0(info$smooth_int_type, collapse = ","))
      }
    })

    output$smooth_int_type_control_note = renderUI({
      if(input$com_model == "gam"){
        HTML(paste0("eg: linear,factor-smooth <br><br>",
                    "<strong>Note</strong>: The detailed explanation of the interaction type can be found below <br><br>",
                    "<strong>linear</strong>: linear interaction terms <br>",
                    "<strong>categorical-continuous</strong>: categorical-continuous interactions (s(covariate1, by = categorical_covariate)) <br>",
                    "<strong>factor-smooth</strong>: includes categorical variable as part of the smooth (s(covariate1,categorical_covariate, bs = 'fs')) <br>",
                    "<strong>tensor</strong>: represents interactions with different scales (ti(covariate1,covariate2)) <br>",
                    "<strong>smooth-smooth</strong>: represents interaction between smoothed variables (s(covariate1,covariate2)) <br><br>"))
      }
    })

    output$cov_eb = renderUI({
      if(input$com_type == "covfam"){
        radioButtons("eb_check_type", "Select which type of EB assumption to be checked", choices = c("First-step ComBat", "ComBat in Scores"), selected = "First-step ComBat")
      }
    })

    output$eb_explain = shiny::renderUI({
      HTML(paste0("<br>",
                  "This section aims to check the <strong>prior distribution assumption</strong> of the L/S model batch parameters. <br>",
                  "<br>",
                  "If the Empirical Bayes (EB) method is used, we expect to see: <br>",
                  "<ul>
                          <li>An <strong>estimated density distribution of the empirical values</strong> for both the location parameter gamma hat and the scale parameter delta hat (dotted line)</li>
                          <li>The <strong>EB-based prior distribution</strong> for both the location parameter gamma hat and the scale parameter delta hat (solid line)</li>
                    </ul>",
                  "If not, we expect to see: <br>",
                  "<ul>
                          <li>An <strong>estimated density distribution of the empirical values</strong> for both the location parameter gamma hat and the scale parameter delta hat (solid line)</li>
                    </ul>",
                  "If <strong>All</strong> batches are selected, line plots are colored by <strong>batch level</strong>."))
    })

    observeEvent(input$ComBat,{
      save_path = input$save_path
      if(length(input$interaction) > 0 & input$interaction != ""){
        interaction_enco = sapply(str_split(input$interaction, ",")[[1]], function(x) gsub("\\*", "\\,", x), USE.NAMES = FALSE)
        if(input$com_model == "gam"){
          smooth_int_type_enco = str_split(input$smooth_int_type, ",")[[1]]
        }else{smooth_int_type_enco = NULL}
      }else{
        interaction_enco = NULL
        smooth_int_type_enco = NULL}
      eb = ifelse(input$eb_control == "Yes", TRUE, FALSE)
      if(input$com_model == "gam"){
        smooth = input$smooth
      }else{smooth = NULL}
      if(input$com_model == "lmer"){
        random = input$random
      }else{random = NULL}

      if(input$ref_bat_select == "None"){
        ref_bat = NULL
      }else{ref_bat = input$ref_bat_select}

      # Set up harmonization progress notification
      msg = sprintf('Start harmonization progress')
      withProgress(message = msg, value = 0, {
        setProgress(0.5, 'Harmonizing...')
        if(input$com_type == "covfam"){
          score_eb = ifelse(input$score_eb_control == "Yes",  TRUE, FALSE)
          combat_result = combat_harm(result, type = input$com_model, random = input$random, smooth = input$smooth, interaction = interaction_enco, smooth_int_type = smooth_int_type_enco, family = input$com_type, ref.batch = ref_bat, predict = FALSE, object = NULL, reference = NULL, eb = eb, score_eb = score_eb, ...)
        }else{combat_result = combat_harm(result, type = input$com_model, random = input$random, smooth = input$smooth, interaction = interaction_enco, smooth_int_type = smooth_int_type_enco, family = input$com_type, ref.batch = ref_bat, predict = FALSE, object = NULL, reference = NULL, eb = eb, ...)}
        #assign("combat_result", combat_result, envir = .GlobalEnv)
        combat_result_s(combat_result)
        harm_df = combat_result$harmonized_df
        if(length(save_path) > 0 & save_path != ""){
          write.csv(harm_df, save_path, row.names = FALSE)
        }
        setProgress(1, 'Complete!')
      })

      showNotification('Harmonization Completed', type = "message")

      output$output_msg <- renderPrint({
        paste("DataFrame saved to:", input$save_path)
      })

      output$eb_location = shiny::renderPlot({
        if(combat_result$com_family == "comfam"){
          if(eb){
            min_x = combat_result$eb_df %>% filter(grepl("^gamma_*", type)) %>% pull(eb_values) %>% min()
            max_x = combat_result$eb_df %>% filter(grepl("^gamma_*", type)) %>% pull(eb_values) %>% max()
            if(input$batch_selection == "All"){
              ggplot(combat_result$eb_df %>% filter(grepl("^gamma_*", type), batch != "reference") %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                               type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Gamma", y = "Density", color = "Batch", linetype = "Estimate Type") +
                guides(color = "none") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }else{
              ggplot(combat_result$eb_df %>% filter(grepl("^gamma_*", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                                         type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Gamma", y = "Density", linetype = "Estimate Type") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }
          }else{
            min_x = combat_result$eb_df %>% filter(grepl("gamma_hat", type)) %>% pull(eb_values) %>% min()
            max_x = combat_result$eb_df %>% filter(grepl("gamma_hat", type)) %>% pull(eb_values) %>% max()
            if(input$batch_selection == "All"){
              ggplot(combat_result$eb_df %>% filter(grepl("gamma_hat", type), batch != "reference") %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                                type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Gamma", y = "Density", color = "Batch", linetype = "Estimate Type") +
                guides(color = "none") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }else{
              ggplot(combat_result$eb_df %>% filter(grepl("gamma_hat", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                                          type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Gamma", y = "Density", linetype = "Estimate Type") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }
          }
        }else if(combat_result$com_family == "covfam"){
          if(input$eb_check_type == "First-step ComBat"){
            if(eb){
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("^gamma_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^gamma_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^gamma_*", type), batch != "reference") %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                                 type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("^gamma_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^gamma_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^gamma_*", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                                           type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }else{
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("^gamma_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^gamma_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^gamma_hat", type), batch != "reference") %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                                   type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("^gamma_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^gamma_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^gamma_hat", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "gamma_prior" ~ "EB prior",
                                                                                                                                             type == "gamma_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }
          }else{
            if(score_eb){
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("^score_gamma_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^score_gamma_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^score_gamma_*", type), batch != "reference") %>% mutate(type = case_when(type == "score_gamma_prior" ~ "EB prior",
                                                                                                                                       type == "score_gamma_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("^score_gamma_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^score_gamma_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^score_gamma_*", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "score_gamma_prior" ~ "EB prior",
                                                                                                                                                 type == "score_gamma_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }else{
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("score_gamma_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("score_gamma_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("score_gamma_hat", type), batch != "reference") %>% mutate(type = case_when(type == "score_gamma_prior" ~ "EB prior",
                                                                                                                                        type == "score_gamma_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("score_gamma_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("score_gamma_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("score_gamma_hat", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "score_gamma_prior" ~ "EB prior",
                                                                                                                                                  type == "score_gamma_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Gamma", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }
          }
        }
      })

      output$eb_scale = shiny::renderPlot({
        if(combat_result$com_family == "comfam"){
          if(eb){
            min_x = combat_result$eb_df %>% filter(grepl("^delta_*", type)) %>% pull(eb_values) %>% min()
            max_x = combat_result$eb_df %>% filter(grepl("^delta_*", type)) %>% pull(eb_values) %>% max()
            if(input$batch_selection == "All"){
              ggplot(combat_result$eb_df %>% filter(grepl("^delta_*", type), batch != "reference") %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                               type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Delta", y = "Density", color = "Batch", linetype = "Estimate Type") +
                guides(color = "none") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }else{
              ggplot(combat_result$eb_df %>% filter(grepl("^delta_*", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                                         type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Delta", y = "Density", linetype = "Estimate Type") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }
          }else{
            if(input$batch_selection == "All"){
              min_x = combat_result$eb_df %>% filter(grepl("delta_hat", type)) %>% pull(eb_values) %>% min()
              max_x = combat_result$eb_df %>% filter(grepl("delta_hat", type)) %>% pull(eb_values) %>% max()
              ggplot(combat_result$eb_df %>% filter(grepl("delta_hat", type), batch != "reference") %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                                type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Delta", y = "Density", color = "Batch", linetype = "Estimate Type") +
                guides(color = "none") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }else{
              min_x = combat_result$eb_df %>% filter(grepl("delta_hat", type)) %>% pull(eb_values) %>% min()
              max_x = combat_result$eb_df %>% filter(grepl("delta_hat", type)) %>% pull(eb_values) %>% max()
              ggplot(combat_result$eb_df %>% filter(grepl("delta_hat", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                                          type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                geom_density() +
                xlim(min_x, max_x) +
                labs(x = "Delta", y = "Density", linetype = "Estimate Type") +
                theme(
                  axis.title.x = element_text(size = 12, face = "bold"),
                  axis.title.y = element_text(size = 12, face = "bold"),
                  axis.text.x = element_text(size = 12, face = "bold"),
                  axis.text.y = element_text(size = 12, face = "bold"),
                )
            }
          }
        }else if(combat_result$com_family == "covfam"){
          if(input$eb_check_type == "First-step ComBat"){
            if(eb){
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("^delta_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^delta_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^delta_*", type), batch != "reference") %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                                 type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("^delta_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^delta_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^delta_*", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                                           type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }else{
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("^delta_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^delta_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^delta_hat", type), batch != "reference") %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                                   type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("^delta_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^delta_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^delta_hat", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "delta_prior" ~ "EB prior",
                                                                                                                                             type == "delta_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }
          }else{
            if(score_eb){
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("^score_delta_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^score_delta_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^score_delta_*", type), batch != "reference") %>% mutate(type = case_when(type == "score_delta_prior" ~ "EB prior",
                                                                                                                                       type == "score_delta_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("^score_delta_*", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^score_delta_*", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^score_delta_*", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "score_delta_prior" ~ "EB prior",
                                                                                                                                                 type == "score_delta_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }else{
              if(input$batch_selection == "All"){
                min_x = combat_result$eb_df %>% filter(grepl("^score_delta_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^score_delta_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^score_delta_hat", type), batch != "reference") %>% mutate(type = case_when(type == "score_delta_prior" ~ "EB prior",
                                                                                                                                         type == "score_delta_hat" ~ "Emprical values")), aes(x = eb_values, color = batch, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", color = "Batch", linetype = "Estimate Type") +
                  guides(color = "none") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }else{
                min_x = combat_result$eb_df %>% filter(grepl("^score_delta_hat", type)) %>% pull(eb_values) %>% min()
                max_x = combat_result$eb_df %>% filter(grepl("^score_delta_hat", type)) %>% pull(eb_values) %>% max()
                ggplot(combat_result$eb_df %>% filter(grepl("^score_delta_hat", type), batch == input$batch_selection) %>% mutate(type = case_when(type == "score_delta_prior" ~ "EB prior",
                                                                                                                                                   type == "score_delta_hat" ~ "Emprical values")), aes(x = eb_values, linetype = type)) +
                  geom_density() +
                  xlim(min_x, max_x) +
                  labs(x = "Delta", y = "Density", linetype = "Estimate Type") +
                  theme(
                    axis.title.x = element_text(size = 12, face = "bold"),
                    axis.title.y = element_text(size = 12, face = "bold"),
                    axis.text.x = element_text(size = 12, face = "bold"),
                    axis.text.y = element_text(size = 12, face = "bold"),
                  )
              }
            }
          }
        }
      })
    })

    observeEvent(input$ComBat_model,{
      model_save_path = input$model_save_path
      combat_result = combat_result_s()
      msg = sprintf('Saving ComBat Model')
      withProgress(message = msg, value = 0, {
        setProgress(0.5, 'Saving ...')
        harm_model = combat_result$combat.object
        if(combat_result$com_family == "comfam"){
          harm_model$dat.combat = NULL
        }else{
          harm_model$dat.covbat = NULL
        }
        saveRDS(harm_model, file = model_save_path)
        setProgress(1, 'Complete!')
      })
      showNotification('ComBat Model Successfully Saved', type = "message")
    })

    output$output_msg_model <- renderPrint({
      paste("ComBat model saved to:", input$model_save_path)
    })

    ############### Statistical Tests #############################

    output$mul_adjustment = renderUI({
      HTML(paste0("<br>",
                  "<br>",
                  "All p-values have been adjusted by the <strong><span style='color: purple;'>Bonferroni</span></strong> method."))
    })

    output$test_batch_explain = shiny::renderUI({
      if(input$test_batch == "ANOVA"){
        HTML(print("<strong>Note</strong>: The one-way ANOVA test is a statistical technique used to assess whether there are significant differences among the means of three or more groups. It requires meeting several assumptions to obtain reliable results."))
      }else if(input$test_batch == "Kruskal-Wallis"){
        HTML(print("<strong>Note</strong>: The Kruskal-Wallis test is a non-parametric statistical test used to compare the medians of two or more groups, which serves as an alternative to the ANOVA test when the assumption of normality or equal variance is not met."))
      }else if(input$test_batch == "Kenward-Roger (liner mixed model)"){
        HTML(print("<strong>Note</strong>: The Kenward-Roger(KR) test is commonly employed in the context of linear mixed-effects models to estimate the degrees of freedom for hypothesis testing."))
      }
    })

    output$test_variance_explain = shiny::renderUI({
      if(input$test_variance == "Fligner-Killeen"){
        HTML(print("<strong>Note</strong>: The Fligner-Killeen (FK) test is a non-parametric alternative to Levene's and Bartlett's tests for assessing the homogeneity of variances. It doesn't rely on the assumption of normality."))
      }else if(input$test_variance== "Levene's Test"){
        HTML(print("<strong>Note</strong>: The Levene's test is a parametric test used to assess the equality of variances across multiple groups. It relies on the assumption of normality."))
      }else if(input$test_variance == "Bartlett's Test"){
        HTML(print("<strong>Note</strong>: The Bartlett's test is also a parametric test used for the same purpose as the Levene's test. Compared to the Levene's test, it is even more sensitive to departures from normality."))
      }
    })

    output$test_batch_ui = shiny::renderUI({
      fluidRow(
        column(width = 12,
               shinydashboard::box(
                 width = NULL,
                 DT::DTOutput("test_batch_table"))))
      #column(width = 6,
      #       shinydashboard::box(
      #         width = NULL,
      #         title = "P-value Distribution",
      #         shiny::plotOutput("test_batch_plot", height = "600px"))))
    })

    output$test_variance_ui = shiny::renderUI({
      fluidRow(
        column(width = 12,
               shinydashboard::box(
                 width = NULL,
                 DT::DTOutput("test_variance"))))
      #column(width = 6,
      #       shinydashboard::box(
      #         width = NULL,
      #         title = "P-value Distribution",
      #         shiny::plotOutput("test_variance_plot", height = "600px"))))
    })

    output$test_batch_mdmr = DT::renderDT({
      result$mdmr.summary %>% DT::datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                                  targets = "_all")))) %>% formatStyle(columns = c("p.value"),color = styleEqual(result$red, "red")) %>% formatStyle(
                                                                                    'sig',
                                                                                    target = 'row',
                                                                                    backgroundColor = styleEqual(c("*", "**", "***"), "lightyellow"))
    })

    output$mdmr_sig_text = renderUI({
      if(is.na(result$mdmr.summary$sig[2])){
        HTML("There's no significant global batch effect based on the MDMR test.")
      }else{HTML("There's a <strong>significant global batch effect</strong> based on the MDMR test.")}
    })

    output$mdmr_note = renderUI({
      HTML(paste0("<strong>Note</strong>: The Multivariate Distance Matrix Regression (MDMR) is utilized to evaluate the overall presence of batch effects within the dataset. <br>",
                  "<br>"))
    })


    output$test_batch_table = DT::renderDT({
      if(input$test_batch == "Kenward-Roger (liner mixed model)"){
        if(type == "lmer"){
          result$kr_test_df %>% dplyr::select(feature, p.value, sig) %>% datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                                                                         targets = "_all")))) %>% formatStyle(columns = c("p.value"),color = styleEqual(result$red, "red")) %>% formatStyle(
                                                                                                                           'sig',
                                                                                                                           target = 'row',
                                                                                                                           backgroundColor = styleEqual(c("*", "**", "***"), "lightyellow")
                                                                                                                         )}else{
                                                                                                                           result$kr_test_df %>% DT::datatable()
                                                                                                                         }
      }else if(input$test_batch== "ANOVA"){
        result$anova_test_df %>% dplyr::select(feature, p.value, sig) %>% datatable(options = list(columnDefs = list(list(className = 'dt-center', targets = "_all")))) %>% formatStyle(columns = c("p.value"),color = styleEqual(result$red, "red")) %>% formatStyle(
          'sig',
          target = 'row',
          backgroundColor = styleEqual(c("*", "**", "***"), "lightyellow"))
      }else if(input$test_batch == "Kruskal-Wallis"){
        result$kw_test_df %>% dplyr::select(feature, p.value, sig) %>% datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                                                                       targets = "_all")))) %>% formatStyle(columns = c("p.value"),color = styleEqual(result$red, "red")) %>% formatStyle(
                                                                                                                         'sig',
                                                                                                                         target = 'row',
                                                                                                                         backgroundColor = styleEqual(c("*", "**", "***"), "lightyellow"))
      }
    })

    #output$test_batch_plot = shiny::renderPlot({
    #  if(input$test_batch == "Kenward-Roger (liner mixed model)"){
    #    p_batch_df = result$kr_test_df
    #  }else if(input$test_batch== "ANOVA"){
    #    p_batch_df = result$anova_test_df
    #  }else if(input$test_batch == "Kruskal-Wallis"){
    #    p_batch_df = result$kw_test_df
    #  }
    #  if(input$test_batch != "Kenward-Roger (liner mixed model)" | type == "lmer"){
    #    ggplot(p_batch_df, aes(x = p.value.raw)) +
    #      geom_density(aes(y = ..scaled..), fill = "blue", alpha = 0.3) +
    #      geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
    #      labs( x = "p.value", y = "density")
    #  }
    #})

    output$test_variance = DT::renderDT({
      if(input$test_variance == "Fligner-Killeen"){
        result$fk_test_df %>% dplyr::select(feature, p.value, sig) %>% datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                                                                       targets = "_all")))) %>% formatStyle(columns = c("p.value"),color = styleEqual(result$red, "red")) %>% formatStyle(
                                                                                                                         'sig',
                                                                                                                         target = 'row',
                                                                                                                         backgroundColor = styleEqual(c("*", "**", "***"), "lightyellow"))
      }else if(input$test_variance == "Levene's Test"){
        result$lv_test_df %>% dplyr::select(feature, p.value, sig) %>% datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                                                                       targets = "_all")))) %>% formatStyle(columns = c("p.value"),color = styleEqual(result$red, "red")) %>% formatStyle(
                                                                                                                         'sig',
                                                                                                                         target = 'row',
                                                                                                                         backgroundColor = styleEqual(c("*", "**", "***"), "lightyellow"))
      }else if(input$test_variance == "Bartlett's Test"){
        if(nrow(result$bl_test_df)!=0){
          result$bl_test_df %>% dplyr::select(feature, p.value, sig) %>% datatable(options = list(columnDefs = list(list(className = 'dt-center',
                                                                                                                         targets = "_all")))) %>% formatStyle(columns = c("p.value"),color = styleEqual(result$red, "red")) %>% formatStyle(
                                                                                                                           'sig',
                                                                                                                           target = 'row',
                                                                                                                           backgroundColor = styleEqual(c("*", "**", "***"), "lightyellow"))}else{
                                                                                                                             result$bl_test_df %>% DT::datatable()
                                                                                                                           }
      }
    })

    #output$test_variance_plot = shiny::renderPlot({
    #  if(input$test_variance == "Fligner-Killeen"){
    #    p_variance_df = result$fk_test_df
    #  }else if(input$test_variance == "Levene's Test"){
    #    p_variance_df = result$lv_test_df
    #  }else if(input$test_variance == "Bartlett's Test"){
    #    p_variance_df = result$bl_test_df
    #  }
    #  ggplot(p_variance_df, aes(x = p.value.raw)) +
    #    geom_density(aes(y = ..scaled..), fill = "blue", alpha = 0.3) +
    #    geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
    #    labs( x = "p.value", y = "density")
    #})

    output$sig_pct_batch = shiny::renderText({
      if(input$test_batch == "Kenward-Roger (liner mixed model)"){
        if(type == "lmer"){
          n = nrow(result$kr_test_df)
          pct = 100 * (n - sum(is.na(result$kr_test_df$sig)))/n
          HTML(paste0("The percentage of significant features is: <strong>", round(pct,2), "%</strong>."))}else{
            HTML("The Kenward-Roger test is a modification of the degrees of freedom in linear mixed models. Not appropriate for the current type of model.")
          }
      }else if(input$test_batch == "ANOVA"){
        n = nrow(result$anova_test_df)
        pct = 100 * (n - sum(is.na(result$anova_test_df$sig)))/n
        HTML(paste0("The percentage of significant features is: <strong>", round(pct,2), "%</strong>."))
      }else if(input$test_batch == "Kruskal-Wallis"){
        n = nrow(result$kw_test_df)
        pct = 100 * (n - sum(is.na(result$kw_test_df$sig)))/n
        HTML(paste0("The percentage of significant features is: <strong>", round(pct,2), "</strong>%."))
      }
    })
    output$sig_pct_variance = shiny::renderText({
      if(input$test_variance == "Fligner-Killeen"){
        n = nrow(result$fk_test_df)
        pct = 100 * (n - sum(is.na(result$fk_test_df$sig)))/n
        HTML(paste0("The percentage of significant features is: <strong>", round(pct,2), "</strong>%."))
      }else if(input$test_variance == "Levene's Test"){
        n = nrow(result$lv_test_df)
        pct = 100 * (n - sum(is.na(result$lv_test_df$sig)))/n
        HTML(paste0("The percentage of significant features is: <strong>", round(pct,2), "</strong>%."))
      }else if(input$test_variance == "Bartlett's Test"){
        n = nrow(result$bl_test_df)
        if(n != 0){
          pct = 100 * (n - sum(is.na(result$bl_test_df$sig)))/n
          HTML(paste0("The percentage of significant features is: <strong>", round(pct,2), "</strong>%."))}else{
            HTML("Bartlett's Test failed due to less than 2 observations in each group.")
          }
      }
    })
  }

  shinyApp(ui = ui, server = server, enableBookmarking = "url")
}

utils::globalVariables(c("count", "cor_1", "cor_2", "percentage", "batch", "features", "eb_values", "type", "percentage (%)", "combat_result", "Principal_Component", "Variance_Explained", ".", "sig", "p.value.raw", "..scaled.."))

