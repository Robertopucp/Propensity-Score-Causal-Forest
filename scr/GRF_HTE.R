
## Course: Research Design and Impact Evaluation ##
## Assignment: Paper Modernization, Heterogeneous Treatment Effect based on Causal Forest
## @author: Roberto Mendoza 

# clean environment variables
rm(list = ls())

# clean plots
graphics.off()

# clean console

cat("\014")

# additional options
options(scipen = 999)      # No scientific notation
options(tigris_use_cache = FALSE)


# Loading packages #

library(pacman) 


p_load(tidyverse, # data managment
       grf,
       haven,
       cowplot,
       DiagrammeR,
       plotly,
       Rcpp,
       writexl,
       readxl,
       twang,
       openxlsx,
       policytree,
       maq,
       lfe,
       scatterplot3d,  
       devtools)

# Change working directory

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load dataset 

data_paper <- read_stata("data/raw/3.final/data_person_4.dta")

# Sub-sample 2005-2006, only treated individuals (ocu500==1  & codperso==1) and 
# creating the treatment variable as the interaction between ln_ya_w_g2 and d2 

data_paper_2005_2006 <- data_paper|>
                  dplyr::filter(between(year,2005, 2006)) |>
                  dplyr::filter(ocu500==1  & codperso==1) |>
                  mutate(treatment = ln_ya_w_g2*d2)


# Getting the Y, X and W (treatment variable , in this case is continuous)

Y <- select(data_paper_2005_2006, ln_y)  # selecting nominal income

# Selecting observable variables 

#  model.matrix is convenient to include categorical variables

X <- model.matrix(~ -1 + urban + as.factor(p507) + as.factor(ciuu_cod1) + schooling + age+  electricity+ water+ isfemale+ mieperho+ percepho,
                  data = data_paper_2005_2006)

# Selecting treatment variable 

W <- select(data_paper_2005_2006,
            treatment)


### Convert tibbles into vectors ###
x <- as.matrix(X)
y <- as.matrix(Y)
w <- as.matrix(W) 


### Create test and training sample ###
set.seed(10)

cases <- sample(seq_len(nrow(x)), round(nrow(x)*0.75))
train <- x[cases,]   # 75% training sample
test <- x[-cases,]    # 25% out-of-bag sample (test sample)

### Train  Causal Forest ###

cf <- causal_forest(x,
                    y, 
                    w,
                    honesty = TRUE,   # Honest Causal Forest
                    num.trees = 1000,   # 1000 trees
                    min.node.size = 50,  # target minimum leaf size; not a strict lower bound
                    ci.group.size = 4)


### Average Treatment Effect
ATE = average_treatment_effect (cf)
paste ("99% CI for the Conditional ATE: [", 
       round (ATE[1],6) - round ( qnorm (0.99)*ATE[2],6),
       "," , 
       round (ATE[1],6) + round ( qnorm (0.99)*ATE[2],6),
       "]"
       )


## Importance of variables in predicting treatment heterogeneity

cf |>
  variable_importance() |>
  as.data.frame() |>
  mutate(variable = colnames(cf$X.orig))|>
  arrange(desc(V1))


# Predicted treatment effect across observations 

preds.hat <- predict(cf, 
                     test, 
                     estimate.variance = TRUE)

# Standard error of predicted treatment effect

sigma.hat <- sqrt(preds.hat$variance.estimates)

preds.hat <- as.data.frame(preds.hat)
sigma.hat <- as.data.frame(sigma.hat)

test      <-as.data.frame(test)

test_data <- cbind(test,preds.hat,sigma.hat)


### Figure 3: Density Function###
p_density <- ggplot(test_data, aes(x = predictions)) + 
  
  geom_density(
    fill = "steelblue",
    color = "gray",
    alpha = 0.5,
    linewidth = 0.6
  ) +
  
  # línea vertical en 0 (muy importante para interpretación)
  geom_vline(xintercept = 0, linetype = "dashed",
             color = "black", linewidth = 0.6) +
  
  labs(
    x = "Estimated Treatment Effect",
    y = "Density"
  ) +
  
  theme_classic(base_size = 12) +
  theme(
    axis.line = element_line(color = "black", linewidth = 0.5),
    panel.grid = element_blank(),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 6)
  )

p_density


ggsave("images/treatment_effect_density.pdf", plot = p_density,
       width = 6, height = 4, device = cairo_pdf)


### Figure 4: Cumulative Density Function ### 
ggplot(test_data, aes(x=predictions)) + stat_ecdf(geom = "step", size=2) + 
  geom_hline(yintercept=0.25, linetype="dashed", color = "black")   +
  geom_hline(yintercept=0.75, linetype="dashed", color = "black")   +
  labs( x = "Estimated Treatment Effect", y = "Density") +
  theme_light()


### Plot CATE's and 3 covariable most predictive of treatment effect  ### 

# Schooling #

p1 <- ggplot(test_data, aes(x = schooling, y = predictions)) +        
  
  # points
  geom_point(color = "darkgray", size = 0.5) + 
  
  # smooth line + CI
  geom_smooth(method = "loess", se = TRUE,
              color = "steelblue", fill = "lightblue",
              linewidth = 0.6, alpha = 0.3) +
  
  # horizontal reference line at 0
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.6) +
  
  # labels
  labs(
    x = "Years of education",
    y = "Estimated Treatment Effect"
  ) +
  
  # clean theme similar to your image
  theme_classic(base_size = 12) +
  theme(
    axis.line = element_line(color = "black"),
    panel.grid = element_blank(),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 6)
  )

p1

# Exporting figure

ggsave("images/HTE_schooling_plot.pdf", plot = p1,
       width = 6, height = 4, device = cairo_pdf)



p2 <-ggplot(test_data, aes(x = age, y = predictions)) +        
  
  # points
  geom_point(color = "darkgray", size = 0.5) + 
  
  # smooth line + CI
  geom_smooth(method = "loess", se = TRUE,
              color = "steelblue", fill = "lightblue",
              linewidth = 0.6, alpha = 0.3) +
  
  # horizontal reference line at 0
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.6) +
  
  # labels
  labs(
    x = "Age",
    y = "Estimated Treatment Effect"
  ) +
  
  # clean theme similar to your image
  theme_classic(base_size = 12) +
  theme(
    axis.line = element_line(color = "black"),
    panel.grid = element_blank(),
    axis.title = element_text(size = 8),
    axis.text = element_text(size =6)
  )


p2

# Exporting figure

ggsave("images/HTE_age_plot.pdf", plot = p2,
       width = 6, height = 4, device = cairo_pdf)


p3 <- ggplot(test_data, aes(x = mieperho, y = predictions)) +        
  
  # points
  geom_point(color = "darkgray", size = 0.5) + 
  
  # smooth line + CI
  geom_smooth(method = "loess", se = TRUE,
              color = "steelblue", fill = "lightblue",
              linewidth = 0.6, alpha = 0.3) +
  
  # horizontal reference line at 0
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.6) +
  
  # labels
  labs(
    x = "Number of Household Members",
    y = "Estimated Treatment Effect"
  ) +
  
  # clean theme similar to your image
  theme_classic(base_size = 12) +
  theme(
    axis.line = element_line(color = "black"),
    panel.grid = element_blank(),
    axis.title = element_text(size = 8),
    axis.text = element_text(size = 6)
  )

p3


# Exporting figure

ggsave("images/HTE_hhsize_plot.pdf", plot = p3,
       width = 6, height = 4, device = cairo_pdf)


## Plot of Predictied Treatment Effect and their confidence intervel ##


plot_htes <- function(preds, ci = FALSE, z = 1.96) {
  
  if (is.null(preds$predictions) || length(preds$predictions) == 0) {
    stop("preds must include a non-empty vector called 'predictions'")
  }
  
  ord <- order(preds$predictions)
  
  plot_data <- data.frame(
    rank = seq_along(preds$predictions),
    predictions = preds$predictions[ord]
  )
  
  if (ci && !is.null(preds$variance.estimates) && length(preds$variance.estimates) > 0) {
    plot_data$se    <- sqrt(preds$variance.estimates[ord])
    plot_data$ymin  <- plot_data$predictions - z * plot_data$se
    plot_data$ymax  <- plot_data$predictions + z * plot_data$se
  }
  
  out <- ggplot(plot_data, aes(x = rank, y = predictions)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.6) +
    geom_point(color = "gray", size = 1.1) +
    labs(
      x = "Rank",
      y = "Estimated Treatment Effect"
    ) +
    theme_classic(base_size = 12) +
    theme(
      axis.line = element_line(color = "black", linewidth = 0.5),
      panel.grid = element_blank(),
      axis.title = element_text(size = 8),
      axis.text = element_text(size = 6)
    )
  
  if (ci && "ymin" %in% names(plot_data)) {
    out <- out +
      geom_errorbar(
        aes(ymin = ymin, ymax = ymax),
        width = 0,
        color = "#7db7e8",
        linewidth = 0.45
      ) +
      geom_point(color = "gray", size = 1)
  }
  
  return(out)
}


# Estimate and Confidence Interval for each test sample observation 

preds.hat <- predict(cf, 
                     test, 
                     estimate.variance = TRUE)

p_htes <- plot_htes(preds.hat, ci = TRUE)

ggsave(
  "images/HTE_test_sample_plot.pdf",
  plot = p_htes,
  width = 7,
  height = 4.5,
  device = cairo_pdf
)



















