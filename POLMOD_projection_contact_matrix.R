library(conmat)
library(tidyverse)

polymod_contact_data <- get_polymod_contact_data()
polymod_survey_data <- get_polymod_population()

age_data <- data.frame(
  lower.age.limit = seq(0,90,5),
  population = c(288387,
                 311736,
                 336174,
                 320637,
                 311952,
                 335715,
                 374079,
                 345537,
                 315765,
                 302220,
                 322635,
                 304074,
                 296418,
                 252492,
                 213438,
                 163632,
                 107991,
                 57939,
                 33093
  )
)

age_tibble = conmat_population(age_data, age='lower.age.limit', population='population')

contact_model <- fit_single_contact_model(
  contact_data = polymod_contact_data,
  population = polymod_survey_data
)

synthetic_contact_NZ <- predict_contacts(
  model = contact_model,
  population = age_tibble,
  age_breaks = c(seq(0, 90, by = 5), Inf)
)

mat = predictions_to_matrix(synthetic_contact_NZ)

mat %>%
  autoplot()

# Force detailed balanced condition
mat_DB = 0.5 * (mat + (age_data$population %*% (1/t(age_data$population)) ) * t(mat) )

mat_DB %>%
  autoplot()

write.csv(mat_DB, "C:/Users/lonep/Documents/Uni/2025/Thesis/Age_ethnicity_stratification/contact_matrix_NZ.csv", row.names = FALSE)
