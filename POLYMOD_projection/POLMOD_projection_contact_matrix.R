library(conmat)
library(tidyverse)

# Get POLYMOD data
polymod_contact_data <- get_polymod_contact_data()
polymod_survey_data <- get_polymod_population()

# Set up age structure of NZ
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

# Fit model
contact_model <- fit_single_contact_model(
  contact_data = polymod_contact_data,
  population = polymod_survey_data
)

synthetic_contact_NZ <- predict_contacts(
  model = contact_model,
  population = age_tibble,
  age_breaks = c(seq(0, 90, by = 5), Inf)
)

# Convert to matrix
mat = predictions_to_matrix(synthetic_contact_NZ)

mat %>%
  autoplot()

# Force detailed balanced condition
mat_DB = 0.5 * (mat + (age_data$population %*% (1/t(age_data$population)) ) * t(mat) )

mat_DB %>%
  autoplot()

# Save as csv
write.csv(mat_DB, "../data/contact_matrix_NZ.csv", row.names = FALSE)

labels <- seq(0,90,5)

png("Social_contact_rate_plot.png", width = 6, height = 4, units = "in", res = 300)

bp <- barplot(rowSums(mat),
        main = "Social contact rate of age groups (NZ estimate)",
        xlab = "Age group",
        ylab = "Contact rate",
        col = "#C00000",names.arg = rep("", length(labels)))

# Add only every second label (1st, 3rd, 5th...)
axis(1,
     at = bp[seq(1, length(labels), by = 2)],
     labels = labels[seq(1, length(labels), by = 2)])


dev.off()