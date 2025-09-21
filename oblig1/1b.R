# b) ----------------------------------------------------------------------

library(tidyverse)
library(ggcorrplot)
library(patchwork)

df <- read_delim("data/finance_data.csv",
                 delim = ";",
                 locale = locale(decimal_mark = ",")) %>% 
  rename(trading_day = `Trading day`)

returns <- df %>% 
  pivot_longer(-trading_day, 
               names_to = "asset",
               values_to = "price") %>% 
  group_by(asset) %>% 
  arrange(trading_day, .by_group = TRUE) %>% 
  mutate(return = log(price/lag(price))) %>% 
  drop_na()

# summary statistics: daily mean and volatility
stats <- returns %>% 
  group_by(asset) %>% 
  summarise(mu = mean(return),
            sigma = sd(return)) %>%
  mutate(mu_ann = mu * 252,
         sigma_ann = sigma * sqrt(252))

# wide matrix of returns
Rmat <- returns %>%
  select(trading_day, asset, return) %>%
  pivot_wider(names_from = asset, values_from = return) %>%
  select(-trading_day)

# matrices
cor_mat <- cor(Rmat, use = "pairwise.complete.obs")
cov_mat <- cov(Rmat, use = "pairwise.complete.obs") * 252  # annualized

# correlation heatmap
p_cor <- ggcorrplot(cor_mat, lab = TRUE, lab_size = 3, digits = 2,
                    colors = c("#cfd8dc", "#0072B2", "#d32f2f"),
                    outline.col = "white", show.legend = FALSE) +
  labs(title = "Correlation matrix") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0),
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# covariance heatmap
p_cov <- ggcorrplot(cov_mat, lab = TRUE, lab_size = 3, digits = 4,
                    colors = c("#cfd8dc", "#0072B2", "#d32f2f"),
                    outline.col = "white", show.legend = FALSE) +
  labs(title = "Ann. covariance matrix") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0),
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# side-by-side
p <- p_cor + p_cov + plot_layout(ncol = 2)

print(p)

# save
ggsave("plots/cor_cov_matrices.pdf", p, width = 10, height = 5, device = cairo_pdf)
