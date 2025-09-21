
# problem 1

library(tidyverse)


# a) ----------------------------------------------------------------------

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

# build fitted normal curves per asset
overlay <- stats %>% 
  rowwise() %>% 
  mutate(x = list(seq(min(returns$return), max(returns$return), length.out = 400)),
         y = list(dnorm(x, mean = mu, sd = sigma))) %>% 
  unnest(c(x, y))

# plot: histogram (empirical) + fitted normal
ggplot(returns, aes(x = return)) +
  geom_histogram(aes(y = ..density..), bins = 30,
                 fill = "grey80", color = "white") +
  geom_line(data = overlay, aes(x = x, y = y), color = "blue", linewidth = 0.8) +
  facet_wrap(~ asset, scales = "free") +
  theme_bw()

print(stats)

# b) ----------------------------------------------------------------------

library(tidyverse)
library(ggcorrplot)

# wide matrix of returns
Rmat <- returns %>%
  select(trading_day, asset, return) %>%
  pivot_wider(names_from = asset, values_from = return) %>%
  select(-trading_day)

# correlation and covariance matrices
cor_mat <- cor(Rmat, use = "pairwise.complete.obs")
cov_mat <- cov(Rmat, use = "pairwise.complete.obs")

# --- Plot correlation heatmap (Economist-like styling) ---
p <- ggcorrplot(cor_mat, 
                lab = TRUE, lab_size = 3, 
                colors = c("#cfd8dc", "#0072B2", "#d32f2f"), # grey???blue???red
                outline.col = "white") +
  labs(title = "Correlation between asset returns") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0),
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p)

print(round(cor_mat, 3))
print(round(cov_mat, 6))


# save
dir.create("plots", showWarnings = FALSE)
ggsave("plots/correlations.pdf", p, width = 6, height = 5, device = cairo_pdf)




