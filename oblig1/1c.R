# c) ----------------------------------------------------------------------

library(tidyverse)
library(ggcorrplot)
library(patchwork)
library(tidyverse)
library(quadprog)

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

# ----- wide matrix of daily returns -----
Rmat <- returns %>%
  select(trading_day, asset, return) %>%
  pivot_wider(names_from = asset, values_from = return) %>%
  select(-trading_day) %>% as.matrix()

# ----- sample moments (daily) -----
mu    <- colMeans(Rmat, na.rm = TRUE)            # expected return
Sigma <- cov(Rmat, use = "pairwise.complete.obs")# covariance matrix
n     <- ncol(Rmat)
assets <- colnames(Rmat)

# ----- helper: annualize -----
ann_mu    <- function(x) x * 252
ann_sigma <- function(x) x * sqrt(252)

# ----- efficient frontier (long-only) -----
targets <- seq(min(mu), max(mu), length.out = 60)

ef <- map_dfr(targets, function(m) {
  Dmat <- 2 * Sigma
  dvec <- rep(0, n)
  # constraints: 1'w = 1  AND  mu'w = m  (equalities),  w >= 0 (inequalities)
  Amat <- cbind(rep(1, n), mu, diag(n))  # columns are constraints
  bvec <- c(1, m, rep(0, n))
  sol  <- solve.QP(Dmat, dvec, Amat, bvec, meq = 2)
  w    <- sol$solution
  tibble(mu = sum(w * mu),
         sigma = sqrt(drop(t(w) %*% Sigma %*% w)))
})

# ----- global minimum-variance portfolio (long-only) -----
Dmat <- 2 * Sigma
dvec <- rep(0, n)
Amat <- cbind(rep(1, n), diag(n))  # 1'w = 1 (equality), w >= 0
bvec <- c(1, rep(0, n))
gmv  <- solve.QP(Dmat, dvec, Amat, bvec, meq = 1)$solution
gmv_pt <- tibble(mu = sum(gmv * mu),
                 sigma = sqrt(drop(t(gmv) %*% Sigma %*% gmv)))

# ----- individual stocks as (sigma, mu) points -----
stocks_pts <- tibble(asset = assets,
                     mu = as.numeric(mu),
                     sigma = sqrt(diag(Sigma)))

# ----- plot -----
ggplot() +
  geom_path(data = ef %>% mutate(mu = ann_mu(mu), sigma = ann_sigma(sigma)),
            aes(x = sigma, y = mu), linewidth = 1, color = "#0072B2") +
  geom_point(data = stocks_pts %>% mutate(mu = ann_mu(mu), sigma = ann_sigma(sigma)),
             aes(x = sigma, y = mu), size = 2.5, color = "#333333") +
  geom_text(data = stocks_pts %>% mutate(mu = ann_mu(mu), sigma = ann_sigma(sigma)),
            aes(x = sigma, y = mu, label = asset), nudge_y = 0.01, size = 3) +
  geom_point(data = gmv_pt %>% mutate(mu = ann_mu(mu), sigma = ann_sigma(sigma)),
             aes(x = sigma, y = mu), size = 3, color = "#d32f2f") +
  labs(x = "risk ??", y = "return ??",
       title = "Efficient frontier (long-only) with assets and GMV") +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank()
  )

# annualize stats for each stock
stocks_table <- stocks_pts %>%
  mutate(mu_ann = ann_mu(mu),
         sigma_ann = ann_sigma(sigma)) %>%
  select(asset, mu_ann, sigma_ann)

# GMV point annualized
gmv_table <- gmv_pt %>%
  mutate(mu_ann = ann_mu(mu),
         sigma_ann = ann_sigma(sigma)) %>%
  mutate(asset = "GMV") %>%
  select(asset, mu_ann, sigma_ann)

# Combine
# assume risk-free rate = 0
all_table <- bind_rows(
  stocks_pts %>%
    mutate(mu_ann    = ann_mu(mu),
           sigma_ann = ann_sigma(sigma),
           sharpe    = mu_ann / sigma_ann) %>%
    select(asset, mu_ann, sigma_ann, sharpe),
  gmv_pt %>%
    mutate(mu_ann    = ann_mu(mu),
           sigma_ann = ann_sigma(sigma),
           sharpe    = mu_ann / sigma_ann,
           asset     = "GMV") %>%
    select(asset, mu_ann, sigma_ann, sharpe)
)

print(all_table)



# Save
ggsave("plots/efficient_frontier.pdf", width = 12, height = 6, device = cairo_pdf)
