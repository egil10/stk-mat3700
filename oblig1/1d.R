# d) ----------------------------------------------------------------------

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

# pick one asset to drop
drop_asset <- "Tech"

# build return matrix with all 5
Rmat5 <- returns %>%
  select(trading_day, asset, return) %>%
  pivot_wider(names_from = asset, values_from = return) %>%
  select(-trading_day) %>% as.matrix()

# build return matrix with 4 assets (drop chosen one)
Rmat4 <- Rmat5[, !(colnames(Rmat5) %in% drop_asset)]

# function to compute efficient frontier (long-only)
get_frontier <- function(Rmat, n_points = 60) {
  mu    <- colMeans(Rmat, na.rm = TRUE)
  Sigma <- cov(Rmat, use = "pairwise.complete.obs")
  n     <- ncol(Rmat)
  targets <- seq(min(mu), max(mu), length.out = n_points)
  
  ef <- map_dfr(targets, function(m) {
    sol <- solve.QP(
      Dmat = 2 * Sigma,
      dvec = rep(0, n),
      Amat = cbind(rep(1, n), mu, diag(n)),
      bvec = c(1, m, rep(0, n)),
      meq  = 2
    )
    w <- sol$solution
    tibble(mu = sum(w * mu),
           sigma = sqrt(drop(t(w) %*% Sigma %*% w)))
  })
  ef %>% mutate(mu = mu*252, sigma = sigma*sqrt(252))
}

ef5 <- get_frontier(Rmat5)
ef4 <- get_frontier(Rmat4)
# --- GMV (long-only) helper
gmv_longonly <- function(Sigma) {
  n <- ncol(Sigma)
  solve.QP(Dmat = 2*Sigma, dvec = rep(0,n),
           Amat = cbind(rep(1,n), diag(n)),
           bvec = c(1, rep(0,n)), meq = 1)$solution
}

# moments (daily)
mu5    <- colMeans(Rmat5);  S5 <- cov(Rmat5)
mu4    <- colMeans(Rmat4);  S4 <- cov(Rmat4)

# GMV weights & points (annualized)
w_gmv5 <- gmv_longonly(S5)
w_gmv4 <- gmv_longonly(S4)

gmv5 <- tibble(mu = sum(w_gmv5*mu5),
               sigma = sqrt(drop(t(w_gmv5)%*%S5%*%w_gmv5))) %>%
  mutate(mu = mu*252, sigma = sigma*sqrt(252)) %>%
  mutate(sharpe = mu/sigma, label = "GMV (5)")

gmv4 <- tibble(mu = sum(w_gmv4*mu4),
               sigma = sqrt(drop(t(w_gmv4)%*%S4%*%w_gmv4))) %>%
  mutate(mu = mu*252, sigma = sigma*sqrt(252)) %>%
  mutate(sharpe = mu/sigma, label = paste0("GMV (4, -", drop_asset, ")"))

# Sharpe on frontiers (already annualized in your get_frontier)
ef5_s <- ef5 %>% mutate(sharpe = mu/sigma)
ef4_s <- ef4 %>% mutate(sharpe = mu/sigma)

# --- Plot: frontiers + stocks + both GMVs
ggplot() +
  geom_path(data = ef5, aes(sigma, mu), color = "#0072B2", linewidth = 1) +
  geom_path(data = ef4, aes(sigma, mu), color = "#D32F2F", linewidth = 1, linetype = "dashed") +
  geom_point(data = stocks_pts, aes(sigma, mu), size = 2.5, color = "#333333") +
  geom_text(data = stocks_pts, aes(sigma, mu, label = asset), nudge_y = 0.01, size = 3) +
  geom_point(data = gmv5, aes(sigma, mu), color = "#0072B2", size = 3) +
  geom_point(data = gmv4, aes(sigma, mu), color = "#D32F2F", size = 3) +
  labs(x = "risk ??", y = "return ??",
       title = "Efficient frontier: 5 vs 4 assets with MVP",
       caption = paste("Red dashed frontier excludes", drop_asset)) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0),
        panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank())

ggsave("plots/efficient_frontier_4.pdf", width = 12, height = 6, device = cairo_pdf)

# --- Prints: stocks table (+ Sharpe), EF5 (+ Sharpe), EF4 (+ Sharpe), GMVs
stocks_table <- stocks_pts %>%
  transmute(asset, mu_ann = mu, sigma_ann = sigma, sharpe = mu_ann/sigma_ann)

gmv_table <- bind_rows(
  tibble(asset = "MVP", mu_ann = gmv5$mu, sigma_ann = gmv5$sigma, sharpe = gmv5$sharpe),
  tibble(asset = paste0("MVP without Tech"),
         mu_ann = gmv4$mu, sigma_ann = gmv4$sigma, sharpe = gmv4$sharpe)
)

cat("\n--- Stocks (annualized) ---\n"); print(stocks_table)
cat("\n--- Frontier (5 assets) with Sharpe ---\n"); print(ef5_s)
cat("\n--- Frontier (4 assets) with Sharpe ---\n"); print(ef4_s)
cat("\n--- GMV summary ---\n"); print(gmv_table)
