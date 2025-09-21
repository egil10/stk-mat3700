
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
