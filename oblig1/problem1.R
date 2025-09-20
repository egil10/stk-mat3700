

library(tidyverse)

df <- read_delim("data/finance_data.csv", delim = ";",
                 locale = locale(decimal_mark = ",")) %>% 
  rename("Trading_day" = `Trading day`)

# a)

df %>% 
  mutate(r1 = Trading_day/lag(Trading_day)/100,
         r2 = Energy/lag(Energy)/100)
