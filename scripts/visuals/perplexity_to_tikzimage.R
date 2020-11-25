library(tikzDevice)
library(tidyverse)
library(ggplot2)

setwd("~/github/GCM")

# Combine with UBM data first, separate plot for positive, negative and combined perplexity
dat_czm <- read.csv("./results/CZM_perplexity.csv")
dat_ubm <- read.csv("./results/UBM_perplexity.csv")

dat_melted <- dat_czm %>%
  pivot_longer(!item_order) %>%
  mutate(model="CZM") %>%
  bind_rows(dat_ubm %>%
              pivot_longer(!item_order) %>%
              mutate(model="UBM"))


tikz("./paper_plots/overall_perplexity.tikz", width=4, height=3)
dat_melted %>%
  filter(name=="perplexity") %>%
ggplot(.)+
  geom_bar(aes(x=item_order, y=value, fill=model), stat="identity", position="dodge")+
  labs(x="Item position", y="Perplexity")+
  theme_classic()+
  scale_fill_grey()+
  scale_x_continuous(breaks= 1:10, labels=as.character(1:10))+
  guides(fill=guide_legend(title="Model"))
dev.off()


dat_melted %>%
  filter(name=="pos_perplexity") %>%
  ggplot(.)+
  geom_bar(aes(x=item_order, y=value, fill=model), stat="identity", position="dodge")

dat_melted %>%
  filter(name=="neg_perplexity") %>%
  ggplot(.)+
  geom_bar(aes(x=item_order, y=value, fill=model), stat="identity", position="dodge")







