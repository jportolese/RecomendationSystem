# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Edx is the training set and validation is the test set. Develop algorithm on edx
names(edx)
names(validation)

# how many records in each
nrow(edx)
nrow(validation)

#Start with the simplest model that assumes all unknown ratings are just the average of all ratings
mu_hat <- mean(edx$rating)
mu_hat

# Average Rating for all movies in the edx training set is 3.51

# Based on this very simple model what is the RMSE?  We need a function that calculates the RMSE based 
# on a vector of ratings and their corresponding predictors

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

# 1.06 is pretty bad.  We need to consider movie and user variation in the model

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu_hat))

head(movie_avgs)

# movie_avgs is a tibble of the residuals of each movies rating when compared to the average
# rating for all movies.

# We can use this to work on the model but what about individual user rating variations.  Each user
# is either a harsh critic or they like everything so while we're at it lets deal with this variation.

user_avgs <- edx %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat))

head(user_avgs)

#Like above this creates a tibble of residuals of each users rating when compared to the average 
# rating for all movies

# So our model now can take into account user and movie variation with these two datasets

predicted_ratings <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(prediction = mu_hat + b_i + b_u) %>%
  .$prediction

better_model <- RMSE(edx$rating, predicted_ratings)
better_model

# by including the movie and user variation into the model we improve the model 
#significantely 0.885 from 1.06
# We can do better.

movie_titles <- edx %>%
  select(movieId, title) %>%
  distinct()


# Top ten largest and smallest residuals show that they are movies that are obscure
# and not rated by very many users.


# Ten worst movies and total reviews
edx %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title,b_i,n) %>%
  slice(1:10) %>%
  knitr::kable()

#ten best movies and total reviews
edx %>% count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title,b_i,n) %>%
  slice(1:10) %>%
  knitr::kable()
  

# apply regularization for both user and movie effect
# identify the lambda by cross validation using only the training set (edx)

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/ (n()+l))
  
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId)  %>%
    summarize(b_u = sum(rating - b_i - mu)/ (n()+l))
  
  predicted_ratings <- edx %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    mutate(prediction = mu + b_i + b_u) %>%
    .$prediction
  
  return(RMSE(predicted_ratings, edx$rating))
  
})
    
#plot the rmses for each lambda

qplot(lambdas, rmses)


# find min
lambda <- lambdas[which.min(rmses)]
lambda

#apply the min
min(rmses)



# apply lambda to the test set (validation)
# Get final RMSE on the validation set

lambda = 0.5

mu <- mean(validation$rating)

b_i <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/ (n()+lambda))

b_u <- validation %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId)  %>%
  summarize(b_u = sum(rating - b_i - mu)/ (n()+lambda))

predicted_ratings <- validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  mutate(prediction = mu + b_i + b_u) %>%
  .$prediction

bestModel_wReg <- RMSE(predicted_ratings, validation$rating)
bestModel_wReg

# final RMSE on validation set is 0.8260111








