
rm(list=ls())
require('data.table')
require("RcppRoll")
require('forecast')
wd<-getwd()
df<-fread(paste0(paste0(wd , "/"), "test.csv"), stringsAsFactors = FALSE)

n_train<-150000
roll_stat_window = 512

# fun def
'%ni%' <- Negate('%in%')
name_vec<- function(text, names) {sapply(text, names,
                                         FUN = function(text, names) {grep(text, names, value=TRUE)}, 
                                         USE.NAMES = TRUE)}
name_paste <- function(pref,name) {paste(pref, name, sep=".")}
diff_log <- function(x) {`*`(diff(log(x)), 100)}
rcpp_roll <-function(x) RcppRoll::roll_sd(x, 
                                          n = roll_stat_window, 
                                          na.rm = TRUE, 
                                          fill = NA, 
                                          align = 'right')

# subset training data
df<-df[1:n_train,]

# sort dt by time
timestamp <- seq(ISOdate(2011,8,1), by = "secs", length.out = n_train)
df[, timestamp := timestamp]
keycol <- "timestamp"
data.table::setorderv(df, keycol, order=1)

# vec of colnames for easier matching
col_names<-colnames(df)

# subset colnames based on patterns
pat<-c("^x|^y", "^y", "^x", "^s")

# lagged and rolling features
name_roll <- name_paste("roll_sd", name_vec(pat, col_names)$`^x`)
name_lag <- name_paste("lag", name_vec(pat, col_names)$`^x`)

# factor features (not used yet)
name_f <- paste0("as.factor(",name_vec(pat, col_names)$`^s`,")")

# compute log-returns of  x's and y's  
df2<-df[, lapply(.SD, diff_log), .SDcols = name_vec(pat, col_names)$`^x|^y`]

# augment data with lagged features
df2[, `:=`( (name_lag), 
           shift(.SD, type='lag') ), 
        .SDcols = name_vec(pat, col_names)$`^x`]

end <- NROW(df)- (NROW(df2) - 
                  NROW(df2[!complete.cases(df2), ]))

# only complete cases and cbind factors and y's to them
df_data <- cbind(df[ -c(1:end), .SD, .SDcols = col_names %ni% name_vec(pat, col_names)$`^x|^y`], 
                  df2[complete.cases(df2), ])

# augment data with rolling features
df_data[,  `:=`( (name_roll), 
                 lapply(.SD, rcpp_roll) ), 
        .SDcols = name_vec(pat, col_names)$`^x` ]

# only complete cases 
df_data <- df_data[complete.cases(df_data), ]

# init
i <- 500
window <- 10000
Rsq<-list()
sse<-0
sum_real<-0
N_fit<-0
iter <- 0
# formulae
xreg <- c(name_vec(pat, col_names)$`^x`,name_lag,name_roll)
flae <- reformulate(termlabels = xreg, response = name_vec(pat, col_names)$`^y`)


while(i <= nrow(df_data)-window){
  
  fit<-tslm(flae, data=ts(df_data[1:i,]))
  
  Rsq[["R_sq_fit"]]<-rbind(Rsq[["R_sq_fit"]],summary(fit)$r.squared)
  
  new_x = df_data[(i+1):(i+window), .SD, .SDcols = xreg ]
  
  pred_y = data.table::data.table("y_pred" = forecast(fit, newdata=new_x)$mean[1:window])
  
  y_real<-df_data[(i+1):(i+window), .SD, .SDcols = name_vec(pat, col_names)$`^y` ]
  
  sum_real_init <- y_real[, colSums(.SD)]
  
  sum_real <- sum_real+sum_real_init
  
  sse<-sse+sum(`-`(y_real,pred_y)^2)
   
  N_fit<-N_fit+NROW(new_x) 

  i = i + window
  iter = iter + 1
  
  cat("Iteration:",iter,"Training window size:",i,"\n")
  
}     

#mean_real<-sum_real/N_fit
#nrmse = sqrt((sse/N_fit))/mean_real
nrmse = `/`(sqrt(`*`(sse,N_fit)),sum_real)
cat("Median R-sq across all folds:", sapply(Rsq, FUN = median),"\n",
    "Normalised RMSE", nrmse)

# save and load workspace .Rdata
#save.image(file = "cv_lm6.Rdata")
#load("~/Desktop/cv_lm6.Rdata")
