# Tennis-Datasets-Decision-Tree-Model

Sklearn was used to build a decision tree to classify whether we should or shouldn't play tennis on a particular day. 
The predictor variables were outlook, temperature, humidity and wind. In tennis.csv the data was all categorical. 
In tennis_num.csv temperature and humidity values were numerical. For each model the decision tree rules 
and number of samples for a particular branch were printed. Further pythons’ seaborn library was used to 
display a confusion matrix of the correctly and incorrectly classified samples. For the decision tree model itself
a max depth of 42 was used, the minimum samples per leaf was 1. 
