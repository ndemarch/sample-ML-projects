**Prediciting Stock Value Over Time** *(amazon_stock_prediction.ipynb)*

Isolating Amazon's stock value since its initial IOP, this project looks to explore the occurence of Bull and Bear markets, 
the recent volatility of the stock, along with predicting its value over time. To predict the stock value we use a time series 
analysis method and Long Short Term Memory (LSTM) neural network. To asses the accuracy, the final ~5 years of Amazon's stock value
are used to test how well we can predict the daily opening market value. Our model has an average accuracy of 94.8% percent when predicting
the opening value over ~5 years. For the first ~2 years our model is ~99% accurate and our precision starts to drop thereafter.
