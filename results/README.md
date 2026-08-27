# Results

Model performance improved substantially when trained on full article text compared with shorter-form political text datasets. The **Articles** dataset produced the strongest overall results, achieving an **R² of approximately 0.74**, with an **RMSE of 0.60** and **MAE of 0.42**. This suggests that the BERT regression model was able to explain a large share of the variation in the continuous ideological scores when given longer, information-rich article text.

Performance was weaker on the other datasets. **MITweet + Articles** achieved an R² of roughly **0.37**, while **MITweet** and **QBias** produced substantially lower R² values of about **0.15** and **0.12**, respectively. Although the article-only model had higher absolute RMSE and MAE values, these metrics are not directly comparable across datasets without accounting for differences in target-score scale.

Overall, the results suggest that **long-form article text provides substantially stronger signal for predicting continuous ideological position than shorter or more limited text sources**. The improvement in R² is especially important, indicating that the model captures considerably more of the underlying variation when trained on full news articles.

<img width="767" height="457" alt="2026-08-26_23-32-30 (1)" src="https://github.com/user-attachments/assets/bdcb0b06-3cb5-4588-8fdf-c1e67ca809fd" />
