SELECT order_id,
    review_comment_message,
    review_score
FROM `[PROJECT_ID].shop_data.order_reviews`
WHERE review_comment_message IS NOT NULL;