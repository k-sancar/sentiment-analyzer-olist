WITH CartTotals AS (
    SELECT 
        order_id,
        SUM(freight_value) AS total_freight
    FROM `[PROJECT_ID].shop_data.order_items`
    GROUP BY order_id
),

MainItems AS (
    SELECT 
        i.order_id,
        p.product_category_name AS main_product_category,
        p.product_description_lenght AS product_description_lenght,
        p.product_photos_qty AS product_photos_qty,
        P.product_weight_g AS product_weight_g,
        s.seller_state AS main_seller_state
    FROM `[PROJECT_ID].shop_data.order_items` i
    LEFT JOIN `[PROJECT_ID].shop_data.products` p 
        ON i.product_id = p.product_id
    LEFT JOIN `[PROJECT_ID].shop_data.sellers` s 
        ON i.seller_id = s.seller_id
    QUALIFY ROW_NUMBER() OVER(PARTITION BY i.order_id ORDER BY i.price DESC) = 1
),

Payments AS (
    SELECT 
        order_id,
        SUM(payment_value) AS total_payment_value,
        MAX(payment_installments) AS installments,
        STRING_AGG(payment_type, ', ') AS payment_type
    FROM `[PROJECT_ID].shop_data.order_payment`
    GROUP BY order_id
),

OrderSequence AS (
    SELECT 
        c.customer_unique_id,
        o.order_id AS order_id,
        o.order_purchase_timestamp,
        
        ROW_NUMBER() OVER(
            PARTITION BY c.customer_unique_id 
            ORDER BY o.order_purchase_timestamp
        ) AS order_sequence,
        
        LAG(o.order_purchase_timestamp) OVER(
            PARTITION BY c.customer_unique_id 
            ORDER BY o.order_purchase_timestamp
        ) AS previous_purchase_date
        
    FROM `[PROJECT_ID].shop_data.orders` o
    JOIN `[PROJECT_ID].shop_data.customers` c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
)

SELECT 
    c.customer_state,
    mi.main_seller_state,
    mi.main_product_category,
    mi.product_description_lenght,
    mi.product_photos_qty,
    mi.product_weight_g,
    pa.total_payment_value,
    pa.installments,
    ct.total_freight,
    o.order_status,
    os.order_sequence,
    DATE_DIFF(DATETIME(o.order_approved_at), DATE(o.order_purchase_timestamp), MINUTE) AS minutes_for_approval,
    DATE_DIFF(DATE(os.order_purchase_timestamp), DATE(os.previous_purchase_date), DAY) AS days_between_orders,    
    DATE_DIFF(DATE(o.order_estimated_delivery_date), DATE(o.order_purchase_timestamp), DAY) AS estimated_waiting_days,
    
    ROUND(
        SAFE_DIVIDE(
            DATE_DIFF(DATE(o.order_delivered_customer_date), DATE(o.order_purchase_timestamp), DAY), 
            DATE_DIFF(DATE(o.order_estimated_delivery_date), DATE(o.order_purchase_timestamp), DAY)  
        ), 2
    ) AS delivery_time_ratio,

    r.review_score

FROM `[PROJECT_ID].shop_data.orders` o
LEFT JOIN `[PROJECT_ID].shop_data.order_reviews` r
    ON o.order_id = r.order_id
LEFT JOIN `[PROJECT_ID].shop_data.customers` c
    ON c.customer_id = o.customer_id
LEFT JOIN Payments pa
    ON pa.order_id = o.order_id
LEFT JOIN CartTotals ct
    ON o.order_id = ct.order_id
LEFT JOIN MainItems mi
    ON o.order_id = mi.order_id
LEFT JOIN OrderSequence os
    ON os.order_id = o.order_id

WHERE o.order_status IN ('canceled', 'delivered')
    AND mi.main_product_category IS NOT NULL
    AND r.review_score IS NOT NULL;