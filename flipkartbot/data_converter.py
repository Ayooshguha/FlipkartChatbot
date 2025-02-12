import pandas as pd
from langchain_core.documents import Document

def dataconverter():
    # Fix 1: Use raw string for Windows path
    product_data = pd.read_csv(r"D:\flipkartbot\FlipkartChatbot\data\flipkart_product_review.csv")
    
    # Fix 2: Use consistent column names
    data = product_data[['product_title', 'review']]
    
    product_list = []

    # Fix 3: Changed 'df' to 'data' (variable name mismatch)
    for index, row in data.iterrows():
        object = {
            "product_name": row['product_title'],  # Fix 4: Changed 'product_title' to 'product_name'
            "review": row['review']
        }
        product_list.append(object)
    
    docs = []
    for object in product_list:
        metadata = {"product_name": object['product_name']}
        page_content = object['review']

        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    
    return docs