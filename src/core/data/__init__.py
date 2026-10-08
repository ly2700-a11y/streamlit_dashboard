"""
Data Module - 数据加载和预处理

包含内容:
    - loader.py: 数据加载器
    - preprocessor.py: 数据预处理
    - contract_manager.py: 期货合约管理

主要类:
    - DataLoader: 通用数据加载器
    - DataPreprocessor: 数据预处理
    - ContractManager: 合约管理

使用示例:
    from src.core.data import DataLoader, DataPreprocessor
    
    loader = DataLoader(data_dir="data/")
    df = loader.load_price_data("Commodity", "2011-01-01", "2019-12-31")
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.process(df)
"""

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "ContractManager",
]
